import XCTest
@testable import KwipMLCore
@testable import KwipMLApp

final class LayerNormLearningWorkerTests: XCTestCase {
    @MainActor func testAllCoreTasksGradeActualWorkerMeasurementsAndRejectWrongEpsilon() async throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the explicit development runtime for normalization assessment interoperability.")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:URL(fileURLWithPath:path),allowDevelopment:true)
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let adapter=LearningAssessmentAdapter();try adapter.configureExecution(workspace:directory,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil));adapter.discoverRuntime(runtime);try await finished(adapter)
        let version=try XCTUnwrap(adapter.runtimeVersions["pytorch"],adapter.error ?? "Missing runtime")
        for task in LayerNormLearningTask.allCases {
            for seed:UInt64 in [0,17] {
                let plan=try LayerNormAssessment.plan(task:task,runtimeVersion:version,seed:seed)
                var graph=GraphDocument(nodes:[.init(id:"input",type:"input"),.init(id:"norm",type:"layer_norm",config:["normalized_shape":.array(task.normalizedShape.map {.integer(Int64($0))}),"eps":.number(task.epsilon),"elementwise_affine":.bool(task.scale != nil),"bias":.bool(task.bias != nil)]),.init(id:"output",type:"output")],edges:[.init(source:"input",target:"norm"),.init(source:"norm",target:"output")])
                var parameters:[String:[String:LearningTensor]]=[:]
                if let scale=task.scale,let bias=task.bias {parameters["norm"]=["weight":scale,"bias":bias]}
                for correct in [true,false] {
                    graph.nodes[1].config["eps"] = .number(correct ? task.epsilon : 0.5)
                    let submission=try LayerNormLearningSubmission(graph:graph,parameters:parameters,gradientNodeID:task == .affine ? "norm" : nil)
                    let bytes=try submission.encoded()
                    XCTAssertEqual(try LayerNormLearningSubmission.decode(bytes),submission)
                    adapter.assess(plan:plan,submission:bytes,runtime:runtime);try await finished(adapter)
                    XCTAssertNil(adapter.error)
                    XCTAssertEqual(adapter.grade?.status,correct ? .passed : .failed,"\(task.rawValue): \(adapter.grade?.summary ?? "Missing grade")")
                    XCTAssertEqual(adapter.grade?.outcomes.count,task == .affine ? 4 : 2)
                    let evidence=try XCTUnwrap(adapter.evidence)
                    XCTAssertNil(evidence.backendError)
                    XCTAssertEqual(evidence.submissionDigest,learningHash(bytes))
                    XCTAssertEqual(evidence.fixtures.count,4)
                    XCTAssertEqual(evidence.runtime,plan.runtime)
                    let projectID=UUID().uuidString
                    let mode:LearningMode=seed == 0 ? .guided : .assessment
                    let receipt=try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
                    let storage=directory.appendingPathComponent("saved-receipts")
                    try TensorLearningReceiptRepository(directory:storage).save(receipt)
                    let restored=try XCTUnwrap(TensorLearningReceiptRepository(directory:storage).latest(projectID:projectID))
                    XCTAssertEqual(try restored.verifiedReplay(expectedProjectID:projectID),plan)
                    XCTAssertEqual(restored.attempt.grade,adapter.grade)
                    let projection=restored.formativeProjection(expectedProjectID:projectID,currentSubmission:bytes,currentMode:.guided)
                    XCTAssertEqual(projection != nil,mode == .guided)
                    XCTAssertNil(restored.formativeProjection(expectedProjectID:projectID,currentSubmission:bytes,currentMode:.assessment))
                }
            }
        }
    }
    @MainActor private func finished(_ adapter:LearningAssessmentAdapter) async throws {
        let deadline=Date().addingTimeInterval(20)
        while adapter.isRunning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(adapter.isRunning,"Worker deadline exceeded")
    }
}
