import XCTest
@testable import KwipMLCore
@testable import KwipMLApp

final class EmbeddingLearningWorkerTests:XCTestCase {
    @MainActor func testSwiftPlansGradeActualEmbeddingWorkerAndRejectWrongPolicies()async throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the development runtime for lookup assessment interoperability.")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:URL(fileURLWithPath:path),allowDevelopment:true)
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let adapter=LearningAssessmentAdapter();try adapter.configureExecution(workspace:directory,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil));adapter.discoverRuntime(runtime);try await finished(adapter)
        let version=try XCTUnwrap(adapter.runtimeVersions["pytorch"],adapter.error ?? "Missing runtime")
        for task in EmbeddingLearningTask.allCases {
            for seed:UInt64 in [0,17] {
                let plan=try EmbeddingAssessment.plan(task:task,runtimeVersion:version,seed:seed)
                for correct in [true,false] {
                    var config=try task.configuration.values
                    if !correct {
                        switch task {
                        case .gather:config["scale_grad_by_freq"]=true
                        case .paddingFrequency:config["padding_idx"] = .null
                        case .renormalize:config["max_norm"] = .null
                        }
                    }
                    let graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"lookup",type:"embedding",config:config),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"lookup"),.init(source:"lookup",target:"o")])
                    let submission=try EmbeddingLearningSubmission(graph:graph),bytes=try submission.encoded()
                    XCTAssertEqual(try EmbeddingLearningSubmission.decode(bytes),submission)
                    adapter.assess(plan:plan,submission:bytes,runtime:runtime);try await finished(adapter)
                    XCTAssertNil(adapter.error)
                    XCTAssertEqual(adapter.grade?.status,correct ? .passed : .failed,"\(task.rawValue): \(adapter.grade?.summary ?? "Missing grade")")
                    let evidence=try XCTUnwrap(adapter.evidence)
                    XCTAssertNil(evidence.backendError)
                    XCTAssertEqual(evidence.submissionDigest,learningHash(bytes))
                    XCTAssertEqual(evidence.fixtures.count,4)
                    XCTAssertEqual(evidence.runtime,plan.runtime)
                    XCTAssertEqual(adapter.grade?.outcomes.count,3)
                    let projectID=UUID().uuidString
                    let receipt=try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:.guided,submissionData:bytes,plan:plan,evidence:evidence)
                    let storage=TensorLearningReceiptRepository(directory:directory.appendingPathComponent("receipts"))
                    try storage.save(receipt)
                    let restored=try XCTUnwrap(storage.latest(projectID:projectID))
                    XCTAssertEqual(try restored.verifiedReplay(expectedProjectID:projectID),plan)
                    XCTAssertEqual(restored.attempt.grade,adapter.grade)
                    let projection=try XCTUnwrap(restored.formativeProjection(expectedProjectID:projectID,currentSubmission:bytes,currentMode:.guided))
                    let shown=try XCTUnwrap(projection.fixtures.first)
                    XCTAssertEqual(shown.inputLabel,"Integer selectors")
                    XCTAssertEqual(shown.sourceTable,plan.fixtures[0].inputs["table"])
                    XCTAssertEqual(Set(TensorMeasuredOutput.available(in:shown).map(\.rawValue)),["output","tableGradient","tableAfterForward"])
                    XCTAssertEqual(TensorMeasuredOutput.tableGradient.displayTensor(shown.measured["tableGradient"]!,observation:1)?.shape,[4,3])
                    XCTAssertNil(restored.formativeProjection(expectedProjectID:projectID,currentSubmission:bytes,currentMode:.assessment))
                    var changed=restored;changed.fixtureSeed=seed &+ 1
                    XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:projectID))
                    changed=restored;changed.evidence.fixtures[0].tensors.removeValue(forKey:"tableAfterForward")
                    XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:projectID))
                }
            }
        }
    }
    @MainActor private func finished(_ adapter:LearningAssessmentAdapter)async throws {
        let deadline=Date().addingTimeInterval(20)
        while adapter.isRunning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(adapter.isRunning,"Worker deadline exceeded")
    }
}
