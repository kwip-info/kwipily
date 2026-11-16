import XCTest
@testable import KwipMLCore
@testable import KwipMLApp

final class MatrixLearningWorkerTests:XCTestCase {
    @MainActor func testSwiftReferenceGradesRealMatrixWorkerAcrossSeeds()async throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable development runtime for matrix interoperability.")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:URL(fileURLWithPath:path),allowDevelopment:true)
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let adapter=LearningAssessmentAdapter();try adapter.configureExecution(workspace:directory,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil));adapter.discoverRuntime(runtime);try await finished(adapter)
        let version=try XCTUnwrap(adapter.runtimeVersions["pytorch"],adapter.error ?? "Missing runtime")
        let graph=GraphDocument(nodes:[.init(id:"a",type:"input",config:["fixture":.string("left")]),.init(id:"b",type:"input",config:["fixture":.string("right")]),.init(id:"m",type:"matmul"),.init(id:"o",type:"output")],edges:[.init(source:"a",target:"m",targetPort:"a"),.init(source:"b",target:"m",targetPort:"b"),.init(source:"m",target:"o")])
        for task in MatrixLearningTask.allCases {
            for seed:UInt64 in [0,17] {
                let plan=try MatrixAssessment.plan(task:task,runtimeVersion:version,seed:seed)
                let bytes=try MatrixLearningSubmission(graph:graph).encoded()
                adapter.assess(plan:plan,submission:bytes,runtime:runtime);try await finished(adapter)
                XCTAssertNil(adapter.error)
                XCTAssertEqual(adapter.grade?.status,.passed,adapter.grade?.summary ?? "Missing grade")
                let evidence=try XCTUnwrap(adapter.evidence)
                XCTAssertNil(evidence.backendError)
                XCTAssertEqual(evidence.fixtures.count,4)
                XCTAssertEqual(evidence.submissionDigest,learningHash(bytes))
                XCTAssertEqual(adapter.grade?.outcomes.count,3)
                let projectID=UUID().uuidString
                let receipt=try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:.guided,submissionData:bytes,plan:plan,evidence:evidence)
                let repository=TensorLearningReceiptRepository(directory:directory.appendingPathComponent("receipts"))
                try repository.save(receipt)
                let restored=try XCTUnwrap(repository.latest(projectID:projectID))
                XCTAssertEqual(try restored.verifiedReplay(expectedProjectID:projectID),plan)
                let projection=try XCTUnwrap(restored.formativeProjection(expectedProjectID:projectID,currentSubmission:bytes,currentMode:.guided))
                let shown=try XCTUnwrap(projection.fixtures.first)
                XCTAssertEqual(shown.inputLabel,"Left operand A")
                XCTAssertEqual(shown.secondaryInput,plan.fixtures[0].inputs["right"])
                XCTAssertEqual(Set(TensorMeasuredOutput.available(in:shown).map(\.rawValue)),["output","leftGradient","rightGradient"])
                XCTAssertNil(restored.formativeProjection(expectedProjectID:projectID,currentSubmission:bytes,currentMode:.assessment))
                XCTAssertNil(restored.formativeProjection(expectedProjectID:projectID,currentSubmission:Data(),currentMode:.guided))
                var altered=restored;altered.fixtureSeed=seed &+ 1
                XCTAssertThrowsError(try altered.verifiedReplay(expectedProjectID:projectID))
                altered=restored;altered.evidence.fixtures[0].tensors.removeValue(forKey:"rightGradient")
                XCTAssertThrowsError(try altered.verifiedReplay(expectedProjectID:projectID))
                let hidden=try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:.assessment,submissionData:bytes,plan:plan,evidence:evidence)
                XCTAssertNil(hidden.formativeProjection(expectedProjectID:projectID,currentSubmission:bytes,currentMode:.guided))
                // A valid additional transpose changes results or shape; never silently repair it.
                var wrong=graph
                wrong.nodes.append(.init(id:"t",type:"transpose",config:["dim0":-1,"dim1":-2]))
                wrong.edges[2].target="t";wrong.edges.append(.init(source:"t",target:"o"))
                adapter.assess(plan:plan,submission:try MatrixLearningSubmission(graph:wrong).encoded(),runtime:runtime);try await finished(adapter)
                XCTAssertNotEqual(adapter.grade?.status,.passed)
                XCTAssertNotNil(adapter.grade,"Wrong graph must produce an assessment result, not lose the request")
            }
        }
    }
    @MainActor private func finished(_ adapter:LearningAssessmentAdapter)async throws {
        let deadline=Date().addingTimeInterval(20)
        while adapter.isRunning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(adapter.isRunning,"Worker deadline exceeded")
    }
}
