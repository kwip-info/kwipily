import XCTest
@testable import KwipMLCore
@testable import KwipMLApp

final class BilinearLearningWorkerTests:XCTestCase {
    @MainActor func testSwiftReferenceGradesRealBilinearWorkerAcrossSeeds()async throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable development runtime for Bilinear interoperability.")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:URL(fileURLWithPath:path),allowDevelopment:true)
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let adapter=LearningAssessmentAdapter();try adapter.configureExecution(workspace:directory,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil));adapter.discoverRuntime(runtime);try await finished(adapter)
        let version=try XCTUnwrap(adapter.runtimeVersions["pytorch"],adapter.error ?? "Missing runtime")
        for task in BilinearLearningTask.allCases {
            let roles = ["left", "right", "weight"] + (task == .withBias ? ["bias"] : [])
            let ports = ["a", "b", "weight"] + (task == .withBias ? ["bias"] : [])
            var nodes = roles.map { GraphNode(id: $0, type: "input", config: ["fixture": .string($0)]) }
            nodes += [.init(id: "m", type: "bilinear", config: ["bias": .bool(task == .withBias)]), .init(id: "o", type: "output")]
            let edges = zip(roles, ports).map { GraphEdge(source: $0.0, target: "m", targetPort: $0.1) } + [GraphEdge(source: "m", target: "o")]
            let graph = GraphDocument(nodes: nodes, edges: edges)
            for seed:UInt64 in [0,17] {
                let plan=try BilinearAssessment.plan(task:task,runtimeVersion:version,seed:seed)
                let bytes=try BilinearLearningSubmission(graph:graph).encoded()
                adapter.assess(plan:plan,submission:bytes,runtime:runtime);try await finished(adapter)
                XCTAssertNil(adapter.error)
                XCTAssertEqual(adapter.grade?.status,.passed,adapter.grade?.summary ?? "Missing grade")
                let evidence=try XCTUnwrap(adapter.evidence)
                XCTAssertNil(evidence.backendError)
                XCTAssertEqual(evidence.fixtures.count,4)
                XCTAssertEqual(evidence.submissionDigest,learningHash(bytes))
                XCTAssertEqual(adapter.grade?.outcomes.count,task == .withBias ? 5 : 4)
                let projectID=UUID().uuidString
                let receipt=try TensorLearningReceipt(projectID:projectID,operation:task.rawValue,mode:.guided,submissionData:bytes,plan:plan,evidence:evidence)
                let repository=TensorLearningReceiptRepository(directory:directory.appendingPathComponent("receipts"))
                try repository.save(receipt)
                let restored=try XCTUnwrap(repository.latest(projectID:projectID))
                XCTAssertEqual(try restored.verifiedReplay(expectedProjectID:projectID),plan)
                let projection = try XCTUnwrap(restored.formativeProjection(expectedProjectID: projectID, currentSubmission: bytes, currentMode: .guided))
                let shown = try XCTUnwrap(projection.fixtures.first)
                XCTAssertEqual(shown.sharedParameters["weight"], plan.fixtures[0].inputs["weight"])
                XCTAssertEqual(shown.sharedParameters["bias"], plan.fixtures[0].inputs["bias"])
                XCTAssertEqual(shown.secondaryInput, plan.fixtures[0].inputs["right"])
                XCTAssertEqual(Set(TensorMeasuredOutput.available(in: shown).map(\.rawValue)), Set(["output", "leftGradient", "rightGradient", "weightGradient"] + (task == .withBias ? ["biasGradient"] : [])))
                XCTAssertNil(restored.formativeProjection(expectedProjectID: projectID, currentSubmission: bytes, currentMode: .assessment))
                XCTAssertNil(restored.formativeProjection(expectedProjectID: projectID, currentSubmission: Data(), currentMode: .guided))
                let hidden = try TensorLearningReceipt(projectID: projectID, operation: task.rawValue, mode: .assessment, submissionData: bytes, plan: plan, evidence: evidence)
                XCTAssertNil(hidden.formativeProjection(expectedProjectID: projectID, currentSubmission: bytes, currentMode: .guided))
                var altered=restored;altered.fixtureSeed=seed &+ 1
                XCTAssertThrowsError(try altered.verifiedReplay(expectedProjectID:projectID))
                altered=restored;altered.evidence.fixtures[0].tensors.removeValue(forKey:"rightGradient")
                XCTAssertThrowsError(try altered.verifiedReplay(expectedProjectID:projectID))
                // Mode mismatch must be refused, never silently repaired.
                let other = try BilinearAssessment.plan(task: task == .withBias ? .withoutBias : .withBias, runtimeVersion: version, seed: seed)
                adapter.assess(plan:other,submission:bytes,runtime:runtime);try await finished(adapter)
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
