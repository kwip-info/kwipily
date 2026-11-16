import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class BagHistoryReaderTests: XCTestCase {
    private func root() throws -> URL {
        let value=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:value,withIntermediateDirectories:true)
        guard let physical=realpath(value.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        defer {free(physical)}
        return URL(fileURLWithPath:String(cString:physical))
    }
    func testActualColdBinaryAndObjectiveArchivesRetainWholeBagResults() async throws {
        for objective in [false,true] {
            let root=try root();defer {try? FileManager.default.removeItem(at:root)}
            let saved=try await BagHistoryTestFixture.clone(objective:objective,into:root)
            let content=try XCTUnwrap(WorkflowHistoryReader.loadModelContent(saved.reference,root:root))
            let context=try XCTUnwrap(content.bagContext)
            let preview=try XCTUnwrap(content.bagPredictionPreview)
            XCTAssertEqual(context.mapping,saved.plan.bagInput)
            XCTAssertEqual(context.runID,saved.workerRunID)
            XCTAssertEqual(preview.runID,saved.workerRunID)
            XCTAssertEqual(preview.evaluationCount,context.preparation.validation.count)
            XCTAssertFalse(preview.rows.isEmpty)
            XCTAssertTrue(preview.rows.contains {$0.input.indices.isEmpty})
            XCTAssertTrue(preview.rows.allSatisfy {$0.input.perSampleWeights?.count == $0.input.indices.count})
            XCTAssertNil(content.predictionPreview);XCTAssertNil(content.objectivePredictionPreview)
            XCTAssertNotNil(content.activationPreviews)
            XCTAssertEqual(content.metrics.map(\.epoch),[1,2])
            if objective {
                XCTAssertNotNil(content.objectiveEvaluationContext);XCTAssertNotNil(content.objectiveEvaluation)
                XCTAssertTrue(content.metrics.allSatisfy {$0.accuracy == nil})
                XCTAssertTrue(preview.rows.allSatisfy {$0.label == nil && $0.predicted == nil && $0.targets["target"] != nil && $0.output != nil})
            } else {
                XCTAssertNil(content.objectiveEvaluationContext);XCTAssertNil(content.objectiveEvaluation)
                XCTAssertTrue(preview.rows.allSatisfy {$0.label != nil && $0.predicted != nil})
            }
            XCTAssertEqual(content.code,try String(contentsOf:saved.directory.appendingPathComponent("training/artifacts/model.py"),encoding:.utf8))
        }
    }
    func testColdReadRechecksActualPacketBytesAndRecoversOnlyRestoredEvidence() async throws {
        let root=try root();defer {try? FileManager.default.removeItem(at:root)}
        let saved=try await BagHistoryTestFixture.clone(objective:true,into:root)
        XCTAssertNotNil(try WorkflowHistoryReader.loadModelContent(saved.reference,root:root)?.bagPredictionPreview)
        let file=saved.directory.appendingPathComponent("training/artifacts/predictions.json")
        let original=try Data(contentsOf:file)
        try Data("{}".utf8).write(to:file)
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(saved.reference,root:root))
        guard case .unavailable=WorkflowHistoryReader.load(saved.reference,root:root) else {return XCTFail("Changed packet bytes must invalidate cold History")}
        // A corrupted physical clone must not poison the immutable genuine capture.
        let isolatedRoot=try self.root();defer {try? FileManager.default.removeItem(at:isolatedRoot)}
        let isolated=try await BagHistoryTestFixture.clone(objective:true,into:isolatedRoot)
        XCTAssertEqual(try Data(contentsOf:isolated.directory.appendingPathComponent("training/artifacts/predictions.json")),original)
        XCTAssertNotNil(try WorkflowHistoryReader.loadModelContent(isolated.reference,root:isolatedRoot)?.bagPredictionPreview)
        try original.write(to:file)
        XCTAssertNotNil(try WorkflowHistoryReader.loadModelContent(saved.reference,root:root)?.bagPredictionPreview)
        try FileManager.default.removeItem(at:saved.directory.appendingPathComponent("training/artifacts/activations.json"))
        XCTAssertThrowsError(try WorkflowHistoryReader.loadModelContent(saved.reference,root:root))
    }
}
