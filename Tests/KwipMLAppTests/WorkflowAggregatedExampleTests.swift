import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowAggregatedExampleTests:XCTestCase {
    private var repo:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    @MainActor func testBundledMeasurementsHaveIndependentSubjectsAndConsistentLabels()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(store.loadAggregatedWorkflowExample(),store.statusMessage)
        let host=try XCTUnwrap(store.workflowSession),document=host.owner.document
        let path=try XCTUnwrap(host.bindings.path(for:WorkflowExamples.sourceAddress))
        let lines=try String(contentsOfFile:path,encoding:.utf8).split(whereSeparator:\.isNewline)
        XCTAssertEqual(lines.count,193)
        XCTAssertEqual(lines.first,"subject_id,signal,context,label")
        let rows=lines.dropFirst().map {$0.split(separator:",").map(String.init)}
        let groups=Dictionary(grouping:rows,by:{$0[0]})
        XCTAssertEqual(groups.count,64)
        for values in groups.values {
            XCTAssertEqual(values.count,3)
            XCTAssertEqual(Set(values.map {$0[3]}).count,1,"This fixture guarantees consistent labels; min is not conflict validation")
        }
        XCTAssertFalse(String(decoding:try WorkflowCodec.encode(document),as:UTF8.self).contains(path))
        XCTAssertTrue(store.loadPreparedWorkflowExample())
        XCTAssertTrue(store.beginWorkflowProject(id:document.projectID),store.statusMessage)
        XCTAssertEqual(store.workflowSession?.bindings.path(for:WorkflowExamples.sourceAddress),path)
    }
    @MainActor func testActualAggregationTrainingAndArchivedCounts()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual worker")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(store)
        XCTAssertTrue(store.loadAggregatedWorkflowExample(device:"cpu"),store.statusMessage)
        let host=try XCTUnwrap(store.workflowSession)
        let done=expectation(description:"Repeated measurements aggregated and trained")
        let observation=host.$outcomes.filter{!$0.isEmpty}.prefix(1).sink{_ in done.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(host.run(),host.statusMessage)
        await fulfillment(of:[done],timeout:60)
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        let listing=try WorkflowHistoryReader.list(root:root,projectID:host.owner.projectID,limit:1000)
        let reference=try XCTUnwrap(listing.entries.first?.reference)
        guard case .verified(let archive)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Archive did not verify")}
        XCTAssertEqual(archive.modelContent?.metrics.count,3)
        XCTAssertEqual(archive.modelContent?.partitionEvidence?.trainingCount,32)
        XCTAssertEqual(archive.modelContent?.partitionEvidence?.validationCount,16)
        let snapshot=try XCTUnwrap(host.snapshot(for:.init(domain:.data,localID:"aggregate")))
        guard case .object(let table)=snapshot.ports["table"],case .object(let preview)=table["preview"] else {return XCTFail("Missing measured aggregation")}
        XCTAssertEqual(preview["rowCount"],.integer(64))
        XCTAssertEqual(preview["partition"],.string("unassigned"))
    }
}
