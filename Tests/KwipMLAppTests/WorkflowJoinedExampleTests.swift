import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowJoinedExampleTests:XCTestCase {
    private var repo:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    @MainActor func testJoinedExamplePersistsBothPrivateSourcesAndReopens()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(store.loadJoinedWorkflowExample(),store.statusMessage)
        let host=try XCTUnwrap(store.workflowSession),document=host.owner.document
        var paths:[WorkflowNodeAddress:String]=[:]
        for source in PreparedWorkflowExample.joined.sourceResources {
            let path=try XCTUnwrap(host.bindings.path(for:source.address));paths[source.address]=path
            XCTAssertTrue(path.hasPrefix(root.appendingPathComponent("example-sources").path+"/"))
            XCTAssertEqual(try String(contentsOfFile:path,encoding:.utf8).split(whereSeparator:\.isNewline).count,65)
            XCTAssertFalse(String(decoding:try WorkflowCodec.encode(document),as:UTF8.self).contains(path))
        }
        XCTAssertTrue(store.loadPreparedWorkflowExample())
        XCTAssertTrue(store.beginWorkflowProject(id:document.projectID),store.statusMessage)
        let restored=try XCTUnwrap(store.workflowSession)
        XCTAssertEqual(restored.owner.document,document)
        for (address,path) in paths {XCTAssertEqual(restored.bindings.path(for:address),path)}
    }
    @MainActor func testMissingSecondResourceDoesNotCreatePartialProjectOrBindings()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(store.loadPreparedWorkflowExample())
        let host=try XCTUnwrap(store.workflowSession)
        let before=try store.projectLibrary.listSources().map(\.id)
        let sourcesBefore=try FileManager.default.contentsOfDirectory(atPath:root.appendingPathComponent("example-sources").path)
        var requested:[String]=[]
        XCTAssertFalse(store.loadPreparedWorkflowExample(device:"cpu",kind:.joined,resourceURL:{ name in
            requested.append(name)
            return name == "join-signals" ? self.repo.appendingPathComponent("Sources/KwipMLApp/Resources/join-signals.csv") : nil
        }))
        XCTAssertEqual(requested,["join-signals","join-context"])
        XCTAssertTrue(store.workflowSession === host)
        XCTAssertFalse(host.isRetired)
        XCTAssertEqual(try store.projectLibrary.listSources().map(\.id),before)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.appendingPathComponent("example-sources").path),sourcesBefore)
    }
    @MainActor func testJoinedExampleActualTrainingAndVerifiedTwoSourceArchive()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual worker runtime")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(store)
        XCTAssertTrue(store.loadJoinedWorkflowExample(device:"cpu"),store.statusMessage)
        let host=try XCTUnwrap(store.workflowSession)
        let done=expectation(description:"Joined example trains")
        let observation=host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in done.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(host.run(),host.statusMessage)
        await fulfillment(of:[done],timeout:60)
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        let listing=try WorkflowHistoryReader.list(root:root,projectID:host.owner.projectID,limit:1000)
        let reference=try XCTUnwrap(listing.entries.first?.reference)
        guard case .verified(let archive)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Archive verification failed")}
        XCTAssertEqual(Set(archive.capture.sources.map(\.address)),[WorkflowExamples.sourceAddress,WorkflowExamples.joinContextAddress])
        XCTAssertEqual(archive.modelContent?.metrics.count,3)
        XCTAssertTrue(archive.modelContent?.code.contains("class Model") == true)
        let snapshot=try XCTUnwrap(host.snapshot(for:.init(domain:.data,localID:"join")))
        guard case .object(let table)=snapshot.ports["table"],case .object(let preview)=table["preview"] else {return XCTFail("Missing joined preview")}
        XCTAssertEqual(preview["rowCount"],.integer(64))
        XCTAssertEqual(preview["partition"],.string("unassigned"))
    }
}
