import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowHistoryStoreTests:XCTestCase {
    private func fixture()throws->(WorkspaceStore,WorkflowWorkspaceSession) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let document=WorkflowDocument(projectID:UUID().uuidString)
        _=try workspace.projectLibrary.saveSource(.workflow(document))
        workspace.activeProjectID=document.projectID
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let host=try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.workflowSession=host
        return(workspace,host)
    }
    private func reference(_ host:WorkflowWorkspaceSession)->WorkflowHistoryReference {.init(projectID:host.owner.projectID,directoryID:UUID())}
    private func awaitIdle(_ store:WorkflowHistoryStore)async {
        if !store.isLoading && !store.isRefreshing {return}
        let done=expectation(description:"History operation finishes")
        let observation=store.$isLoading.combineLatest(store.$isRefreshing).filter {!$0 && !$1}.prefix(1).sink {_ in done.fulfill()}
        await fulfillment(of:[done],timeout:3);observation.cancel()
    }
    func testSlowOldSelectionCannotOverwriteNewSelectionAndRefreshRetainsSnapshot()async throws {
        let(workspace,host)=try fixture(),loader=HistoryLoaderProbe()
        let store=WorkflowHistoryStore(workspace:workspace,host:host,loader:loader)
        let first=reference(host),second=reference(host)
        store.select(first);await loader.waitForLoads(1)
        store.select(second);await loader.waitForLoads(2)
        await loader.finishLoad(1,.incomplete("Second selected result"));await awaitIdle(store)
        XCTAssertEqual(store.selectedReference,second)
        guard case .incomplete("Second selected result")=store.selection else {return XCTFail("Expected selected snapshot")}
        await loader.finishLoad(0,.unavailable("Old late result"))
        await Task.yield()
        guard case .incomplete("Second selected result")=store.selection else {return XCTFail("Late first selection overwrote result")}
        store.refresh();await loader.waitForLists(1)
        XCTAssertTrue(store.isRefreshing)
        guard case .incomplete("Second selected result")=store.selection else {return XCTFail("Refresh cleared selected snapshot")}
        await loader.finishList(0,.init(entries:[],issues:["Bounded scan"],truncated:true));await awaitIdle(store)
        XCTAssertEqual(store.listing.issues,["Bounded scan"]);XCTAssertTrue(store.listing.truncated)
        guard case .incomplete("Second selected result")=store.selection else {return XCTFail("Refresh replaced immutable selection")}
    }
    func testCancelAndForeignProjectRejectLatePublicationAndClearLoading()async throws {
        let(workspace,host)=try fixture(),loader=HistoryLoaderProbe()
        let store=WorkflowHistoryStore(workspace:workspace,host:host,loader:loader),ref=reference(host)
        store.select(ref);await loader.waitForLoads(1)
        XCTAssertTrue(store.isLoading)
        store.cancelLoading();XCTAssertFalse(store.isLoading);XCTAssertFalse(store.isRefreshing)
        await loader.finishLoad(0,.incomplete("Cancelled result"));await Task.yield()
        XCTAssertNil(store.selection)
        store.select(ref);await loader.waitForLoads(2)
        workspace.workflowSession=nil
        await loader.finishLoad(1,.incomplete("Former project"));await awaitIdle(store)
        XCTAssertNil(store.selection);XCTAssertFalse(store.isLoading)
        store.invalidate();XCTAssertFalse(store.isRefreshing)
        store.select(.init(projectID:UUID().uuidString,directoryID:UUID()))
        let count=await loader.loadCount();XCTAssertEqual(count,2)
    }
    func testActualReaderListsLegacyInterruptedAndCorruptRecordsWithoutPromotingThem()async throws {
        let(workspace,host)=try fixture(),project=host.owner.projectID
        let legacy=reference(host),interrupted=reference(host),corrupt=reference(host)
        func folder(_ ref:WorkflowHistoryReference)throws->URL {
            let value=workspace.workspaceDirectory.appendingPathComponent("workflow-runs/\(project)/\(ref.directoryID.uuidString)")
            try FileManager.default.createDirectory(at:value,withIntermediateDirectories:true);return value
        }
        let receipt=WorkflowLegacyOutcome(schemaVersion:1,runID:UUID().uuidString,projectID:project,requestIdentity:String(repeating:"a",count:64),status:.failed,failedPhase:.preparing,preparationManifestSHA256:nil,modelManifestSHA256:nil,artifactHashes:[:],sourceHashes:[:],error:"Recorded failure")
        try JSONEncoder().encode(receipt).write(to:folder(legacy).appendingPathComponent("outcome.json"))
        _=try folder(interrupted)
        try Data("{bad".utf8).write(to:folder(corrupt).appendingPathComponent("outcome.json"))
        let store=WorkflowHistoryStore(workspace:workspace,host:host)
        XCTAssertTrue(store.listing.entries.isEmpty,"Construction performs no disk read.")
        store.refresh();await awaitIdle(store)
        XCTAssertEqual(store.listing.entries.count,3)
        XCTAssertEqual(store.listing.entries.first {$0.reference == legacy}?.verification,.legacyV1)
        XCTAssertEqual(store.listing.entries.first {$0.reference == interrupted}?.verification,.incomplete)
        XCTAssertEqual(store.listing.entries.first {$0.reference == corrupt}?.verification,.unavailable)
        store.select(legacy);await awaitIdle(store)
        guard case .legacyUnverifiable=store.selection else {return XCTFail("Legacy cannot claim verified v2 evidence")}
        XCTAssertNil(store.modelContent)
        XCTAssertFalse(store.pinComparisonBaseline())
        XCTAssertNil(store.comparisonBaseline)
    }
}

private actor HistoryLoaderProbe:WorkflowHistoryLoading {
    private var loads:[CheckedContinuation<WorkflowHistoryLoad,Error>?]=[]
    private var lists:[CheckedContinuation<WorkflowHistoryListing,Error>?]=[]
    private var loadWaiters:[(Int,CheckedContinuation<Void,Never>)]=[]
    private var listWaiters:[(Int,CheckedContinuation<Void,Never>)]=[]
    func load(_ reference:WorkflowHistoryReference,root:URL)async throws->WorkflowHistoryLoad {
        try await withCheckedThrowingContinuation {continuation in
            loads.append(continuation)
            let ready=loadWaiters.filter {$0.0<=loads.count};loadWaiters.removeAll {$0.0<=loads.count}
            ready.forEach {$0.1.resume()}
        }
    }
    func list(root:URL,projectID:String)async throws->WorkflowHistoryListing {
        try await withCheckedThrowingContinuation {continuation in
            lists.append(continuation)
            let ready=listWaiters.filter {$0.0<=lists.count};listWaiters.removeAll {$0.0<=lists.count}
            ready.forEach {$0.1.resume()}
        }
    }
    func waitForLoads(_ count:Int)async {if loads.count<count {await withCheckedContinuation {loadWaiters.append((count,$0))}}}
    func waitForLists(_ count:Int)async {if lists.count<count {await withCheckedContinuation {listWaiters.append((count,$0))}}}
    func finishLoad(_ index:Int,_ value:WorkflowHistoryLoad) {loads[index]?.resume(returning:value);loads[index]=nil}
    func finishList(_ index:Int,_ value:WorkflowHistoryListing) {lists[index]?.resume(returning:value);lists[index]=nil}
    func loadCount()->Int {loads.count}
}
