import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WebhookLedgerTests:XCTestCase {
    private func endpoint()->WebhookEndpoint {
        let id=UUID().uuidString
        return .init(id:id,rule:ScheduleRule(id:id,projectID:UUID().uuidString,projectName:"Events",versionID:UUID().uuidString,versionNumber:1,versionHash:String(repeating:"a",count:64),name:"Events",timing:.init()),enabled:true)
    }
    func testDurableDedupConflictQueueLimitAndRestartDoNotReplayClaim()async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        var ledger=try WebhookLedger(root:root);let e=endpoint();try await ledger.save(e)
        try await ledger.configure(enabled:true,port:8788,body:1024,queued:1,rate:60,days:30,maximum:10)
        let liveState=await ledger.snapshot();let live=try XCTUnwrap(liveState.endpoints.first)
        let accepted=try await ledger.accept(endpoint:live,deliveryID:"event-1",digest:String(repeating:"b",count:64),payload:Data("[]".utf8),preview:nil,rows:0,busy:false)
        XCTAssertFalse(accepted.1)
        let duplicate=try await ledger.accept(endpoint:live,deliveryID:"event-1",digest:String(repeating:"b",count:64),payload:Data("[]".utf8),preview:nil,rows:0,busy:false)
        XCTAssertTrue(duplicate.1);XCTAssertEqual(duplicate.0.id,accepted.0.id)
        do{_ = try await ledger.accept(endpoint:live,deliveryID:"event-1",digest:String(repeating:"c",count:64),payload:nil,preview:nil,rows:nil,busy:false);XCTFail("Conflicting body accepted")}catch{XCTAssertEqual((error as? WebhookRefusal)?.status,409)}
        do{_ = try await ledger.accept(endpoint:live,deliveryID:"event-2",digest:String(repeating:"b",count:64),payload:nil,preview:nil,rows:nil,busy:false);XCTFail("Queue overflow accepted")}catch{XCTAssertEqual((error as? WebhookRefusal)?.status,429)}
        _ = try await ledger.claim(accepted.0.id);await ledger.close()
        ledger=try WebhookLedger(root:root);let restored=await ledger.snapshot()
        XCTAssertEqual(restored.deliveries.count,1);XCTAssertEqual(restored.deliveries[0].state,.needsReview);XCTAssertNil(restored.deliveries[0].payload);XCTAssertFalse(restored.endpoints[0].enabled)
        await ledger.close()
    }
    func testStaleDraftPendingRemovalAndCaptureCleanup()async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let ledger=try WebhookLedger(root:root);var e=endpoint();try await ledger.save(e)
        let savedState=await ledger.snapshot();e=try XCTUnwrap(savedState.endpoints.first);var stale=e;e.name="New name";try await ledger.save(e)
        stale.name="Lost update";do{try await ledger.save(stale);XCTFail("Lost update")}catch{}
        try await ledger.configure(enabled:true,port:8788,body:1024,queued:32,rate:60,days:30,maximum:10)
        let liveState=await ledger.snapshot();let live=try XCTUnwrap(liveState.endpoints.first)
        let d=try await ledger.accept(endpoint:live,deliveryID:"one",digest:String(repeating:"a",count:64),payload:Data("[]".utf8),preview:nil,rows:0,busy:true).0
        do{try await ledger.remove(live.id);XCTFail("Removed accepted queued work")}catch{}
        try await ledger.cancelQueued(d.id)
        let cancelledState=await ledger.snapshot();let cancelled=try XCTUnwrap(cancelledState.deliveries.first)
        XCTAssertEqual(cancelled.state,.cancelled);XCTAssertNil(cancelled.payload)
        try await ledger.remove(live.id)
        let end=await ledger.snapshot();XCTAssertEqual(end.deliveries.count,1)
        await ledger.close()
    }
}

extension WebhookLedgerTests {
    func testManagedStorageAdmissionAndPayloadDigestRefuseUnsafeRetention()async throws {
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:root)}
        let ledger=try WebhookLedger(root:root);try await ledger.save(endpoint());try await ledger.configure(enabled:true,port:8788,body:1024,queued:32,rate:60,days:30,maximum:10)
        let s=await ledger.snapshot(),e=try XCTUnwrap(s.endpoints.first)
        await ledger.setStorageLimit(0)
        do{_ = try await ledger.accept(endpoint:e,deliveryID:"blocked",digest:String(repeating:"a",count:64),payload:Data("[]".utf8),preview:nil,rows:0,busy:false);XCTFail("Storage quota ignored")}catch{}
        let unchanged=await ledger.snapshot();XCTAssertTrue(unchanged.deliveries.isEmpty)
        await ledger.setStorageLimit(nil)
        _ = try await ledger.accept(endpoint:e,deliveryID:"valid",digest:String(repeating:"a",count:64),payload:Data("[]".utf8),preview:nil,rows:0,busy:false)
        await ledger.close()
        let file=root.appendingPathComponent("webhooks/state.json")
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any]);var rows=try XCTUnwrap(object["deliveries"] as? [[String:Any]])
        rows[0]["payload"]=Data("[1]".utf8).base64EncodedString();object["deliveries"]=rows;try JSONSerialization.data(withJSONObject:object).write(to:file)
        XCTAssertThrowsError(try WebhookLedger(root:root),"Corrupt queued input must never execute")
    }
}
