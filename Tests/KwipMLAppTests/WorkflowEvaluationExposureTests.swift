import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationExposureTests:XCTestCase {
    typealias Store = WorkflowEvaluationExposureStore
    private let source=String(repeating:"a",count:64)
    private let checkpoint=String(repeating:"b",count:64)
    private func root()throws->URL {
        guard let path=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw DataFlowContractError("Temporary directory is unavailable")}
        defer {free(path)}
        let value=URL(fileURLWithPath:String(cString:path)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:value,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:value)}
        return value
    }
    private func population(_ ids:[String]=["a","b","c"],source:String?=nil,checkpoint:String?=nil)throws->Store.Population {
        try .init(checkpointSHA256:checkpoint ?? self.checkpoint,sourceSHA256:source ?? self.source,observationIDs:ids)
    }
    private func event(_ population:Store.Population,kind:Store.Event.Kind = .inspectionRecorded,
                       evaluationID:String=UUID().uuidString,projectID:String=UUID().uuidString,
                       manifest:String=String(repeating:"c",count:64))->Store.Event {
        .init(kind:kind,evaluationID:evaluationID,projectID:projectID,manifestSHA256:manifest,
              snapshotReceiptSHA256:String(repeating:"d",count:64),population:population)
    }
    private func report(_ store:Store,_ population:Store.Population,file:StaticString=#filePath,line:UInt=#line)throws->Store.Report {
        if case .tracked(let report)=store.history(for:population) {return report}
        XCTFail("Expected readable tracking history",file:file,line:line)
        throw DataFlowContractError("Exposure history is unavailable")
    }
    func testMissingStoreIsUnknownAndExplicitStartDoesNotInventPriorFreshness()throws {
        let root=try root(),store=Store(root:root),p=try population()
        if case .unknown=store.history(for:p) {} else {XCTFail("Missing history must not mean fresh")}
        XCTAssertThrowsError(try store.record(event(p)))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("workflow-evaluation-exposure").path))
        try store.startTracking()
        let state=try report(store,p)
        XCTAssertTrue(state.priorHistoryUnknown)
        XCTAssertEqual(state.evaluatedCount,0);XCTAssertEqual(state.inspectionCount,0)
        try store.startTracking()
        XCTAssertTrue(try report(Store(root:root),p).priorHistoryUnknown)
    }
    func testEvaluatedAndDisclosedAreDistinctDurableAndSurviveResultDeletion()throws {
        let root=try root(),store=Store(root:root),p=try population(),id=UUID().uuidString,project=UUID().uuidString
        try store.startTracking()
        try store.record(event(p,kind:.evaluated,evaluationID:id,projectID:project))
        var state=try report(store,p)
        XCTAssertEqual(state.evaluatedCount,1);XCTAssertEqual(state.inspectionCount,0)
        try store.record(event(p,evaluationID:id,projectID:project))
        let disposable=root.appendingPathComponent("workflow-evaluations/\(project)/\(id)")
        try FileManager.default.createDirectory(at:disposable,withIntermediateDirectories:true)
        try Data("saved results".utf8).write(to:disposable.appendingPathComponent("outcome.json"))
        try FileManager.default.removeItem(at:root.appendingPathComponent("workflow-evaluations"))
        state=try report(Store(root:root),p)
        XCTAssertEqual(state.evaluatedCount,1);XCTAssertEqual(state.inspectionCount,1)
        XCTAssertEqual(state.checkpointInspectionCount,1);XCTAssertEqual(state.sameSourceInspectedOverlap,3)
    }
    func testExactHoldoutCrossesProjectAndCheckpointAndOverlapCrossesSplits()throws {
        let root=try root(),store=Store(root:root),first=try population()
        try store.startTracking();try store.record(event(first))
        let newCheckpoint=try population(["c","a","b"],checkpoint:String(repeating:"e",count:64))
        var state=try report(store,newCheckpoint)
        XCTAssertEqual(state.inspectionCount,1);XCTAssertEqual(state.checkpointInspectionCount,0)
        XCTAssertEqual(state.sameSourceInspectedOverlap,3);XCTAssertTrue(state.overlapIsComplete)
        try store.record(event(newCheckpoint))
        let overlapping=try population(["b","c","z"])
        state=try report(Store(root:root),overlapping)
        XCTAssertEqual(state.inspectionCount,0);XCTAssertEqual(state.sameSourceInspectedOverlap,2)
        let changedSource=try population(["a","b","c"],source:String(repeating:"f",count:64))
        state=try report(store,changedSource)
        XCTAssertEqual(state.sameSourceInspectedOverlap,0)
        XCTAssertFalse(state.overlapIsComplete,"Changed source bytes prevent proving row equivalence")
    }
    func testIdempotentDisclosureRejectsConflictingReplayAndPreservesBytes()throws {
        let root=try root(),store=Store(root:root),p=try population(),id=UUID().uuidString,project=UUID().uuidString
        try store.startTracking()
        let original=event(p,evaluationID:id,projectID:project)
        try store.record(original)
        let before=try files(root)
        try store.record(original)
        XCTAssertEqual(try files(root),before)
        XCTAssertThrowsError(try store.record(event(p,evaluationID:id,projectID:project,manifest:String(repeating:"e",count:64))))
        XCTAssertThrowsError(try store.record(event(try population(["different"]),evaluationID:id,projectID:project)))
        XCTAssertEqual(try files(root),before)
        XCTAssertEqual(try report(store,p).inspectionCount,1)
    }
    func testCorruptTruncatedAndRemovedEvidenceAreUnknownAndCannotBeOverwritten()throws {
        let root=try root(),store=Store(root:root),p=try population()
        try store.startTracking();try store.record(event(p))
        let originals=try files(root)
        XCTAssertFalse(originals.isEmpty)
        for (relative,original) in originals where !relative.hasSuffix(".lock") {
            let path=root.appendingPathComponent(relative)
            for bytes in [Data("{".utf8),Data(original.prefix(max(0,original.count/2)))] {
                try bytes.write(to:path)
                let damaged=Store(root:root)
                if case .unknown=damaged.history(for:p) {} else {XCTFail("Damaged \(relative) must be unknown")}
                XCTAssertThrowsError(try damaged.record(event(p)))
                XCTAssertThrowsError(try damaged.startTracking())
                XCTAssertEqual(try Data(contentsOf:path),bytes)
                try original.write(to:path)
            }
            try FileManager.default.removeItem(at:path)
            if case .unknown=Store(root:root).history(for:p) {} else {XCTFail("Missing \(relative) must be unknown")}
            try original.write(to:path)
        }
    }
    func testStoredPopulationUsesHashedIdentitiesWithoutRawLabels()throws {
        let root=try root(),store=Store(root:root)
        let raw=["private-observation-customer-8193","private-observation-customer-4721"]
        try store.startTracking();try store.record(event(try population(raw)))
        for bytes in try files(root).values {
            let text=String(decoding:bytes,as:UTF8.self)
            XCTAssertFalse(raw.contains(where:text.contains))
        }
    }
    func testCrossKindAndNoncanonicalIdentityConflictsCannotBeRecorded()throws {
        let root=try root(),store=Store(root:root),p=try population(),id=UUID().uuidString,project=UUID().uuidString
        try store.startTracking();try store.record(event(p,kind:.evaluated,evaluationID:id,projectID:project))
        let before=try files(root)
        XCTAssertThrowsError(try store.record(event(p,evaluationID:id,projectID:project,manifest:String(repeating:"e",count:64))))
        XCTAssertThrowsError(try store.record(event(p,evaluationID:id.lowercased(),projectID:project)))
        XCTAssertEqual(try files(root),before)
        try store.record(event(p,evaluationID:id,projectID:project))
        XCTAssertEqual(try report(store,p).inspectionCount,1)
    }
    func testRepeatedPopulationCountsEventsButDoesNotMultiplyOverlap()throws {
        let store=Store(root:try root()),p=try population()
        try store.startTracking()
        for _ in 0..<20 {try store.record(event(p))}
        let result=try report(store,p)
        XCTAssertEqual(result.inspectionCount,20);XCTAssertEqual(result.sameSourceInspectedOverlap,3)
    }
    func testOversizedJournalIndexIsUnknownAndWriteBlocked()throws {
        let root=try root(),store=Store(root:root),p=try population()
        try store.startTracking();try store.record(event(p))
        let path=root.appendingPathComponent("workflow-evaluation-exposure/index.json")
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:path)) as? [String:Any])
        let entries=try XCTUnwrap(object["events"] as? [[String:Any]])
        object["events"]=Array(repeating:try XCTUnwrap(entries.first),count:10001)
        let bytes=try JSONSerialization.data(withJSONObject:object);try bytes.write(to:path)
        if case .unknown=store.history(for:p) {} else {XCTFail("Oversized index must not disclose counts")}
        XCTAssertThrowsError(try store.record(event(p)))
        XCTAssertEqual(try Data(contentsOf:path),bytes)
    }
    func testPopulationRejectsInvalidIdentitiesAndLimitsBeforeStoreWrites()throws {
        XCTAssertThrowsError(try population([]))
        XCTAssertThrowsError(try population(["a"+String(repeating:"\u{0301}",count:201)]))
        XCTAssertThrowsError(try population(["same","same"]))
        XCTAssertThrowsError(try population([String(repeating:"x",count:201)]))
        XCTAssertThrowsError(try population(["a"],source:"bad"))
        XCTAssertThrowsError(try population(["a"],checkpoint:String(repeating:"B",count:64)))
        XCTAssertThrowsError(try population((0...100000).map(String.init)))
    }
    func testAtomicInspectionReturnsPriorAndReplayDoesNotAppend()throws {
        let root=try root(),store=Store(root:root),p=try population(),first=event(p),second=event(p)
        try store.startTracking()
        XCTAssertEqual(try store.recordInspectionReturningPrior(first).inspectionCount,0)
        XCTAssertEqual(try store.recordInspectionReturningPrior(second).inspectionCount,1)
        XCTAssertEqual(try store.recordInspectionReturningPrior(first).inspectionCount,2)
        XCTAssertEqual(try report(store,p).inspectionCount,2)
        XCTAssertThrowsError(try store.recordInspectionReturningPrior(event(p,kind:.evaluated)))
        XCTAssertEqual(try report(store,p).evaluatedCount,0)
    }
    func testConcurrentInspectionReportsAreSerializedOrExplicitlyBusy()throws {
        final class Results:@unchecked Sendable {
            let lock=NSLock()
            var values:[Int:Int]=[:]
            var failures:[Int:Error]=[:]
            func save(_ index:Int,_ result:Result<Int,Error>) {
                lock.lock();defer {lock.unlock()}
                switch result {case .success(let count):values[index]=count;case .failure(let error):failures[index]=error}
            }
        }
        for _ in 0..<8 {
            let root=try root(),store=Store(root:root),p=try population(),events=[event(p),event(p)]
            try store.startTracking()
            let ready=DispatchSemaphore(value:0),start=DispatchSemaphore(value:0),done=DispatchGroup(),results=Results()
            for index in events.indices {
                done.enter()
                DispatchQueue.global().async {
                    ready.signal();start.wait()
                    results.save(index,Result {try Store(root:root).recordInspectionReturningPrior(events[index]).inspectionCount})
                    done.leave()
                }
            }
            XCTAssertEqual(ready.wait(timeout:.now()+5),.success)
            XCTAssertEqual(ready.wait(timeout:.now()+5),.success)
            start.signal();start.signal()
            guard done.wait(timeout:.now()+10) == .success else {XCTFail("Bounded concurrent inspection did not finish");return}
            XCTAssertFalse(results.values.isEmpty)
            XCTAssertEqual(results.values.values.sorted(),Array(0..<results.values.count),"Successful disclosures cannot both report zero prior inspections")
            for (index,error) in results.failures {
                XCTAssertEqual(error as? WorkflowFileLeaseError,.busy)
                XCTAssertEqual(try store.recordInspectionReturningPrior(events[index]).inspectionCount,1)
            }
            XCTAssertEqual(try report(store,p).inspectionCount,2)
            for event in events {XCTAssertEqual(try store.recordInspectionReturningPrior(event).inspectionCount,2)}
        }
    }
    private func files(_ root:URL)throws->[String:Data] {
        let paths=try XCTUnwrap(FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey]))
        var result:[String:Data]=[:]
        for case let url as URL in paths where try url.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile == true {
            result[String(url.path.dropFirst(root.path.count+1))]=try Data(contentsOf:url)
        }
        return result
    }
}
