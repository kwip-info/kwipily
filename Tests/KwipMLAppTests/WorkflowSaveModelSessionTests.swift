import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

private actor SaveModelStageGate:WorkflowSaveModelIO {
    let reached:()->Void
    private var release:CheckedContinuation<Void,Never>?
    init(reached:@escaping()->Void) {self.reached=reached}
    func proceed() {release?.resume();release=nil}
    func stage(reference:WorkflowFrozenModelReference,format:WorkflowSaveModelConfiguration.Format,root:URL,
               destination:URL,accountingRoot:URL?,storageLimitBytes:Int64)async throws->WorkflowSaveModelPrepared {
        let bytes=Data("captured checkpoint".utf8),hash=WorkflowArchiveCodec.hash(bytes)
        let publication=try WorkflowExportPublication.stageExclusiveCheckpoint(bytes,sha256:hash,to:destination)
        let prepared=WorkflowSaveModelPrepared(receipt:.init(reference:reference,format:format,destination:publication.destination,
            files:[.init(path:destination.lastPathComponent,bytes:bytes.count,sha256:hash)]),publication:publication)
        await withCheckedContinuation {release=$0;reached()}
        // Return even when cancelled: the MainActor must reject this stale result itself.
        return prepared
    }
}

final class WorkflowSaveModelSessionTests:XCTestCase {
    private func folder()throws->URL {
        let url=try canonicalBackendTemporaryDirectory()
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:false,attributes:[.posixPermissions:0o700])
        addTeardownBlock {try? FileManager.default.removeItem(at:url)};return url
    }
    private var reference:WorkflowFrozenModelReference {
        let hash=String(repeating:"a",count:64)
        return .init(projectID:UUID().uuidString,directoryID:UUID().uuidString,runID:UUID().uuidString,
            requestIdentity:hash,outcomeSHA256:hash,modelManifestSHA256:hash,preparationManifestSHA256:hash,checkpointSHA256:hash)
    }
    @MainActor func testSuccessfulSaveRetainsReservationAndGrantThroughPublication()async throws {
        let root=try folder(),destination=root.appendingPathComponent("model.pt"),reference=reference
        let reached=expectation(description:"Stage awaits final authority"),completed=expectation(description:"Saved")
        let io=SaveModelStageGate {reached.fulfill()}
        var held:UUID?,releases=0
        let access=try WorkflowModelDestinationAccess(destination:destination,accessURL:root,revision:3)
        let session=WorkflowSaveModelSession(reference:reference,format:.checkpoint,root:root,storageLimitBytes:Int64.max,
            reservation:.init(acquire:{let token=UUID();held=token;return token},release:{token in XCTAssertEqual(held,token);held=nil;releases+=1}),
            io:io,isCurrent:{true},onSaved:{receipt in
                XCTAssertEqual(receipt.reference,reference);XCTAssertEqual(receipt.destination,destination)
                XCTAssertEqual(receipt.format,.checkpoint);XCTAssertNotNil(receipt.savedAt)
                XCTAssertEqual(receipt.files.first?.sha256,WorkflowArchiveCodec.hash(Data("captured checkpoint".utf8)))
                completed.fulfill()
            })
        XCTAssertTrue(session.save(using:access));await fulfillment(of:[reached],timeout:5)
        XCTAssertTrue(session.isBusy);XCTAssertNotNil(held);try access.check()
        XCTAssertFalse(FileManager.default.fileExists(atPath:destination.path))
        XCTAssertFalse(session.save(to:root.appendingPathComponent("duplicate.pt"),access:root))
        await io.proceed();await fulfillment(of:[completed],timeout:5)
        XCTAssertFalse(session.isBusy);XCTAssertNil(held);XCTAssertEqual(releases,1)
        XCTAssertEqual(session.savedURL,destination);XCTAssertNotNil(session.receipt)
        XCTAssertThrowsError(try access.check())
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["model.pt"])
    }
    @MainActor func testCancelledOrStaleStageNeverPublishesAndAlwaysReleases()async throws {
        for change in ["cancel","invalidate","current","grant"] {
            let root=try folder(),destination=root.appendingPathComponent("model.pt")
            let reached=expectation(description:"Stage held"),finished=expectation(description:"Stage joined")
            let io=SaveModelStageGate {reached.fulfill()}
            var current=true,held=false,releases=0,failures=0
            let access=try WorkflowModelDestinationAccess(destination:destination,accessURL:root,revision:0)
            let session=WorkflowSaveModelSession(reference:reference,format:.checkpoint,root:root,storageLimitBytes:Int64.max,
                reservation:.init(acquire:{held=true;return UUID()},release:{_ in held=false;releases+=1;finished.fulfill()}),
                io:io,isCurrent:{current},onSaved:{_ in XCTFail("Stale or cancelled result was published")},onFailed:{failures+=1})
            XCTAssertTrue(session.save(using:access));await fulfillment(of:[reached],timeout:5)
            switch change {case "cancel":session.cancel();case "invalidate":session.invalidate();case "current":current=false;default:access.close()}
            XCTAssertTrue(held,"Reservation must survive until the asynchronous stage joins")
            await io.proceed();await fulfillment(of:[finished],timeout:5)
            XCTAssertFalse(session.isBusy);XCTAssertFalse(held);XCTAssertEqual(releases,1);XCTAssertEqual(failures,1)
            XCTAssertNil(session.savedURL);XCTAssertNil(session.receipt);XCTAssertThrowsError(try access.check())
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),[])
        }
    }
    @MainActor func testDeniedReservationAndMissingGrantNeverInvokeStage()throws {
        let root=try folder(),access=try WorkflowModelDestinationAccess(destination:root.appendingPathComponent("model.pt"),accessURL:root,revision:0)
        let io=SaveModelStageGate {XCTFail("Denied action must not stage")}
        let session=WorkflowSaveModelSession(reference:reference,format:.checkpoint,root:root,storageLimitBytes:Int64.max,
            reservation:.init(acquire:{nil},release:{_ in XCTFail("No reservation was acquired")}),io:io,isCurrent:{true})
        XCTAssertFalse(session.save(to:root.appendingPathComponent("model.pt")))
        XCTAssertFalse(session.save(using:access));XCTAssertThrowsError(try access.check())
        XCTAssertFalse(session.isBusy);XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),[])
    }
}
