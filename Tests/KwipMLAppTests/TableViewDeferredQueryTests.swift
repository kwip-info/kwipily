import XCTest
@testable import KwipMLCore
@testable import KwipMLApp

@MainActor final class TableViewDeferredQueryTests:XCTestCase {
    func testRejectedReservationDefersAndCancelOrStaleSelectionNeverStartsWork()throws {
        // Paths deliberately do not exist: a blocked read must not touch source
        // files, launch a worker or acquire a second execution reservation.
        let root=URL(fileURLWithPath:"/unused/owned"),pin=String(repeating:"a",count:64)
        let runtime=RuntimeLocation(kind:.development,packageVersions:[:],executableURL:root,entryURL:root,registryURL:root,workingDirectoryURL:root)
        var attempts=0,current=true
        let session=TableViewQuerySession(artifact:.init(indexURL:root.appendingPathComponent("captured.table.json"),indexSHA256:pin,sourceSHA256:pin,inputKind:"table"),workspace:root,projectID:UUID().uuidString,runtime:runtime,reservation:.init(acquire:{attempts+=1;return nil},release:{_ in XCTFail("No reservation was acquired")}),accepts:{current})
        XCTAssertFalse(session.load(offset:100,search:"hello",sort:.init()))
        XCTAssertEqual(attempts,1);XCTAssertTrue(session.waitingForExecution);XCTAssertFalse(session.isBusy)
        session.cancel()
        XCTAssertFalse(session.waitingForExecution)
        XCTAssertFalse(session.resumeAfterExecution());XCTAssertEqual(attempts,1)
        XCTAssertFalse(session.load(offset:200,search:"world",sort:.init()))
        current=false
        XCTAssertFalse(session.resumeAfterExecution());XCTAssertEqual(attempts,2)
        XCTAssertFalse(session.waitingForExecution)
        XCTAssertTrue(session.statusMessage.contains("no longer current"))
    }
}
