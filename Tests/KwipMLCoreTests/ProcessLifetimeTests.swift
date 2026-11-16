import XCTest
import Foundation
import Darwin
@testable import KwipMLCore

final class ProcessLifetimeTests:XCTestCase {
    func testWriterKeepsChildAliveAndCancellationDeliversEOFBeforeExit()throws {
        let ready=expectation(description:"ready"),exit=expectation(description:"exit after EOF")
        var events:[String]=[]
        let handle=try ProcessPump.launchWithLifetime(executable:URL(fileURLWithPath:"/bin/sh"),arguments:["-c","printf 'READY\\n'; while IFS= read -r line; do :; done; printf 'EOF_ACK\\nfinal'"]) {line in
            events.append(line);if line == "READY" {ready.fulfill()}
        } onExit:{code,stderr in
            XCTAssertEqual(code,0,stderr);XCTAssertEqual(events,["READY","EOF_ACK","final"]);exit.fulfill()
        }
        defer {handle.cancel();if handle.process.isRunning {handle.process.terminate()}}
        wait(for:[ready],timeout:5)
        XCTAssertTrue(handle.process.isRunning,"Parent-only writer holds lifetime open.")
        XCTAssertEqual(events,["READY"])
        handle.cancel();handle.cancel()
        wait(for:[exit],timeout:5)
        XCTAssertFalse(handle.process.isRunning)
    }
    func testDroppingHandleClosesLifetimeWithoutRetainingItThroughDrain()throws {
        let ready=expectation(description:"ready"),exit=expectation(description:"EOF from deinit")
        var handle:ProcessLifetimeHandle?
        handle=try ProcessPump.launchWithLifetime(executable:URL(fileURLWithPath:"/bin/sh"),arguments:["-c","printf 'READY\\n'; while IFS= read -r line; do :; done; printf 'EOF_ACK\\n'"]) {line in
            if line == "READY" {ready.fulfill()}
        } onExit:{code,stderr in XCTAssertEqual(code,0,stderr);exit.fulfill()}
        let process=try XCTUnwrap(handle).process
        defer {if process.isRunning {process.terminate()}}
        wait(for:[ready],timeout:5)
        handle=nil
        wait(for:[exit],timeout:5)
        XCTAssertFalse(process.isRunning)
    }
    func testNaturalExitPreservesBoundedDiagnosticsAndAllowsLaterCancel()throws {
        let done=expectation(description:"natural exit")
        var events:[String]=[]
        let handle=try ProcessPump.launchWithLifetime(executable:URL(fileURLWithPath:"/bin/sh"),arguments:["-c","printf 'first\\nlast'; printf 'diagnostic' >&2; exit 7"]) {events.append($0)} onExit:{code,stderr in
            XCTAssertEqual(code,7);XCTAssertEqual(events,["first","last"]);XCTAssertEqual(stderr,"diagnostic");done.fulfill()
        }
        wait(for:[done],timeout:5)
        XCTAssertFalse(handle.process.isRunning)
        handle.cancel();handle.cancel()
    }
    func testSpawnFailureClosesDescriptorsAndDoesNotInvokeCallbacks()throws {
        func openDescriptors()->Set<Int32> {Set((0..<1024).map(Int32.init).filter {fcntl($0,F_GETFD) != -1})}
        // Warm Foundation launch failure before measuring its process-global initialization.
        XCTAssertThrowsError(try ProcessPump.launchWithLifetime(executable:URL(fileURLWithPath:"/does-not-exist/kwipml-worker"),arguments:[],onLine:{_ in XCTFail("Unexpected line")},onExit:{_,_ in XCTFail("Unexpected exit")}))
        let before=openDescriptors()
        for _ in 0..<8 {
            XCTAssertThrowsError(try ProcessPump.launchWithLifetime(executable:URL(fileURLWithPath:"/does-not-exist/kwipml-worker"),arguments:[],onLine:{_ in XCTFail("Unexpected line")},onExit:{_,_ in XCTFail("Unexpected exit")}))
        }
        XCTAssertEqual(openDescriptors(),before,"Failed launches must close both lifetime ends and output pipes.")
    }
}
