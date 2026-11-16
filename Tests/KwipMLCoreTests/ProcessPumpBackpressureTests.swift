import XCTest
@testable import KwipMLCore

@MainActor final class ProcessPumpBackpressureTests: XCTestCase {
    private let flood = "awk 'BEGIN { p=\"\"; for(j=0;j<2048;j++) p=p \"x\"; for(i=0;i<6000;i++) { printf \"%d:%s\\n\",i,p; printf \"%s\\n\",p > \"/dev/stderr\" } }'"
    private func markerAppears(_ url: URL, seconds: Double) -> Bool {
        // Deliberately occupy main without pumping its run loop. Pipe readers and
        // lifetime cancellation must remain independent of UI availability.
        let end = Date().addingTimeInterval(seconds)
        while Date() < end {
            if FileManager.default.fileExists(atPath: url.path) { return true }
            Thread.sleep(forTimeInterval: 0.01)
        }
        return FileManager.default.fileExists(atPath: url.path)
    }
    func testBusyMainBackpressuresDualPipeProducerThenDeliversExactFIFOAndTailBeforeExit() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let marker = directory.appendingPathComponent("producer-finished")
        let done = expectation(description: "bounded delivery finished")
        var next = 0
        var tail = false
        let script = flood + "; printf 'tail'; printf 'stderr-final' >&2; printf done > \"$1\""
        let process = try ProcessPump.launch(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", script, "kwip-pump", marker.path]) { line in
            XCTAssertTrue(Thread.isMainThread)
            if line == "tail" { tail = true; XCTAssertEqual(next, 6000) }
            else { XCTAssertEqual(Int(line.prefix { $0 != ":" }), next); next += 1 }
        } onExit: { code, stderr in
            XCTAssertEqual(code, 0, stderr)
            XCTAssertEqual(next, 6000)
            XCTAssertTrue(tail)
            XCTAssertTrue(stderr.hasSuffix("stderr-final"))
            XCTAssertLessThanOrEqual(stderr.utf8.count, 8000)
            done.fulfill()
        }
        defer { if process.isRunning { process.terminate() } }
        XCTAssertFalse(markerAppears(marker, seconds: 3), "A busy UI must apply bounded backpressure rather than queue the producer’s entire output on main.")
        XCTAssertEqual(next, 0, "Main was intentionally occupied; callbacks cannot already have run.")
        wait(for: [done], timeout: 20)
    }
    func testLifetimeCancellationReachesProducerWhileMainDeliveryIsSaturated() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let ready = directory.appendingPathComponent("ready")
        let ack = directory.appendingPathComponent("cancel-ack")
        let done = expectation(description: "cancelled delivery finished")
        var next = 0
        var tail = false
        // Simulate a supervisor watching its lifetime independently of an output-
        // blocked worker. Acknowledging EOF must not wait for main-queue delivery.
        let script = flood + " & child=$!; printf ready > \"$2\"; while IFS= read -r line; do :; done; printf ack > \"$1\"; wait \"$child\"; printf 'cancelled-tail'; exit 130"
        let handle = try ProcessPump.launchWithLifetime(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", script, "kwip-pump", ack.path, ready.path]) { line in
            if line == "cancelled-tail" { tail = true; XCTAssertEqual(next, 6000) }
            else { XCTAssertEqual(Int(line.prefix { $0 != ":" }), next); next += 1 }
        } onExit: { code, stderr in
            XCTAssertEqual(code, 130, stderr); XCTAssertEqual(next, 6000); XCTAssertTrue(tail)
            XCTAssertLessThanOrEqual(stderr.utf8.count, 8000); done.fulfill()
        }
        defer { handle.cancel(); if handle.process.isRunning { handle.process.terminate() } }
        XCTAssertTrue(markerAppears(ready, seconds: 3))
        Thread.sleep(forTimeInterval: 0.15)
        handle.cancel(); handle.cancel()
        XCTAssertTrue(markerAppears(ack, seconds: 3), "Cancellation must close lifetime even while main cannot accept queued lines.")
        XCTAssertEqual(next, 0)
        wait(for: [done], timeout: 20)
    }
}
