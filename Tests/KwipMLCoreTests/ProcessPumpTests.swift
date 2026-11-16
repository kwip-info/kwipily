import XCTest
@testable import KwipMLCore
final class ProcessPumpTests: XCTestCase {
    func testOversizedTerminatedEventIsRejectedBeforeDelivery() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        var bytes = Data(repeating: 120, count: 1_000_001)
        bytes.append(contentsOf: "\ncompletion-must-not-be-delivered\n".utf8)
        try bytes.write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let done = expectation(description: "oversized event rejected")
        var events: [String] = []
        _ = try ProcessPump.launch(executable: URL(fileURLWithPath: "/bin/cat"), arguments: [file.path]) { events.append($0) } onExit: { code, stderr in
            XCTAssertTrue(events.isEmpty, "Oversized framed data must not reach model decoders")
            XCTAssertNotEqual(code, 0)
            XCTAssertTrue(stderr.contains("Worker event exceeded 1 MB"))
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
    }
    func testEventAtByteLimitRemainsValid() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        // The bound applies to UTF-8 bytes, not Swift character count.
        let content = String(repeating: "é", count: 500_000)
        try Data(content.utf8).write(to: file)
        defer { try? FileManager.default.removeItem(at: file) }
        let done = expectation(description: "bounded final event")
        var events: [String] = []
        _ = try ProcessPump.launch(executable: URL(fileURLWithPath: "/bin/cat"), arguments: [file.path]) { events.append($0) } onExit: { code, stderr in
            XCTAssertEqual(events, [content]); XCTAssertEqual(code, 0); XCTAssertEqual(stderr, "")
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
    }
    func testLinesAreDeliveredBeforeExitIncludingFinalUnterminatedLine() throws {
        let done = expectation(description: "exit")
        var events: [String] = []
        _ = try ProcessPump.launch(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", "printf 'first\\nsecond'"]) { line in
            XCTAssertTrue(Thread.isMainThread); events.append(line)
        } onExit: { code, stderr in
            XCTAssertEqual(events, ["first", "second"])
            XCTAssertEqual(code, 0); XCTAssertEqual(stderr, ""); done.fulfill()
        }
        wait(for: [done], timeout: 5)
    }
    func testNonzeroExitPreservesDiagnostics() throws {
        let done = expectation(description: "exit")
        _ = try ProcessPump.launch(executable: URL(fileURLWithPath: "/bin/sh"), arguments: ["-c", "printf 'failure' >&2; exit 7"]) { _ in } onExit: { code, stderr in
            XCTAssertEqual(code, 7); XCTAssertEqual(stderr, "failure"); done.fulfill()
        }
        wait(for: [done], timeout: 5)
    }
}
