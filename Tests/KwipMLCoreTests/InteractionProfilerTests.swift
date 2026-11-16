import XCTest
@testable import KwipMLCore

final class InteractionProfilerTests: XCTestCase {
    func testDisabledRecorderAndThrowingOperationPreserveBehavior() throws {
        var time: UInt64 = 0
        let recorder = InteractionProfiler(clock: { time })
        XCTAssertNil(recorder.begin(.canvasBody))
        XCTAssertEqual(recorder.measure(.canvasBody) { 42 }, 42)
        recorder.start(maximumSamples: 10)
        enum Failure: Error { case expected }
        XCTAssertThrowsError(try recorder.measure(.authoringCommand) { time = 5_000_000; throw Failure.expected })
        let report = recorder.stop()
        XCTAssertEqual(report.samples.count, 1)
        XCTAssertEqual(report.samples.first?.milliseconds, 5)
        XCTAssertNil(recorder.begin(.canvasBody))
    }
    func testBoundedCaptureAndOldTokensCannotLeakIntoNextSession() {
        var time: UInt64 = 0
        let recorder = InteractionProfiler(clock: { time })
        recorder.start(maximumSamples: 2)
        recorder.scenario = .selection
        let old = recorder.begin(.canvasBody)
        for _ in 0..<3 {
            let token = recorder.begin(.selection)
            time += 1_000_000
            recorder.end(token)
        }
        let report = recorder.stop()
        XCTAssertEqual(report.samples.count, 2)
        XCTAssertEqual(report.droppedSamples, 1)
        XCTAssertTrue(report.samples.allSatisfy { $0.scenario == .selection })
        recorder.start(maximumSamples: 2)
        recorder.end(old)
        XCTAssertTrue(recorder.stop().samples.isEmpty)
    }
    func testTokenCapturesScenarioAndReportContainsNoPayloadFields() throws {
        var time: UInt64 = 100
        let recorder = InteractionProfiler(clock: { time })
        recorder.start()
        recorder.scenario = .configuration
        let token = recorder.begin(.projection)
        recorder.scenario = .idle
        time += 2_000_000
        recorder.end(token)
        let report = recorder.stop()
        XCTAssertEqual(report.samples.first?.scenario, .configuration)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(report)) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["schemaVersion", "samples", "droppedSamples"])
    }
    func testConcurrentProducersRemainBoundedAndAccountForOverflow() {
        let recorder = InteractionProfiler()
        recorder.start(maximumSamples: 100)
        DispatchQueue.concurrentPerform(iterations: 500) { _ in
            recorder.measure(.workerEvent) { _ = 1 + 1 }
        }
        let report = recorder.stop()
        XCTAssertEqual(report.samples.count, 100)
        XCTAssertEqual(report.droppedSamples, 400)
        XCTAssertTrue(report.samples.allSatisfy { $0.milliseconds >= 0 })
    }
    func testDisabledRecordingDoesNotReadClock() {
        var calls = 0
        let recorder = InteractionProfiler(clock: { calls += 1; return 0 })
        for _ in 0..<100 { recorder.measure(.canvasBody) {} }
        XCTAssertEqual(calls, 0)
        XCTAssertTrue(recorder.stop().samples.isEmpty)
    }

}
