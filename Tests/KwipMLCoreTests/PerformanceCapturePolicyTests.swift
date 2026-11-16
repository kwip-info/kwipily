import XCTest
@testable import KwipMLCore

final class PerformanceCapturePolicyTests: XCTestCase {
    func testHistoricalPolicyStaysOffAndVersionTwoRoundtripsThroughScopedPreferences() throws {
        let decoder = JSONDecoder(), encoder = JSONEncoder()
        let old = Data(#"{"schemaVersion":1,"activationValues":false,"predictionRows":true}"#.utf8)
        let legacy = try decoder.decode(TrainingSampleCapturePolicy.self, from: old)
        XCTAssertFalse(legacy.performanceMeasurements)
        XCTAssertEqual(legacy.schemaVersion, 1)
        XCTAssertEqual(try decoder.decode(JSONValue.self, from: encoder.encode(legacy)), try decoder.decode(JSONValue.self, from: old))
        let captured = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false, performanceMeasurements: true)
        XCTAssertEqual(captured.schemaVersion, 2)
        let global = try ExecutionPreferences(sampleCapture: captured)
        let override = try ProjectExecutionOverrides(sampleCapture: captured)
        XCTAssertEqual(try decoder.decode(ExecutionPreferences.self, from: encoder.encode(global)).sampleCapture, captured)
        XCTAssertEqual(try decoder.decode(ProjectExecutionOverrides.self, from: encoder.encode(override)).sampleCapture, captured)
        let off = try decoder.decode(TrainingSampleCapturePolicy.self, from: Data(#"{"schemaVersion":2,"activationValues":false,"predictionRows":false,"performanceMeasurements":false}"#.utf8))
        XCTAssertEqual(off.schemaVersion, 2)
        XCTAssertFalse(off.performanceMeasurements)
        XCTAssertEqual(try decoder.decode(TrainingSampleCapturePolicy.self, from: encoder.encode(off)), off)
    }
    func testMissingUnknownNonbooleanOrOldSchemaPerformanceChoiceRefuses() {
        for json in [
            #"{"schemaVersion":2,"activationValues":true,"predictionRows":true}"#,
            #"{"schemaVersion":2,"activationValues":true,"predictionRows":true,"performanceMeasurements":1}"#,
            #"{"schemaVersion":1,"activationValues":true,"predictionRows":true,"performanceMeasurements":true}"#,
            #"{"schemaVersion":2,"activationValues":true,"predictionRows":true,"performanceMeasurements":true,"other":false}"#
        ] { XCTAssertThrowsError(try JSONDecoder().decode(TrainingSampleCapturePolicy.self, from: Data(json.utf8))) }
    }
}
