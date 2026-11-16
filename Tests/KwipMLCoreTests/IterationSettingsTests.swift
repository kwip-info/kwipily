import XCTest
@testable import KwipMLCore

final class IterationSettingsTests: XCTestCase {
    func testLegacySettingsGetIterationDefaults() throws {
        let data = Data(#"{"device":"cpu","epochs":5,"featureColumns":[]}"#.utf8)
        let value = try JSONDecoder().decode(ExecutionSettings.self, from: data)
        XCTAssertEqual(value.device, "cpu")
        XCTAssertEqual(value.epochs, 5)
        XCTAssertEqual(value.batchSize, 1024)
        XCTAssertEqual(value.sampleLimit, 0)
        XCTAssertEqual(value.seed, 42)
        XCTAssertEqual(value.splitSeed, 42)
        XCTAssertEqual(value.validationPercent, 20)
        XCTAssertEqual(value.learningRate, 0.03)
    }
    func testIterationSettingsRoundtrip() throws {
        let original = ExecutionSettings(batchSize: 7, sampleLimit: 40, seed: 9, splitSeed: 3, validationPercent: 25, learningRate: 0.01)
        let decoded = try JSONDecoder().decode(ExecutionSettings.self, from: JSONEncoder().encode(original))
        XCTAssertEqual(decoded, original)
    }
}
