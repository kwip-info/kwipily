import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GradientRunArgumentsTests: XCTestCase {
    func testOffPreservesLegacyArgumentsAndEnabledRequiresVerifiedSupport() throws {
        XCTAssertEqual(try GradientRunArguments.arguments(config: nil, supportedDevices: [], device: "cpu"), [])
        let config = GradientCaptureConfiguration(everyEpochs: 3, maxBatchesPerEpoch: 2, maxCaptures: 12)
        XCTAssertThrowsError(try GradientRunArguments.arguments(config: config, supportedDevices: [], device: "cpu"))
        XCTAssertThrowsError(try GradientRunArguments.arguments(config: config, supportedDevices: ["cpu"], device: "mps"))
        XCTAssertEqual(try GradientRunArguments.arguments(config: config, supportedDevices: ["cpu"], device: "cpu"), ["--capture-gradients", "--gradient-every-epochs", "3", "--gradient-max-batches-per-epoch", "2", "--gradient-max-captures", "12"])
    }
    func testInvalidBudgetDoesNotBecomeAWorkerRequest() {
        let config = GradientCaptureConfiguration(maxCaptures: 0)
        XCTAssertThrowsError(try GradientRunArguments.arguments(config: config, supportedDevices: ["cpu"], device: "cpu"))
    }
}
