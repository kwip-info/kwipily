import XCTest
@testable import KwipMLApp

final class RuntimeSettingsPresentationTests: XCTestCase {
    func testRuntimeSearchFindsExistingExecutionPage() {
        for query in ["runtime", "refresh runtime", "verified devices", "PyTorch"] {
            XCTAssertEqual(SettingsPage.matches(query), [.execution], query)
        }
    }
    func testUnavailableContractDoesNotPresentDefaultCPUOrCountsAsVerified() {
        let presentation = RuntimeSettingsPresentation(status: "Probe failed", version: "2.14.1", contractVerified: false,
            devices: ["cpu"], componentCount: 30, isBusy: false, isChecking: false)
        XCTAssertEqual(presentation.devicesText, "Not verified")
        XCTAssertEqual(presentation.componentsText, "Not verified")
        XCTAssertEqual(presentation.versionText, "Not verified")
        XCTAssertEqual(presentation.status, "Probe failed")
        XCTAssertTrue(presentation.canRefresh)
    }
    func testVerifiedStatusPresentsActualDevicesCountAndVersion() {
        let presentation = RuntimeSettingsPresentation(status: "Verified", version: "2.14.1", contractVerified: true,
            devices: ["cpu", "mps"], componentCount: 27, isBusy: false, isChecking: false)
        XCTAssertEqual(presentation.devicesText, "CPU, Apple GPU (MPS)")
        XCTAssertEqual(presentation.componentsText, "27 verified components")
        XCTAssertEqual(presentation.versionText, "PyTorch 2.14.1")
    }
    func testRefreshUnavailableDuringRunOrExistingProbe() {
        for (busy, checking) in [(true, false), (false, true), (true, true)] {
            let presentation = RuntimeSettingsPresentation(status: "Checking", version: nil, contractVerified: false,
                devices: [], componentCount: 0, isBusy: busy, isChecking: checking)
            XCTAssertFalse(presentation.canRefresh)
        }
    }
}
