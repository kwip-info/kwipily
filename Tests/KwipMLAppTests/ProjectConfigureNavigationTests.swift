import XCTest
@testable import KwipMLApp

final class ProjectConfigureNavigationTests: XCTestCase {
    func testUniversalPagesExposeCanvasControlsWithoutDuplicateOverlay() {
        XCTAssertEqual(ProjectConfigurePage.pages(universal: true), [.identity, .files, .schedules, .webhooks, .resources, .capture, .workspace])
        XCTAssertEqual(ProjectConfigurePage.matches("EXPORT", universal: true), [.files])
        XCTAssertTrue(ProjectConfigurePage.matches("run setup", universal: true).isEmpty)
        XCTAssertEqual(ProjectConfigurePage.matches("workspace", universal: true), [.workspace])
        XCTAssertTrue(ProjectConfigurePage.matches("flow direction", universal: true).isEmpty)
        XCTAssertTrue(ProjectConfigurePage.matches("node detail", universal: true).isEmpty)
        XCTAssertEqual(ProjectConfigurePage.matches("ports", universal: true), [.workspace])
        XCTAssertEqual(ProjectConfigurePage.matches("minimap", universal: true), [.workspace])
    }
    func testLegacyPagesKeepIterationAndProjectPanelControlsSearchable() {
        XCTAssertEqual(ProjectConfigurePage.pages(universal: false), [.identity, .training, .resources, .capture, .workspace])
        XCTAssertEqual(ProjectConfigurePage.matches("batch seed", universal: false), [.training])
        XCTAssertEqual(ProjectConfigurePage.matches("CPU threads", universal: false), [.resources])
        XCTAssertEqual(ProjectConfigurePage.matches("gradient", universal: false), [.capture])
        XCTAssertEqual(ProjectConfigurePage.matches("results panel", universal: false), [.workspace])
        XCTAssertTrue(ProjectConfigurePage.matches("export", universal: false).isEmpty)
        XCTAssertTrue(ProjectConfigurePage.matches("unknown setting", universal: false).isEmpty)
    }
}
