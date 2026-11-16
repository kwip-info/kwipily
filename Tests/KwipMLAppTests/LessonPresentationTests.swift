import XCTest
@testable import KwipMLApp
final class LessonPresentationTests: XCTestCase {
    func testRestrictedPaletteShowsEveryPermittedToolWithoutChangingGeneralDefault() {
        XCTAssertEqual(ComponentRibbonCatalog.initialScope(restricted: true), .all)
        XCTAssertEqual(ComponentRibbonCatalog.initialScope(restricted: false), .common)
    }
    func testProseSeparatesMathAndListsWithoutChangingTheirContent() {
        let blocks = LessonTextLayout.blocks("# Task\n\nKeep **observations** independent.\n\n- Preserve axis 0\n- Use 1.0 exactly\n\n```\ny = P x\n∂L/∂x = Pᵀ g\n```")
        XCTAssertEqual(blocks.map(\.kind), [.heading, .paragraph, .bullet, .bullet, .code])
        XCTAssertEqual(blocks.last?.text, "y = P x\n∂L/∂x = Pᵀ g")
        XCTAssertEqual(blocks[3].text, "Use 1.0 exactly")
    }
    func testPlainExplanationIsReadableAndUnchanged() {
        let prose = "Each operation preserves the batch axis. Values remain 1.0 and 2.0. The gradient is Pᵀ g."
        XCTAssertEqual(LessonTextLayout.blocks(prose).map(\.text).joined(separator: " "), prose)
    }
}
