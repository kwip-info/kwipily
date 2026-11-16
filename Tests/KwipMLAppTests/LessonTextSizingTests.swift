import XCTest
@testable import KwipMLApp

final class LessonTextSizingTests: XCTestCase {
    func testReadableDefaultAndBoundedPreference() {
        XCTAssertEqual(LessonTypography.defaultSize, 16)
        XCTAssertEqual(LessonTypography.normalized(0), 12)
        XCTAssertEqual(LessonTypography.normalized(99), 24)
        XCTAssertEqual(LessonTypography.normalized(18), 18)
        XCTAssertEqual(LessonTypography(bodySize: 24).size(12), 24)
        XCTAssertEqual(LessonTypography(bodySize: 24).size(10), 20)
    }
    func testPreferencePersistsSeparatelyFromLessonProgress() {
        let name = "LessonTextSizingTests." + UUID().uuidString
        let defaults = UserDefaults(suiteName: name)!
        defer { defaults.removePersistentDomain(forName: name) }
        XCTAssertEqual(LessonTypography.load(defaults).bodySize, 16)
        defaults.set(22, forKey: LessonTypography.preferenceKey)
        XCTAssertEqual(LessonTypography.load(UserDefaults(suiteName: name)!).bodySize, 22)
        defaults.set(1000, forKey: LessonTypography.preferenceKey)
        XCTAssertEqual(LessonTypography.load(defaults).bodySize, 24)
    }
}
