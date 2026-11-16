import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LazyLinearLessonRegistrationTests: XCTestCase {
    func testMeasuredTaskDiscoverableSeparatelyFromClassifierExample() throws {
        let task = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "lazy-linear-lesson" })
        XCTAssertEqual(task.kind, "Construction")
        XCTAssertTrue(task.matches(section: "Library", search: "inferred"))
        XCTAssertFalse(task.matches(section: "Examples", search: "inferred"))
        XCTAssertEqual(LearnCatalogContent.items.filter { $0.id == "lazy-linear" }.count, 1)
        XCTAssertFalse(LearnCatalogContent.displayedComponentExampleEntries.contains { $0.componentID == "lazy_linear" })
    }
    @MainActor func testLessonRequiresSpecificRuntimeAndExplainsActualStateChecks() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let lesson = TensorLearningStore(workspace: workspace)
        XCTAssertTrue(lesson.start(operation: LazyLinearAssessment.operation), lesson.statusMessage)
        XCTAssertTrue(lesson.runtimeReadinessError?.contains("LazyLinear") == true)
        XCTAssertFalse(lesson.canSubmit)
        let guide = LessonAudienceGuide.make(operation: LazyLinearAssessment.operation)
        XCTAssertTrue(guide.goal.localizedCaseInsensitiveContains("width"))
        let terms = Set(guide.terms.map { $0.name.lowercased() })
        XCTAssertTrue(terms.isSuperset(of: ["inferred", "weights", "initialization", "seed"]))
        XCTAssertTrue(LessonAudienceGuide.requirements(operation: LazyLinearAssessment.operation, fallback: "").contains("four"))
    }
}
