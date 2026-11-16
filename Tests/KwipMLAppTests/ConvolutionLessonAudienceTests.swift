import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ConvolutionLessonAudienceTests: XCTestCase {
    func testThreeTasksHaveVisualControlsDefinedTermsAndAutomaticChecks() {
        for task in ConvolutionLearningTask.allCases.filter({!$0.isTransposed}) {
            let required=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            XCTAssertTrue(required.contains("Check construction"))
            XCTAssertTrue(required.contains("Kernel"))
            XCTAssertTrue(required.contains("2"))
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertTrue(guide.goal.contains("window"))
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["channel","spatial direction","kernel","shared weights","gradient"]))
        }
    }
    @MainActor func testEachTaskRefusesMissingDimensionSpecificRuntime() throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        for task in ConvolutionLearningTask.allCases.filter({!$0.isTransposed}) {
            XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
            XCTAssertNotNil(lesson.runtimeReadinessError)
            XCTAssertFalse(lesson.canSubmit)
        }
    }
}
