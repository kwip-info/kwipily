import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DropoutLearningContentTests:XCTestCase {
    @MainActor func testDedicatedLessonKeepsOriginalScoredTasksAndExampleRoute() {
        XCTAssertEqual(DropoutLearningContent.lessonID,"dropout")
        XCTAssertEqual(DropoutLearningContent.componentID,"dropout")
        XCTAssertEqual(DropoutLearningContent.tasks,[.dropoutTraining,.dropoutEvaluation])
        XCTAssertEqual(Set(DropoutLearningContent.tasks.map(\.rawValue)),["dropout-training-pattern","dropout-evaluation-pattern"])
        XCTAssertTrue(DropoutLearningContent.tasks.allSatisfy(\.isDropout))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:DropoutLearningContent.lessonID))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"remaining-activations"))
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"dropout"),DropoutLearningContent.lessonID)
    }

    func testProbabilityEndpointsAndParentOwnedModeRemainThePublishedContract() throws {
        let definition=try XCTUnwrap(ComponentRegistry.builtIn().component(type:"dropout"))
        XCTAssertEqual(definition.fields.map(\.key),["p"])
        XCTAssertEqual(definition.fields.first?.defaultValue,.number(0.5))
        XCTAssertEqual(definition.fields.first?.min,0)
        XCTAssertEqual(definition.fields.first?.max,1)
        XCTAssertEqual(DropoutLearningContent.tasks.map(\.configuration),[["p":.number(0.5)],["p":.number(0.5)]])
    }
}
