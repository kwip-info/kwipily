import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LoadModelLearningTests:XCTestCase {
    func testConsolidatedDiscoveryAndLessonAreUsable()throws {
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"frozenModel"),"load-model")
        let item=try XCTUnwrap(LearningLibraryPresentation.items.first{$0.id=="load-model"})
        XCTAssertEqual(item.kind,"Practice")
        let content=try XCTUnwrap(LessonContentCatalog.content(entryID:"load-model"))
        XCTAssertGreaterThan(content.steps.count,3)
        let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:" ")
        XCTAssertTrue(text.contains("matching tokenizer"));XCTAssertTrue(text.contains("unscored"))
        XCTAssertTrue(text.contains("LoRA training"));XCTAssertFalse(text.contains("trust_remote_code=True"))
    }
}
