import XCTest
import KwipMLCore
@testable import KwipMLApp
final class TensorRoutingLessonAudienceTests:XCTestCase {
    func testSixFindableTasksUseConciseNoCodePresetInstructionsAndDefinedTerms()throws {
        let item=try XCTUnwrap(LearnCatalogContent.items.first {$0.id == "tensor-routing"})
        XCTAssertEqual(item.kind,"Construction")
        XCTAssertEqual(item.category,"Tensors & linear algebra")
        for task in TensorRoutingLearningTask.allCases {
            let text=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for header in ["**Build**","**Configure**","**Check**"] {XCTAssertTrue(text.contains(header))}
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["array","port","observation","gradient"]))
            XCTAssertLessThan(guide.steps[1].count,600);XCTAssertFalse(guide.steps[1].contains("An array is"))
            XCTAssertTrue(item.matches(section:"Library",search:task.title))
            XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:task == .einsumCombine ? "leftGradient":"inputGradient"))
            if task.kind == "einsum" {XCTAssertTrue(text.contains("Arrangement preset"));XCTAssertFalse(guide.steps[1].contains("nab,nbc->nac"))}
        }
        for id in ["split","chunk","gather","einsum"] {
            let entry=try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first {$0.componentID == id})
            XCTAssertEqual(entry.item.category,"Tensors & linear algebra")
            XCTAssertNotNil(LearnCatalogContent.displayedComponentExampleEntries.first {$0.componentID == id})
        }
    }
}
