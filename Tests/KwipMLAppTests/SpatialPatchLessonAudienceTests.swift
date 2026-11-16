import XCTest
import KwipMLCore
@testable import KwipMLApp
final class SpatialPatchLessonAudienceTests:XCTestCase {
    func testFourFindablePatchTasksExplainControlsAndMeasuredInfluenceWithoutCode()throws {
        let item=try XCTUnwrap(LearnCatalogContent.items.first {$0.id == "spatial-patches"})
        XCTAssertEqual(item.kind,"Construction")
        XCTAssertEqual(item.category,"Tensors & linear algebra")
        XCTAssertTrue(item.matches(section:"Library",search:"spatial patches"))
        for task in SpatialPatchLearningTask.allCases {
            XCTAssertTrue(item.matches(section:"Library",search:task.title))
            let requirements=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for header in ["**Build**","**Configure**","**Check**"] {XCTAssertTrue(requirements.contains(header))}
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertEqual(guide.steps.count,3)
            XCTAssertTrue(guide.steps[1].contains("Patch size"))
            XCTAssertLessThan(guide.steps[1].count,600)
            XCTAssertFalse(guide.steps[1].contains("An array is"))
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["array","observation","channel","patch","stride","dilation","overlap","gradient"]))
            for criterion in ["output","inputGradient"] {XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:criterion))}
            if !task.isUnfold {XCTAssertTrue(requirements.lowercased().contains("add"));XCTAssertTrue(requirements.lowercased().contains("zero"))}
        }
        for id in ["unfold","fold"] {
            let example=try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first {$0.componentID == id})
            XCTAssertEqual(example.item.category,"Tensors & linear algebra")
            XCTAssertNotNil(LearnCatalogContent.displayedComponentExampleEntries.first {$0.componentID == id})
        }
    }
}
