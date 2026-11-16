import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LessonNodeFocusTests:XCTestCase {
    func testMultipleExactTargetsAndTypesNeverUseLabelsOrSubstringMatches() throws {
        let address=WorkflowNodeAddress(domain:.model,localID:"output")
        let focus=LessonNodeFocus(nodeIDs:["a","b"],types:["linear"],addresses:[address])
        XCTAssertTrue(focus.contains(nodeID:"a",type:"relu"))
        XCTAssertTrue(focus.contains(nodeID:"b",type:"relu"))
        XCTAssertTrue(focus.contains(nodeID:try WorkflowCanvasIDs.nodeID(address),type:"output"))
        XCTAssertTrue(focus.contains(nodeID:"another",type:"linear"))
        XCTAssertFalse(focus.contains(nodeID:"not-a",type:"lazy_linear"))
        XCTAssertFalse(focus.contains(nodeID:"output",type:"relu"))
        XCTAssertFalse(LessonNodeFocus().contains(nodeID:"a",type:"linear"))
    }
    func testFocusMetadataDoesNotInvalidateReadingAndOldStepsStillDecode() throws {
        let original=LessonContent.step("a","Read","Look at the two projections")
        var annotated=original;annotated.nodeFocus = .init(nodeIDs:["one","two"])
        XCTAssertEqual(annotated.revision,original.revision)
        XCTAssertEqual(try JSONDecoder().decode(LessonContentStep.self,from:JSONEncoder().encode(original)),original)
        XCTAssertEqual(try JSONDecoder().decode(LessonContentStep.self,from:JSONEncoder().encode(annotated)),annotated)
    }
    func testFirstPracticeHasExactSingleAndMultipleNodeReferences() throws {
        let deck=try XCTUnwrap(LessonContentCatalog.content(entryID:"model-output"))
        let first=try XCTUnwrap(deck.steps.first?.nodeFocus)
        XCTAssertEqual(first.addresses,[ModelOutputPracticeExample.address])
        let follow=try XCTUnwrap(deck.steps.first {$0.title=="Follow the result"}?.nodeFocus)
        XCTAssertTrue(follow.addresses.contains(ModelOutputPracticeExample.address))
        XCTAssertTrue(follow.addresses.contains(.init(domain:.model,localID:"classifier")))
    }
}
