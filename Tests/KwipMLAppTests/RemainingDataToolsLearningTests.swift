import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RemainingDataToolsLearningTests:XCTestCase {
    func testExactLessonDestinationsSearchAndLegacyOutputLink()throws {
        let destinations=["kwip.data.sort":"sort","kwip.data.unique":"unique","kwip.data.reshapeTable":"reshapeTable","kwip.data.splitText":"splitText","kwip.data.preprocess":"preprocess","kwip.data.formula":"formula","kwip.data.outputDelimited":"table-output","output":"model-output"]
        for (type,lesson) in destinations {
            XCTAssertEqual(ComponentLessonDestination.entryID(type:type),lesson)
            XCTAssertEqual(LearnCatalogContent.items.filter{$0.id==lesson}.count,1)
        }
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"data02-output"))
        XCTAssertFalse(LearnCatalogContent.items.contains{$0.id=="data02-output"},"Historical output links remain accessible without a duplicate current lesson.")
        let formula=try XCTUnwrap(LearnCatalogContent.items.first{$0.id=="formula"})
        XCTAssertTrue(formula.matches(section:"Library",search:"timezone microsecond"))
        let preprocess=try XCTUnwrap(LearnCatalogContent.items.first{$0.id=="preprocess"})
        XCTAssertTrue(preprocess.matches(section:"Library",search:"Unicode whitespace"))
    }
    func testNewExamplesOwnTheirSourcesAndKeepPortableDocumentsPrivate()throws {
        let ids=[SortPracticeExample.exampleID,UniquePracticeExample.exampleID,ReshapeTablePracticeExample.exampleID,ReshapeTablePracticeExample.wideExampleID,SplitTextPracticeExample.exampleID,FormulaTemporalPracticeExample.exampleID,PreprocessPracticeExample.cleanupExampleID,ModelOutputPracticeExample.exampleID,TableOutputPracticeExample.exampleID]
        for id in ids {
            let example=PreparedWorkflowExample.component(id)
            let document=try example.document(device:"cpu")
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document,id)
            XCTAssertFalse(example.sourceResources.isEmpty,id)
            let sourceAddresses=Set(document.nodes.filter {node in
                if case .dataOperator(let operatorID,_) = node.payload {return operatorID=="kwip.data.sourceCSV"}
                return false
            }.map(\.address))
            XCTAssertEqual(Set(example.sourceResources.map(\.address)),sourceAddresses,id)
            let json=String(decoding:try WorkflowCodec.encode(document),as:UTF8.self)
            XCTAssertFalse(json.contains("/Users/"),id)
            XCTAssertFalse(json.contains("destinationBindings"),id)
        }
    }
}
