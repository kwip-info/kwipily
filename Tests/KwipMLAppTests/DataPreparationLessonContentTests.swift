import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataPreparationLessonContentTests: XCTestCase {
    @MainActor func testEveryDeliveredDataOperatorHasFoundationOrDedicatedTeachingCoverage() throws {
        let expected: Set<String> = [
            "kwip.data.formula", "kwip.data.partition", "kwip.data.preprocess", "kwip.data.sourceCSV",
            "kwip.data.sourceFiles", "kwip.data.columns", "kwip.data.filterRows", "kwip.data.sampleRows",
            "kwip.data.join", "kwip.data.aggregate", "kwip.data.view"
        ]
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let addedPracticeOperators = Set(try DataEngineeringPractice.catalog().lessons.map(\.operatorID))
        XCTAssertEqual(addedPracticeOperators, Set(["kwip.data.filterRows", "kwip.data.formula", "kwip.data.outputDelimited"]))
        XCTAssertTrue(addedPracticeOperators.isSubset(of: expected.union(["kwip.data.outputDelimited"])), "Rewritten exercises reuse current tools, not new capability counts")
        let dedicatedLessons: [String:String] = [
            "kwip.data.textEncode":"text-encoding", "kwip.data.sequenceWindows":"sequence-windows", "kwip.data.union":"union", "kwip.data.sort":"sort", "kwip.data.unique":"unique",
            "kwip.data.reshapeTable":"reshapeTable", "kwip.data.splitText":"splitText",
            "kwip.data.prepareMedia":"prepare-media", "kwip.data.extractText":"extract-text",
            "kwip.data.transcribeAudio":"transcribe-audio", "kwip.data.classifyAudio":"classify-audio",
            "kwip.data.gameRules":"game-rules", "kwip.data.decide":"typed-decisions"
        ]
        let dedicatedOperators = Set(dedicatedLessons.keys)
        XCTAssertTrue(dedicatedOperators.isDisjoint(with:expected.union(addedPracticeOperators)), "Dedicated lessons must not inflate foundation or historical practice coverage")
        XCTAssertEqual(Set(registry.operators.filter {$0.id != DataFlowSourceOperators.capturedTable}.map(\.id)), expected.union(addedPracticeOperators).union(dedicatedOperators))
        for (operatorID,lessonID) in dedicatedLessons {
            XCTAssertEqual(ComponentLessonDestination.entryID(type:operatorID),lessonID)
            let item = try XCTUnwrap(LearnCatalogContent.items.first {$0.id == lessonID})
            XCTAssertEqual(item.category,lessonID == "typed-decisions" ? "Training & evaluation":"Data preparation")
            XCTAssertEqual(item.kind,"Practice")
        }
        XCTAssertEqual(DataPreparationLessonContent.operatorCoverage, expected)
        XCTAssertEqual(DataPreparationLessonContent.topics.count, 8)
        XCTAssertEqual(Set(DataPreparationLessonContent.topics.map(\.id)).count, 8)
    }

    func testCurrentColumnLessonTeachesOneGridAndNoRetiredTools() throws {
        let topic = try XCTUnwrap(DataPreparationLessonContent.topics.first { $0.id == "clean-and-check" })
        XCTAssertTrue(topic.operators.contains("kwip.data.columns"))
        XCTAssertFalse(topic.operators.contains("kwip.data.selectRename"))
        XCTAssertFalse(topic.operators.contains("kwip.data.cast"))
        XCTAssertTrue(topic.steps.joined().contains("Columns"))
        XCTAssertTrue(topic.steps.joined().contains("grid"))
        XCTAssertTrue(topic.exampleIDs.contains("columns-grid"))
    }

    func testEachTopicHasAnActionAndAnObservableResultRatherThanACorrectnessClaim() {
        for topic in DataPreparationLessonContent.topics {
            XCTAssertFalse(topic.objective.isEmpty, topic.id)
            XCTAssertFalse(topic.terms.isEmpty, topic.id)
            XCTAssertGreaterThanOrEqual(topic.steps.count, 3, topic.id)
            XCTAssertFalse(topic.inspect.isEmpty, topic.id)
            XCTAssertFalse(topic.exampleIDs.isEmpty, topic.id)
            XCTAssertFalse(topic.isAssessed, topic.id)
        }
        XCTAssertTrue(DataPreparationLessonContent.disclosure.contains("actual run"))
        XCTAssertTrue(DataPreparationLessonContent.disclosure.contains("not scored"))
    }

    func testLearnedPreparationAndFinalTestAreExplainedWithoutImplyingHeldOutFitting() {
        let fitted = DataPreparationLessonContent.topics.first { $0.id == "learn-from-training" }!
        XCTAssertTrue(fitted.steps.joined(separator: " ").contains("training"))
        XCTAssertTrue(fitted.inspect.contains("same saved"))
        let final = DataPreparationLessonContent.topics.first { $0.id == "protect-final-test" }!
        XCTAssertTrue(final.objective.contains("unseen"))
        XCTAssertTrue(final.steps.joined(separator: " ").contains("once"))
        XCTAssertTrue(final.inspect.contains("unavailable"))
    }

    func testTableReadersAndViewDistinguishInspectionWorkingSamplesAndDatasetChanges() throws {
        let reader = try XCTUnwrap(DataPreparationLessonContent.topics.first { $0.id == "read-tables" })
        XCTAssertTrue(reader.steps.joined(separator: " ").contains("private credentials"))
        XCTAssertTrue(reader.steps.joined(separator: " ").contains("compatibility"))
        let view = try XCTUnwrap(DataPreparationLessonContent.topics.first { $0.id == "explore-complete-table" })
        XCTAssertEqual(view.operators, ["kwip.data.view"])
        XCTAssertEqual(view.exampleIDs, [TableInputViewPracticeExample.exampleID])
        XCTAssertTrue(view.steps.joined(separator: " ").contains("source-order prefix"))
        XCTAssertTrue(view.steps.joined(separator: " ").contains("without replacement"))
        XCTAssertTrue(view.inspect.contains("64"))
        XCTAssertFalse(view.isAssessed)
    }
    func testFileInputTopicLinksActualPracticeAndDisclosesRepresentationBoundaries() throws {
        let topic = try XCTUnwrap(DataPreparationLessonContent.topics.first { $0.id == "read-file-contents" })
        XCTAssertEqual(topic.operators, ["kwip.data.sourceFiles"])
        XCTAssertEqual(topic.exampleIDs, [FileInputViewPracticeExample.exampleID])
        XCTAssertEqual(DataPreparationLessonContent.exampleTitle(FileInputViewPracticeExample.exampleID), "Open file exploration practice")
        XCTAssertTrue(topic.steps.joined().contains("metadata-only"))
        XCTAssertTrue(topic.steps.joined().contains("files before extraction"))
        XCTAssertTrue(topic.steps.joined().contains("does not prepare them for model training"))
        XCTAssertFalse(topic.isAssessed)
    }

}
