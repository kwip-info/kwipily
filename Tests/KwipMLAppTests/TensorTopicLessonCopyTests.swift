import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TensorTopicLessonCopyTests:XCTestCase {
    func testEveryCurrentTensorConstructionUsesItsPublicTaskAndAssessmentOnlyDirections()throws {
        let entries=LearningLibraryPresentation.items.filter {$0.category=="Tensors & linear algebra" && $0.kind=="Construction"}
        var operations=Set<String>()
        for entry in entries {
            for choice in ConstructionLessonCatalog.groups[entry.id] ?? [] {
                operations.insert(choice.id)
                let copy=try XCTUnwrap(TensorTopicLessonCopy.task(choice.id),choice.id)
                let content=try XCTUnwrap(LessonContentCatalog.content(entryID:entry.id,operation:choice.id),choice.id)
                XCTAssertEqual(content.kind,"Construction")
                XCTAssertEqual(content.steps.map(\.id),["goal","build.0","build.1","build.2","check"])
                let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:"\n")
                XCTAssertTrue(text.contains(copy.controls),choice.id)
                XCTAssertTrue(text.contains("Submit assessment"),choice.id)
                XCTAssertFalse(text.contains("Choose Check construction"),choice.id)
                XCTAssertTrue(text.contains("Private assessment fixture tables stay hidden"),choice.id)
                XCTAssertFalse(text.contains("inspect the output and input-influence tables"),choice.id)
                XCTAssertTrue(text.contains("sample shape") || text.contains("sample dimensions"),choice.id)
            }
        }
        XCTAssertEqual(operations.count,21)
        XCTAssertNil(TensorTopicLessonCopy.task("not-an-operation"))
        XCTAssertTrue(TensorTopicLessonCopy.task("flatten")!.controls.contains("Method → Flatten axes"))
        XCTAssertTrue(TensorTopicLessonCopy.task("chunk-uneven-parts")!.controls.contains("part4"))
        XCTAssertTrue(TensorTopicLessonCopy.task("einsum-combine-tables")!.controls.contains("Combine tables"))
    }
    func testEveryCurrentTensorExampleHasPurposeRealLauncherAndCapturedEvidenceDirections()throws {
        let entries=LearningLibraryPresentation.items.filter {$0.category=="Tensors & linear algebra" && $0.kind=="Example"}
        XCTAssertEqual(entries.count,12)
        for entry in entries {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:entry.id),entry.id)
            XCTAssertGreaterThan(content.steps.count,3,entry.id)
            XCTAssertEqual(Set(content.steps.map(\.id)).count,content.steps.count)
            let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:"\n")
            XCTAssertTrue(text.contains("unscored"),entry.id)
            XCTAssertTrue(text.contains("Run Flow"),entry.id)
            XCTAssertTrue(text.contains("captured"),entry.id)
            XCTAssertFalse(text.contains("Use the lesson's exercise choices"),entry.id)
            if let bundled=LearnCatalogContent.componentExampleEntries.first(where:{$0.componentID==entry.id}) {
                XCTAssertTrue(text.contains(bundled.learning.buttonTitle),entry.id)
                let document=try PreparedWorkflowExample.component(entry.id).document(device:"cpu")
                XCTAssertTrue(document.nodes.contains {if case .train=$0.payload {return true};return false},entry.id)
            }
        }
    }
}
