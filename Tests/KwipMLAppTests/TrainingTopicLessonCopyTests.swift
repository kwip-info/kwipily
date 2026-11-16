import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainingTopicLessonCopyTests:XCTestCase {
    func testCurrentTrainingEntriesHaveSpecificReadingAndHonestDelivery()throws {
        let entries=LearningLibraryPresentation.items.filter {$0.category=="Training & evaluation"}
        XCTAssertTrue(entries.contains {$0.id=="fine-tuning"})
        XCTAssertGreaterThanOrEqual(entries.count,29)
        for entry in entries {
            let operations=ConstructionLessonCatalog.groups[entry.id]?.map(\.id) ?? []
            for operation in operations.isEmpty ? [nil]:operations.map(Optional.some) {
                let content=try XCTUnwrap(LessonContentCatalog.content(entryID:entry.id,operation:operation),entry.id)
                let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:"\n")
                XCTAssertGreaterThan(content.steps.count,3,entry.id)
                XCTAssertFalse(text.contains("Use the lesson's exercise choices to open its supplied canvas."),entry.id)
                XCTAssertFalse(text.contains("Run the example or use Check construction where the lesson supports it."),entry.id)
                XCTAssertTrue(text.contains(operation == nil ? "unscored":"Submit assessment"),entry.id)
            }
        }
    }
    func testCheckpointWorkedCopiesPreserveModeControlsAndOriginalEdits()throws {
        _=NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(storageOverride:root,detectCapabilities:false))
        for example in CheckpointLearningExamples.all {
            let original=try registry.openLesson(entryID:"checkpoint-examples",operation:example.task.rawValue,source:source)
            TensorLearningView(lesson:original.session.tensor,workspace:original.session.workspace).synchronizeDeliveryMode()
            XCTAssertEqual(original.session.tensor.draft?.mode,.guided)
            XCTAssertTrue(original.session.tensor.drafts.update {$0.notes="Preserve this attempt"})
            XCTAssertTrue(original.session.save())
            let before=original.session.tensor.draft
            let fresh=try registry.startFreshLesson(entryID:"checkpoint-examples",workspace:original.session.workspace)
            TensorLearningView(lesson:fresh.session.tensor,workspace:fresh.session.workspace).synchronizeDeliveryMode()
            XCTAssertEqual(fresh.session.tensor.draft?.mode,.guided)
            XCTAssertEqual(fresh.session.tensor.draft?.operation,example.task.rawValue)
            XCTAssertNotEqual(fresh.session.tensor.draft?.projectID,before?.projectID)
            XCTAssertEqual(original.session.tensor.draft,before)
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:"checkpoint-examples",operation:example.task.rawValue))
            XCTAssertEqual(content.kind,"Example")
            let text=content.steps.flatMap(\.blocks).map(\.text).joined()
            XCTAssertTrue(text.contains("Check construction"))
            XCTAssertFalse(text.contains("Choose Submit assessment"))
            _=registry.close(fresh);_=registry.close(original)
        }
        XCTAssertThrowsError(try registry.openLesson(entryID:"checkpoint-examples",operation:"reshape",source:source))
    }
    func testBundledTrainingExamplesUseTheirActualRunnableFactories()throws {
        for entry in LearnCatalogContent.componentExampleEntries where entry.item.category=="Training & evaluation" {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:entry.componentID))
            let text=content.steps.flatMap(\.blocks).map(\.text).joined()
            XCTAssertTrue(text.contains(entry.learning.buttonTitle),entry.componentID)
            XCTAssertFalse(text.contains("Connect this recipe"),entry.componentID)
            let example=PreparedWorkflowExample.component(entry.componentID)
            let graph=try example.document(device:"cpu")
            XCTAssertTrue(graph.nodes.contains {if case .train=$0.payload {return true};return false},entry.componentID)
            XCTAssertFalse(example.sourceResources.isEmpty,entry.componentID)
        }
    }
}
