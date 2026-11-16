import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LayerTopicLessonCopyTests:XCTestCase {
    func testEveryCurrentLayerEntryHasSpecificReadingAndPreservesItsDeliveryType()throws {
        let entries=LearningLibraryPresentation.items.filter {$0.category=="Layers & activations"}
        XCTAssertEqual(entries.count,79)
        var operations=Set<String>()
        for entry in entries {
            let choices=ConstructionLessonCatalog.groups[entry.id] ?? []
            for choice in choices {
                operations.insert(choice.id)
                let copy=try XCTUnwrap(LayerTopicLessonCopy.task(choice.id),choice.id)
                let content=try XCTUnwrap(LessonContentCatalog.content(entryID:entry.id,operation:choice.id),choice.id)
                XCTAssertEqual(content.kind,"Construction",entry.id)
                let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:"\n")
                XCTAssertTrue(text.contains(copy.controls),choice.id)
                XCTAssertTrue(text.contains("Submit assessment"),choice.id)
                XCTAssertTrue(text.contains("Private fixture tables stay hidden"),choice.id)
                XCTAssertEqual(Set(content.steps.map(\.id)).count,content.steps.count,choice.id)
            }
            guard !ConstructionLessonCatalog.affineEntries.contains(entry.id),choices.isEmpty else {continue}
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:entry.id),entry.id)
            XCTAssertGreaterThan(content.steps.count,3,entry.id)
            XCTAssertEqual(content.kind,entry.kind)
            let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:"\n")
            XCTAssertFalse(text.contains("Use the lesson's exercise choices to open its supplied canvas."),entry.id)
            XCTAssertFalse(text.contains("Run the example or use Check construction where the lesson supports it."),entry.id)
            XCTAssertTrue(text.contains("unscored") || text.contains("not an assessment"),entry.id)
        }
        XCTAssertEqual(operations.count,60)
        XCTAssertNil(LayerTopicLessonCopy.task("not-a-task"))
    }
    func testWorkedExamplesRemainGuidedChecksWithoutBorrowingAssessmentClaims()throws {
        for task in ModuleCompositionLearningTask.allCases {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:"module-composition-examples",operation:task.rawValue))
            XCTAssertEqual(content.kind,"Example")
            let text=content.steps.flatMap(\.blocks).map(\.text).joined(separator:" ")
            XCTAssertTrue(text.contains("Check construction"))
            XCTAssertTrue(text.contains("not an assessment"))
            XCTAssertFalse(text.contains("Choose Submit assessment"))
        }
        for id in RecurrentAuthoringConfiguration.componentIDs {
            let entry=try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first {$0.componentID==id})
            XCTAssertFalse(entry.learning.paragraphs.joined().contains("Guided construction, optional-hint practice"))
            XCTAssertTrue(entry.learning.paragraphs.joined().contains("unscored"))
        }
    }
    func testFreshWorkedBlockExampleKeepsPreparedGuidedModeAndOriginalEdits()throws {
        _=NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(storageOverride:root,detectCapabilities:false)})
        let source=ProjectWindowSession(workspace:WorkspaceStore(storageOverride:root,detectCapabilities:false))
        for task in ModuleCompositionLearningTask.allCases {
            let original=try registry.openLesson(entryID:"module-composition-examples",operation:task.rawValue,source:source)
            TensorLearningView(lesson:original.session.tensor,workspace:original.session.workspace).synchronizeDeliveryMode()
            XCTAssertTrue(original.session.tensor.isActive)
            XCTAssertEqual(original.session.tensor.draft?.mode,.guided)
            XCTAssertNotNil(original.session.tensor.draft?.compositionModel)
            XCTAssertTrue(original.session.tensor.drafts.update {$0.notes="Keep original work"})
            XCTAssertTrue(original.session.save())
            let before=original.session.tensor.draft
            let fresh=try registry.startFreshLesson(entryID:"module-composition-examples",workspace:original.session.workspace)
            TensorLearningView(lesson:fresh.session.tensor,workspace:fresh.session.workspace).synchronizeDeliveryMode()
            XCTAssertEqual(fresh.session.tensor.draft?.operation,task.rawValue)
            XCTAssertEqual(fresh.session.tensor.draft?.mode,.guided)
            XCTAssertNotNil(fresh.session.tensor.draft?.compositionModel)
            XCTAssertNotEqual(fresh.session.tensor.draft?.projectID,before?.projectID)
            XCTAssertEqual(original.session.tensor.draft,before)
            XCTAssertEqual(fresh.session.tensor.draft?.notes,"")
            _=registry.close(fresh);_=registry.close(original)
        }
        XCTAssertThrowsError(try registry.openLesson(entryID:"module-composition-examples",operation:"reshape",source:source))
    }

    func testReviewedRunnableBundleFactoriesKeepActualTrainAndSourceContracts()throws {
        let entries=LearnCatalogContent.componentExampleEntries.filter {$0.item.category=="Layers & activations"}
        for entry in entries where LearningLibraryPresentation.items.contains(where:{$0.id==entry.componentID && $0.kind=="Example"}) {
            let content=try XCTUnwrap(LessonContentCatalog.content(entryID:entry.componentID),entry.componentID)
            let text=content.steps.flatMap(\.blocks).map(\.text).joined()
            if LayerLegacyExampleLessonCopy.examples[entry.componentID] == nil {XCTAssertTrue(text.contains(entry.learning.buttonTitle),entry.componentID)}
            let document=try PreparedWorkflowExample.component(entry.componentID).document(device:"cpu")
            XCTAssertTrue(document.nodes.contains {if case .train=$0.payload {return true};return false},entry.componentID)
            XCTAssertFalse(PreparedWorkflowExample.component(entry.componentID).sourceResources.isEmpty,entry.componentID)
        }
    }
}
