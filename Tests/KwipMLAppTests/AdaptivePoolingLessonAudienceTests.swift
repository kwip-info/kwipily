import XCTest
import KwipMLCore
@testable import KwipMLApp
final class AdaptivePoolingLessonAudienceTests:XCTestCase {
    func testAllFourControlsHaveAccessibleNoCodeInstructionsAndDefinedTerms() {
        for task in AdaptivePoolingLearningTask.allCases {
            let text=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for phrase in ["**Build**","**Configure**","**Check**","Select","Check construction","automatically"] {XCTAssertTrue(text.contains(phrase),task.rawValue+" "+phrase)}
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["channel","gradient"]),task.rawValue)
            XCTAssertFalse(guide.steps.isEmpty)
            XCTAssertNotNil(TensorLearningTask.title(for:task.rawValue))
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
        }
    }
    func testDiscoveryShowsUniqueAdaptiveLabAndIncludesEveryTaskInSearch() {
        let entry=LearnCatalogContent.items.first {$0.id == "adaptive-pooling"}
        XCTAssertNotNil(entry)
        XCTAssertEqual(entry?.kind,"Construction")
        for task in AdaptivePoolingLearningTask.allCases {
            XCTAssertTrue(entry?.matches(section:"Library",search:task.title) == true,task.rawValue)
        }
    }
    @MainActor func testOptionalWinningPositionPortRetainsIntegerIdentityAndAvailability()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(storageOverride:storage,detectCapabilities:false),lesson=TensorLearningStore(workspace:workspace)
        let task=AdaptivePoolingLearningTask.adaptiveMaximumTwo
        XCTAssertTrue(lesson.start(operation:task.rawValue),lesson.statusMessage)
        let canvas=try XCTUnwrap(lesson.canvas)
        canvas.addNode(type:task.componentID,x:0,y:0)
        let node=try XCTUnwrap(canvas.graph.nodes.first)
        XCTAssertTrue(canvas.configure(nodeID:node.id,values:task.configuration))
        XCTAssertTrue(canvas.canvasPortDescription(nodeID:node.id,port:"indices").contains("int64"))
        var disabled=task.configuration;disabled["return_indices"]=false
        XCTAssertTrue(canvas.configure(nodeID:node.id,values:disabled))
        XCTAssertTrue(canvas.canvasPortDescription(nodeID:node.id,port:"indices").localizedCaseInsensitiveContains("unavailable"))
        XCTAssertFalse(canvas.registry.components.contains {$0.id == "avg_pool2d"},"The task palette must remain restricted")
    }

    func testCommonFamilyPhraseFindsAdaptiveLessons()throws {
        let item=try XCTUnwrap(LearnCatalogContent.items.first {$0.id == "adaptive-pooling"})
        XCTAssertEqual(item.kind,"Construction")
        XCTAssertTrue(item.matches(section:"Library",search:"adaptive pooling"))
        XCTAssertTrue(item.matches(section:"PyTorch",search:"adaptive pooling"))
        for task in AdaptivePoolingLearningTask.allCases where task.componentID.hasPrefix("adaptive_") {
            let example=try XCTUnwrap(LearnCatalogContent.componentExampleEntries.first {$0.componentID == task.componentID})
            XCTAssertTrue(example.item.matches(section:"Examples",search:"adaptive pooling"),task.rawValue)
        }
    }

}
