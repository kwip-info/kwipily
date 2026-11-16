import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp
final class TensorCompositionLessonAudienceTests:XCTestCase {
    func testEachTaskHasFindableNoCodeInstructionsAndFriendlyInfluenceLabels()throws {
        let item=try XCTUnwrap(LearnCatalogContent.items.first {$0.id == "tensor-composition"})
        XCTAssertEqual(item.kind,"Construction")
        XCTAssertEqual(LearnCatalogContent.items.filter {$0.id == item.id}.count,1)
        for task in TensorCompositionLearningTask.allCases {
            let text=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            for phrase in ["**Build**","**Configure**","**Check**","Select","automatically"] {XCTAssertTrue(text.contains(phrase),task.rawValue+" "+phrase)}
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertTrue(Set(guide.terms.map {$0.name.lowercased()}).isSuperset(of:["array","axis","observation","gradient"]))
            XCTAssertTrue(item.matches(section:"Library",search:task.title))
            XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"leftGradient"))
            XCTAssertEqual(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"rightGradient") != nil,task.isBinary)
            XCTAssertNotNil(LearnCatalogContent.componentExampleEntries.first {$0.componentID == task.componentID})
        }
    }
    func testConfigureAndGuidedControlsDoNotRepeatDefinitionsOrWiring()throws {
        for task in TensorCompositionLearningTask.allCases {
            let text=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"")
            let configure=try XCTUnwrap(text.components(separatedBy:"**Configure**\n").last?.components(separatedBy:"\n\n**Check**").first)
            let step=LessonAudienceGuide.make(operation:task.rawValue).steps[1]
            for controls in [configure,step] {
                XCTAssertLessThan(controls.count,600,task.rawValue)
                XCTAssertFalse(controls.contains("An array is"),task.rawValue)
                XCTAssertFalse(controls.contains("Add the left and right Inputs"),task.rawValue)
                XCTAssertTrue(controls.contains("Select"),task.rawValue)
            }
            XCTAssertTrue(configure.contains(task == .concat ? "Join direction" : task == .add ? "Second source multiplier" : task == .stack ? "New axis position" : "Direction order"))
        }
    }
    func testValidEmptyCanvasExplainsMissingConstructionRatherThanMalformedBounds() {
        for task in TensorCompositionLearningTask.allCases {
            let result=TensorLearningCanvas.inspect(GraphDocument(),compositionTask:task)
            let message=result.issues.map(\.message).joined(separator:" ")
            XCTAssertTrue(message.contains("Input"),task.rawValue+" "+message)
            XCTAssertTrue(message.contains("Output"),task.rawValue+" "+message)
            XCTAssertTrue(message.contains(BundledTensorCompositionComponents.definition(task.componentID).title),task.rawValue+" "+message)
            XCTAssertFalse(message.contains("bounded graph"),task.rawValue+" "+message)
            var malformed=GraphDocument();malformed.schemaVersion=1
            XCTAssertTrue(TensorLearningCanvas.inspect(malformed,compositionTask:task).issues.contains {$0.message.contains("bounded graph")},"The friendlier missing-nodes message must not weaken schema validation")
        }
    }
    @MainActor func testLateVerifiedCapabilityPublishesLessonReadinessWithoutCanvasEdits()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        XCTAssertTrue(lesson.start(operation:TensorCompositionLearningTask.stack.rawValue))
        XCTAssertNotNil(lesson.runtimeReadinessError)
        let initialGraph=lesson.canvas?.graph
        var notifications=0
        let subscription=lesson.objectWillChange.sink {_ in notifications+=1}
        defer {subscription.cancel()}
        let entry=try XCTUnwrap(BundledTensorCompositionComponents.entries["stack"])
        workspace.tensorCompositionRuntimeCapabilities = .init(descriptors:["stack":try JSONEncoder().encode(entry.runtimeContract)])
        XCTAssertGreaterThan(notifications,0,"Views observing the lesson must refresh when discovery completes")
        XCTAssertNil(lesson.runtimeReadinessError)
        XCTAssertEqual(lesson.canvas?.graph,initialGraph)
    }
}
