import XCTest
@testable import KwipMLCore

final class InitializationLearningDispatchTests:XCTestCase {
    func testSharedIdentityAndPlanMatchIndependentInitializationContracts()throws {
        for task in InitializationLearningTask.allCases {
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            XCTAssertEqual(TensorLearningTask.title(for:task.rawValue),task.title)
            XCTAssertEqual(try TensorLearningTask.plan(operation:task.rawValue,runtimeVersion:"2.14.1",seed:71),try InitializationAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:71))
        }
    }
    func testPartialColdDraftAndPendingControlsDoNotBecomeExecutable()throws {
        for task in InitializationLearningTask.allCases {
            var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice)
            draft.graph=try InitializationLearningExamples.defaultGraph(task:task)
            draft.pendingFields=["initialization.seed":"-"];draft.notes="Keep the same random starting point."
            XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
            XCTAssertThrowsError(try draft.submissionData())
            draft.pendingFields=[:]
            XCTAssertThrowsError(try draft.submissionData())
            draft.graph=try InitializationLearningExamples.preparedGraph(task:task)
            XCTAssertEqual(try InitializationLearningSubmission.decode(draft.submissionData(),task:task).graph,draft.graph)
            draft.parameters=["first":["weight":.init(shape:[1],values:[0])]]
            XCTAssertThrowsError(try draft.encoded())
        }
    }
    func testLargeTypedEditorDraftOnlyUsesActualInitializationNodeAndUTF8Budget()throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:InitializationLearningTask.startingValues.rawValue,mode:.guided)
        draft.graph=try InitializationLearningExamples.defaultGraph(task:.startingValues)
        let key="14:initialization:initializationEditor"
        let payload=JSONValue.object(["configuration":.object(InitializationLearningTask.startingValues.configuration),"seedText":.string("-"),"rawNumericDrafts":.object(["first.weight.mean":.string(String(repeating:"-",count:1024))])])
        draft.pendingFields=[key:String(decoding:try JSONEncoder().encode(payload),as:UTF8.self)]
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
        XCTAssertThrowsError(try draft.submissionData())
        var missing=draft;missing.graph.nodes.removeAll {$0.type == "initialization"}
        XCTAssertThrowsError(try missing.encoded())
        var wrongKey=draft;wrongKey.pendingFields=["14:initialization:other":String(repeating:"x",count:1024)]
        XCTAssertThrowsError(try wrongKey.encoded())
        var other=TensorLearningDraft(projectID:UUID().uuidString,operation:"reshape",mode:.guided)
        other.pendingFields=draft.pendingFields
        XCTAssertThrowsError(try other.encoded())
        draft.pendingFields=[key:String(repeating:"x",count:131073)]
        XCTAssertThrowsError(try draft.encoded())
        draft.pendingFields=[key:String(repeating:"é",count:65537)]
        XCTAssertThrowsError(try draft.encoded(),"The expanded editor budget is UTF8 bytes, not character count.")
    }

}
