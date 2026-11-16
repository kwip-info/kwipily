import XCTest
@testable import KwipMLCore

final class LayerNormDraftTests: XCTestCase {
    func testIncompleteAndCompleteNormalizationDraftRoundTrip() throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:LayerNormLearningTask.affine.rawValue,mode:.practice)
        draft.graph.nodes=[.init(id:"norm",type:"layer_norm",config:Dictionary(uniqueKeysWithValues:LayerNormComponent.definition.fields.map {($0.key,$0.defaultValue)}))]
        draft.parameters=["norm":["weight":.init(shape:[2],values:[1,2]),"bias":.init(shape:[2],values:[0,1])]]
        draft.gradientNodeID="norm";draft.pendingFields=["norm.eps":"1e-"];draft.notes="Predict how a shared scale gradient accumulates."
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
        XCTAssertThrowsError(try draft.submissionData())
        draft.graph.nodes.insert(.init(id:"input",type:"input"),at:0)
        draft.graph.nodes.append(.init(id:"output",type:"output"))
        draft.graph.edges=[.init(source:"input",target:"norm"),.init(source:"norm",target:"output")]
        XCTAssertThrowsError(try draft.submissionData())
        draft.pendingFields=[:]
        let bytes=try draft.submissionData()
        XCTAssertEqual(try LayerNormLearningSubmission.decode(bytes).parameters,draft.parameters)
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()).submissionData(),bytes)
        XCTAssertThrowsError(try draft.submission(),"Legacy parameter-free API must not discard learned state")
    }
    func testParameterDraftMismatchesSurviveButCannotExecute() throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:LayerNormLearningTask.affine.rawValue,mode:.guided)
        draft.graph=LayerNormLearningSubmissionTests().graph()
        draft.parameters=["norm":["weight":.init(shape:[3],values:[1,1,1]),"bias":.init(shape:[2],values:[0,0])]]
        draft.gradientNodeID="norm"
        XCTAssertNoThrow(try draft.encoded())
        XCTAssertThrowsError(try draft.submissionData())
        draft.parameters?["norm"]?["weight"]?.values[0] = .nan
        XCTAssertThrowsError(try draft.validate())
        let legacy=TensorLearningDraft(projectID:UUID().uuidString,operation:"reshape",mode:.guided)
        XCTAssertNil(try TensorLearningDraft.decode(legacy.encoded()).parameters)
        var wrong=legacy;wrong.parameters=[:]
        XCTAssertThrowsError(try wrong.validate())
    }
}
