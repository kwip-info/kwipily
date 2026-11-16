import XCTest
@testable import KwipMLCore

final class TensorLearningDraftTests:XCTestCase {
    func testOptionalViewportPreservesLegacyDraftsAndRejectsMalformedState()throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:"reshape",mode:.guided)
        XCTAssertNil(try TensorLearningDraft.decode(draft.encoded()).canvasView)
        draft.canvasView=CanvasViewState(zoom:1,scrollX:42)
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft)
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:draft.encoded()) as? [String:Any])
        var view=try XCTUnwrap(raw["canvasView"] as? [String:Any]);view["unexpected"]=true;raw["canvasView"]=view
        XCTAssertThrowsError(try TensorLearningDraft.decode(JSONSerialization.data(withJSONObject:raw)))
        draft.canvasView?.scrollX = -1
        draft.canvasView?.overviewVisible=false
        XCTAssertEqual(try TensorLearningDraft.decode(draft.encoded()),draft,"Signed camera offsets and hidden overview are presentation state.")
        draft.canvasView?.scrollX = -1_000_001
        XCTAssertThrowsError(try draft.encoded())
    }
    func testIncompleteDraftPreservesModeNotesAndUncommittedText()throws {
        var draft=TensorLearningDraft(projectID:UUID().uuidString,operation:"transpose",mode:.assessment)
        draft.notes="Track x[0,1,2] before running."
        draft.pendingFields=["t.dim0":"-"]
        draft.graph.nodes=[.init(id:"t",type:"transpose",config:["dim0":1,"dim1":2])]
        let reopened=try TensorLearningDraft.decode(draft.encoded())
        XCTAssertEqual(reopened,draft)
        XCTAssertThrowsError(try reopened.submission())
        draft.graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"t",type:"transpose",config:["dim0":1,"dim1":2]),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"t"),.init(source:"t",target:"o")])
        XCTAssertThrowsError(try draft.submission())
        draft.pendingFields=[:]
        XCTAssertNoThrow(try draft.submission())
        draft.contentVersion=ContractVersion(99)
        XCTAssertThrowsError(try draft.encoded())
    }
    func testUnknownDraftFieldsAndInvalidIdentityAreRejected()throws {
        let draft=TensorLearningDraft(projectID:UUID().uuidString,operation:"reshape",mode:.guided)
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:draft.encoded()) as? [String:Any])
        raw["grade"]="passed"
        XCTAssertThrowsError(try TensorLearningDraft.decode(JSONSerialization.data(withJSONObject:raw)))
        raw.removeValue(forKey:"grade")
        var graph=try XCTUnwrap(raw["graph"] as? [String:Any])
        graph["code"]="unexpected";raw["graph"]=graph
        XCTAssertThrowsError(try TensorLearningDraft.decode(JSONSerialization.data(withJSONObject:raw)))
        XCTAssertThrowsError(try TensorLearningDraft(projectID:"invalid",operation:"reshape",mode:.practice).encoded())
    }
}
