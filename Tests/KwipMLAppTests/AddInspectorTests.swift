import XCTest
import KwipMLCore
@testable import KwipMLApp
final class AddInspectorTests:XCTestCase {
    func testMultiplierDraftDoesNotLoseDecimalsOrBroadcastPolicy()throws {
        var draft=try TensorOperationInspectorDraft(type:"add",values:["alpha":-0.5,"sample_broadcast":true])
        XCTAssertEqual(try draft.configuration(),["alpha":-0.5,"sample_broadcast":true])
        draft.texts["alpha"]="1e-3";draft.sampleBroadcast=false
        XCTAssertEqual(try draft.configuration(),["alpha":0.001,"sample_broadcast":false])
        for value in ["NaN","inf","1e40","garbage",""] {draft.texts["alpha"]=value;XCTAssertThrowsError(try draft.configuration())}
        draft.texts["alpha"]="0";XCTAssertEqual(try draft.configuration()["alpha"],.number(0))
    }
    func testOmittedOldValuesLoadWithoutAddingNewSourceFields()throws {
        let draft=try TensorOperationInspectorDraft(type:"add",values:[:])
        XCTAssertFalse(draft.sampleBroadcast);XCTAssertEqual(try draft.configuration(),["alpha":1,"sample_broadcast":false])
    }
}
