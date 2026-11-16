import XCTest
import KwipMLCore
@testable import KwipMLApp
final class TensorOperationDraftTests:XCTestCase {
    func testOrderedAxesRetainSignedValuesAndRefuseDuplicates()throws {
        var draft=try TensorOperationInspectorDraft(type:"permute",values:["dims":.array([0,-1,1])])
        draft.moveOrder(id:draft.order[0].id,offset:1)
        XCTAssertEqual(try draft.configuration()["dims"],.array([0,1,-1]))
        draft.order[1].value="1";XCTAssertThrowsError(try draft.configuration())
    }
    func testSectionsAndGeometryAreStrictWholeNumbers()throws {
        var split=try TensorOperationInspectorDraft(type:"split",values:["dim":-1,"mode":.string("sections"),"split_size":2,"sections":.array([2,3,1])])
        XCTAssertEqual(try split.configuration()["sections"],.array([2,3,1]))
        split.sections[1].value="0";XCTAssertThrowsError(try split.configuration())
        var unfold=try TensorOperationInspectorDraft(type:"unfold",values:[:]);unfold.texts["stride.1"]="1.5"
        XCTAssertThrowsError(try unfold.configuration())
    }
    func testGuidedCustomNotationLocksBatchAndRetainsOperandChoice()throws {
        var draft=try TensorOperationInspectorDraft(type:"einsum",values:["pattern":.string("custom"),"equation":.string("nab,nbc->nac")])
        draft.aLabels="aa";draft.bLabels="";draft.hasB=false;draft.outputLabels="a"
        XCTAssertEqual(try draft.configuration()["equation"],.string("naa->na"))
        draft.outputLabels="n";XCTAssertThrowsError(try draft.configuration())
    }
    func testMethodDraftRoundTripKeepsSectionsAndProducesEquivalentKnownPermutation()throws {
        let registry=try ComponentRegistry.builtIn()
        var axes=try TensorOperationInspectorDraft(type:"transpose",values:["dim0":1,"dim1":-1])
        try axes.select("permute",registry:registry,input:try .init(sample:[2,3,4]))
        XCTAssertEqual(try axes.configuration()["dims"],.array([0,3,2,1]))
        try axes.select("transpose",registry:registry,input:nil)
        XCTAssertEqual(try axes.configuration(),["dim0":1,"dim1":-1])
        var parts=try TensorOperationInspectorDraft(type:"split",values:["mode":.string("sections"),"dim":-1,"sections":.array([2,3])])
        try parts.select("chunk",registry:registry,input:nil);try parts.select("split",registry:registry,input:nil)
        XCTAssertEqual(try parts.configuration()["sections"],.array([2,3]))
    }

    func testNegativeBatchAliasIsPreservedUntilAnExplicitRankEdit()throws {
        var draft=try TensorOperationInspectorDraft(type:"permute",values:["dims":.array([-3,2,1])])
        XCTAssertEqual(try draft.configuration()["dims"],.array([-3,2,1]))
        draft.order.append(.init(value:"3"))
        XCTAssertEqual(try draft.configuration()["dims"],.array([0,2,1,3]))
    }

}
