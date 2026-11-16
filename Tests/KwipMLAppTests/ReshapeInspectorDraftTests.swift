import XCTest
import KwipMLCore
@testable import KwipMLApp
final class ReshapeInspectorDraftTests:XCTestCase {
    func testGridInferenceOrderingAndFixedDraftAreRetained()throws {
        var draft=try ReshapeInspectorDraft(variant:.dimensions,values:["shape":.array([2,3,4])])
        draft.rows[1].size="6";draft.rows[1].inferred=true
        XCTAssertEqual(try draft.configuration()["shape"],.array([2,-1,4]))
        draft.rows[1].inferred=false;XCTAssertEqual(draft.rows[1].size,"6")
        draft.moveRow(id:draft.rows[2].id,offset:-1)
        XCTAssertEqual(try draft.configuration()["shape"],.array([2,4,6]))
        draft.rows[0].inferred=true;draft.rows[1].inferred=true
        XCTAssertThrowsError(try draft.configuration())
        draft.rows[1].inferred=false;draft.rows[1].size="0"
        XCTAssertThrowsError(try draft.configuration())
    }
    func testMethodRoundTripKeepsRelativeAxesAndDimensionDraft()throws {
        var draft=try ReshapeInspectorDraft(variant:.dimensions,values:["shape":.array([3,-1])])
        try draft.select(.flatten);draft.startAxis="2";draft.endAxis="-1"
        XCTAssertEqual(try draft.configuration(),["start_dim":2,"end_dim":-1])
        try draft.select(.dimensions);XCTAssertEqual(try draft.configuration()["shape"],.array([3,-1]))
        try draft.select(.flatten);XCTAssertEqual(draft.endAxis,"-1")
        draft.startAxis="0";XCTAssertThrowsError(try draft.configuration())
        XCTAssertThrowsError(try draft.select(.dimensions));XCTAssertEqual(draft.variant,.flatten)
    }
    func testFlattenToDimensionDraftUsesDeclaredShapeAndPreservesSignedAxes()throws {
        var draft=try ReshapeInspectorDraft(variant:.flatten,values:["start_dim":-2,"end_dim":-1])
        try draft.select(.dimensions,input:try .init(sample:[2,3,4]))
        XCTAssertEqual(try draft.configuration()["shape"],.array([2,12]))
        try draft.select(.flatten);XCTAssertEqual(try draft.configuration(),["start_dim":-2,"end_dim":-1])
    }
    func testRestrictedPaletteCannotOfferAnUnavailableMethodAndLessonsRetainIDs()throws {
        let flatten=ComponentRibbonEntry(id:"flatten",title:"Flatten",shortTitle:"Flatten",purpose:"",category:"Tensor layout",typeIcon:"flatten")
        let reshape=ComponentRibbonEntry(id:"reshape",title:"Reshape",shortTitle:"Reshape",purpose:"",category:"Tensor layout",typeIcon:"reshape")
        let entries=ReshapePalette.consolidated(entries:[flatten,reshape])
        XCTAssertEqual(entries.count,1);XCTAssertEqual(entries[0].id,"reshape");XCTAssertEqual(entries[0].typeIcon,"kwip.reshape")
        XCTAssertEqual(ReshapePalette.variants(allowedTypes:["flatten"]),[.flatten])
        XCTAssertEqual(ReshapePalette.consolidated(entries:[flatten])[0].id,"flatten")
        for id in ["reshape","flatten"] {
            XCTAssertNotNil(LearnCatalogContent.componentExampleEntries.first {$0.componentID==id})
            let document=try PreparedWorkflowExample.component(id).document(device:"cpu")
            XCTAssertTrue(document.nodes.contains {if case .modelPrimitive(let type,_,_)=$0.payload {return type==id};return false})
        }
    }

}
