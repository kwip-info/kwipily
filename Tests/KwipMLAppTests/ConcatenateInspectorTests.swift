import XCTest
import KwipMLCore
@testable import KwipMLApp
final class ConcatenateInspectorTests:XCTestCase {
    func testSignedDraftRoundTripAndInvalidAxes()throws {
        var draft=try TensorOperationInspectorDraft(type:"concat",values:[:])
        XCTAssertEqual(try draft.configuration(),["dim":-1])
        draft.texts["dim"]="2";XCTAssertEqual(try draft.configuration(),["dim":2])
        for text in ["0","1.5","8","bad"] {draft.texts["dim"]=text;XCTAssertThrowsError(try draft.configuration())}
        draft.texts["dim"]="-2";XCTAssertEqual(try draft.configuration(),["dim":-2])
    }
    func testConnectedLabelsUseNamesAndQualifiedIdentity()throws {
        let registry=try ComponentRegistry.builtIn()
        let source=WorkflowNodeAddress(domain:.model,localID:"same"),wrong=WorkflowNodeAddress(domain:.data,localID:"same"),join=WorkflowNodeAddress(domain:.model,localID:"join")
        let doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:source,name:"Image features",payload:.modelPrimitive(type:"linear",configurationVersion:2,config:[:])),.init(address:wrong,name:"Wrong table",payload:.dataOperator(operatorID:"kwip.data.tableInput",config:[:])),.init(address:join,payload:.modelPrimitive(type:"concat",configurationVersion:2,config:[:]))],edges:[.init(id:"wire",from:.init(address:source,port:"out"),to:.init(address:join,port:"a"),kind:.tensor)])
        XCTAssertEqual(TensorOperationInputLabels.workflow(doc,address:join,registry:registry),["a":"Image features · out"])
    }
}
