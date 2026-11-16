import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationWorkerRequestTests:XCTestCase {
    func testBoundedExplicitWorkerContractRoundTripsAndRejectsUnknownFields()throws {
        let request=WorkflowEvaluationWorkerRequest(evaluationID:UUID().uuidString,projectID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),snapshotReceiptSHA256:String(repeating:"b",count:64),settings:.init(device:"cpu",batchSize:7,previewLimit:3))
        let bytes=try request.encoded()
        XCTAssertEqual(try WorkflowEvaluationWorkerRequest.decode(bytes),request)
        XCTAssertEqual(try request.requestSHA256(),WorkflowArchiveCodec.hash(bytes))
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        object["output"]="/private/elsewhere"
        XCTAssertThrowsError(try WorkflowEvaluationWorkerRequest.decode(JSONSerialization.data(withJSONObject:object)))
        object.removeValue(forKey:"output");object["settings"]=["device":"cpu","batchSize":7,"previewLimit":3,"epochs":1]
        XCTAssertThrowsError(try WorkflowEvaluationWorkerRequest.decode(JSONSerialization.data(withJSONObject:object)))
        for value in [WorkflowEvaluationWorkerRequest(evaluationID:"bad",projectID:request.projectID,requestIdentity:request.requestIdentity,snapshotReceiptSHA256:request.snapshotReceiptSHA256,settings:request.settings),WorkflowEvaluationWorkerRequest(evaluationID:request.evaluationID,projectID:request.projectID,requestIdentity:request.requestIdentity,snapshotReceiptSHA256:request.snapshotReceiptSHA256,settings:.init(batchSize:0))] {
            XCTAssertThrowsError(try value.encoded())
        }
    }
    func testRetentionOverrideOnlyDisablesOptionalPreviewAndRejectsNullOrCoercedValues()throws {
        var request=WorkflowEvaluationWorkerRequest(evaluationID:UUID().uuidString,projectID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),snapshotReceiptSHA256:String(repeating:"b",count:64),settings:.init())
        request.retainComponentOutputs=false
        XCTAssertEqual(try WorkflowEvaluationWorkerRequest.decode(request.encoded()),request)
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:request.encoded()) as? [String:Any])
        for value:Any in [NSNull(),0,"false",true] {
            object["retainComponentOutputs"]=value
            XCTAssertThrowsError(try WorkflowEvaluationWorkerRequest.decode(JSONSerialization.data(withJSONObject:object)))
        }
    }

}
