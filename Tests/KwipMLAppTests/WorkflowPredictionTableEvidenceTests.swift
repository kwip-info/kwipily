import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPredictionTableEvidenceTests:XCTestCase {
    func testWholePopulationAndExactPinRequired() throws {
        let text=#"{"schemaVersion":1,"kind":"table","columns":[{"name":"observation_id","dtype":"string","nullable":false},{"name":"partition","dtype":"string","nullable":false},{"name":"target","dtype":"string","nullable":false},{"name":"output","dtype":"string","nullable":false}],"rows":[["row-a","validation","0","[1,2]"],["row-b","validation","1","[3,4]"]],"lineage":{"role":"validation","observationIDs":["row-a","row-b"]}}"#
        let value=try JSONDecoder().decode(JSONValue.self,from:Data(text.utf8))
        let bytes=try JSONEncoder().encode(value),file=WorkflowArchiveFile(path:"prediction-table.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
        XCTAssertEqual(try WorkflowPredictionTableEvidence.verify(bytes,file:file,partition:"validation",observationIDs:["row-a","row-b"]).rowCount,2)
        XCTAssertThrowsError(try WorkflowPredictionTableEvidence.verify(bytes,file:file,partition:"validation",observationIDs:["row-a"]))
        XCTAssertThrowsError(try WorkflowPredictionTableEvidence.verify(bytes,file:file,partition:"test",observationIDs:["row-a","row-b"]))
        XCTAssertThrowsError(try WorkflowPredictionTableEvidence.verify(bytes,file:.init(path:file.path,bytes:file.bytes,sha256:String(repeating:"0",count:64)),partition:"validation",observationIDs:["row-a","row-b"]))
    }
    func testClassificationCellsMustMatchTheIndependentlyVerifiedEvaluation()throws {
        let names=["observation_id","partition","target","predicted","probability_0","probability_1","output"]
        let object:[String:Any]=["columns":names.map {["name":$0]},"rows":[["row-a","test",1,0,0.5,0.5,"[0,0]"]]]
        let bytes=try JSONSerialization.data(withJSONObject:object)
        let expected=[WorkflowEvaluationPrediction(observationID:"row-a",logits:[0,0],predicted:0,target:1)]
        XCTAssertNoThrow(try WorkflowPredictionTableEvidence.verifyClassification(bytes,rows:expected))
        var changed=object;changed["rows"]=[["row-a","test",1,0,0.6,0.4,"[0,0]"]]
        XCTAssertThrowsError(try WorkflowPredictionTableEvidence.verifyClassification(JSONSerialization.data(withJSONObject:changed),rows:expected))
        changed["rows"]=[["row-a","test",1,0,0.5,0.5,"[1,0]"]]
        XCTAssertThrowsError(try WorkflowPredictionTableEvidence.verifyClassification(JSONSerialization.data(withJSONObject:changed),rows:expected))
    }

}
