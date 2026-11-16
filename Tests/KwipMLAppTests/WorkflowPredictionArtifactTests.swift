import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPredictionArtifactTests: XCTestCase {
    func testRepinnedWrongIdentityShapeOrSamplingStillFails() throws {
        let runID = UUID().uuidString
        let baseline: [String: Any] = ["schemaVersion":1,"runID":runID,"selection":"firstRowsOfValidationPartition",
            "evaluationCount":1,"predictionInput":["dtype":"int64","sampleShape":[2]],
            "predictionRows":[["input":[1,2],"label":0,"predicted":1]]]
        func decode(_ object: [String: Any]) throws -> PredictionPreview {
            let data = try JSONSerialization.data(withJSONObject:object)
            let metadata: JSONValue = .object(["schemaVersion":1,"file":.string("predictions.json"),
                "bytes":.integer(Int64(data.count)),"sha256":.string(WorkflowPhaseArtifacts.hash(data))])
            return try WorkflowPredictionArtifact.decode(data,metadata:metadata,runID:runID,
                graph:ExampleFlows.embeddingClassifier,evaluationCount:1)
        }
        XCTAssertEqual(try decode(baseline).rows[0].inputs,["1","2"])
        for (field,value) in [("schemaVersion",2 as Any),("runID",UUID().uuidString as Any),
                              ("selection","random" as Any),("evaluationCount",2 as Any),
                              ("predictionInput",["dtype":"float32","sampleShape":[2]] as Any)] {
            var changed=baseline;changed[field]=value
            XCTAssertThrowsError(try decode(changed),field)
        }
    }
    func testArchivedOmissionPreservesIdentityAndShapeValidation() throws {
        let runID = UUID().uuidString
        var object: [String: Any] = ["schemaVersion":2,"runID":runID,"selection":"firstRowsOfValidationPartition",
            "captureStatus":"omittedByPolicy","evaluationCount":12,
            "predictionInput":["dtype":"int64","sampleShape":[2]],"predictionRows":[]]
        func decode() throws -> PredictionPreview {
            let data = try JSONSerialization.data(withJSONObject:object)
            let metadata: JSONValue = .object(["schemaVersion":2,"file":.string("predictions.json"),
                "bytes":.integer(Int64(data.count)),"sha256":.string(WorkflowPhaseArtifacts.hash(data))])
            return try WorkflowPredictionArtifact.decode(data,metadata:metadata,runID:runID,
                graph:ExampleFlows.embeddingClassifier,evaluationCount:12)
        }
        XCTAssertTrue(try decode().rows.isEmpty)
        object["runID"] = UUID().uuidString
        XCTAssertThrowsError(try decode())
        object["runID"] = runID
        object["predictionInput"] = ["dtype":"float32","sampleShape":[2]]
        XCTAssertThrowsError(try decode())
    }

}
