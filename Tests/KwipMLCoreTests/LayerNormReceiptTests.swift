import XCTest
@testable import KwipMLCore

final class LayerNormReceiptTests: XCTestCase {
    func fixture(_ task:LayerNormLearningTask = .affine, mode:LearningMode = .guided) throws -> TensorLearningReceipt {
        let plan=try LayerNormAssessment.plan(task:task,runtimeVersion:"test",seed:17)
        let graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"norm",type:"layer_norm",config:["normalized_shape":.array(task.normalizedShape.map {.integer(Int64($0))}),"eps":.number(task.epsilon),"elementwise_affine":.bool(task.scale != nil),"bias":.bool(task.bias != nil)]),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"norm"),.init(source:"norm",target:"o")])
        var parameters:[String:[String:LearningTensor]]=[:]
        if let scale=task.scale,let bias=task.bias {parameters["norm"]=["weight":scale,"bias":bias]}
        let bytes=try LayerNormLearningSubmission(graph:graph,parameters:parameters,gradientNodeID:task == .affine ? "norm" : nil).encoded()
        let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:learningHash(bytes),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:12)
        return try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
    }
    func testAllTaskReceiptsReplayAndRejectChangedStateOrEvidence() throws {
        for task in LayerNormLearningTask.allCases {
            let receipt=try fixture(task)
            let restored=try TensorLearningReceipt.decode(receipt.encoded())
            XCTAssertEqual(try restored.verifiedReplay(expectedProjectID:receipt.projectID).id,"assessment."+task.rawValue)
            XCTAssertEqual(restored.attempt,receipt.attempt)
            XCTAssertThrowsError(try restored.verifiedReplay(expectedProjectID:UUID().uuidString))
            var changed=receipt;changed.fixtureSeed+=1
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
            changed=receipt;changed.operation="transpose"
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
            changed=receipt;changed.evidence.fixtures[0].tensors["unexpected"] = .init(shape:[1],values:[0])
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
            changed=receipt;changed.evidence.fixtures[0].tensors[task == .affine ? "scaleGradient" : "inputGradient"]!.values[0]+=1
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
            changed=receipt;changed.submissionData=Data("{}".utf8)
            XCTAssertThrowsError(try changed.verifiedReplay(expectedProjectID:receipt.projectID))
        }
    }
    func testModePrivacyAndStalenessForParameterTables() throws {
        let guided=try fixture()
        let view=try XCTUnwrap(guided.formativeProjection(expectedProjectID:guided.projectID,currentSubmission:guided.submissionData,currentMode:.practice))
        XCTAssertEqual(view.fixtures[0].measured["scaleGradient"]?.shape,[3])
        XCTAssertEqual(view.fixtures[0].expected["biasGradient"]?.shape,[3])
        XCTAssertNil(guided.formativeProjection(expectedProjectID:guided.projectID,currentSubmission:guided.submissionData,currentMode:.assessment))
        var source=try LayerNormLearningSubmission.decode(guided.submissionData)
        var params=source.parameters;params["norm"]?["weight"]?.values[0]+=1
        source=try LayerNormLearningSubmission(graph:source.graph,parameters:params,gradientNodeID:source.gradientNodeID)
        XCTAssertNil(guided.formativeProjection(expectedProjectID:guided.projectID,currentSubmission:try source.encoded(),currentMode:.guided))
        let assessment=try fixture(mode:.assessment)
        for mode in [LearningMode.guided,.practice,.assessment] {
            XCTAssertNil(assessment.formativeProjection(expectedProjectID:assessment.projectID,currentSubmission:assessment.submissionData,currentMode:mode))
        }
    }
}
