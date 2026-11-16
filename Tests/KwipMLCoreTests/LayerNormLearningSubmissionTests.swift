import XCTest
@testable import KwipMLCore

final class LayerNormLearningSubmissionTests: XCTestCase {
    func graph() -> GraphDocument {
        let config=Dictionary(uniqueKeysWithValues:LayerNormComponent.definition.fields.map {($0.key,$0.defaultValue)})
        return GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"norm",type:"layer_norm",config:config),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"norm"),.init(source:"norm",target:"o")])
    }
    func testExplicitParameterAndTargetRoundTrip() throws {
        let parameters=["norm":["weight":LearningTensor(shape:[2],values:[0.5,2]),"bias":LearningTensor(shape:[2],values:[1,-1])]]
        let submission=try LayerNormLearningSubmission(graph:graph(),parameters:parameters,gradientNodeID:"norm")
        XCTAssertEqual(try LayerNormLearningSubmission.decode(submission.encoded()),submission)
        var noAffine=graph();noAffine.nodes[1].config["elementwise_affine"]=false
        let simple=try LayerNormLearningSubmission(graph:noAffine,parameters:[:],gradientNodeID:nil)
        let raw=try JSONDecoder().decode(JSONValue.self,from:simple.encoded())
        guard case .object(let fields)=raw else {return XCTFail("Expected object")}
        XCTAssertEqual(fields["gradientNodeID"],.null)
        XCTAssertEqual(try LayerNormLearningSubmission.decode(simple.encoded()),simple)
        XCTAssertThrowsError(try TensorLearningSubmission(graph:noAffine))
    }
    func testInvalidParametersTargetsAndSavedFieldsRejected() throws {
        let valid=["norm":["weight":LearningTensor(shape:[2],values:[1,1]),"bias":LearningTensor(shape:[2],values:[0,0])]]
        XCTAssertThrowsError(try LayerNormLearningSubmission(graph:graph(),parameters:[:],gradientNodeID:nil))
        XCTAssertThrowsError(try LayerNormLearningSubmission(graph:graph(),parameters:valid,gradientNodeID:"i"))
        var wrong=valid;wrong["norm"]?["weight"] = .init(shape:[1,2],values:[1,1])
        XCTAssertThrowsError(try LayerNormLearningSubmission(graph:graph(),parameters:wrong,gradientNodeID:"norm"))
        var disabled=graph();disabled.nodes[1].config["bias"]=false
        XCTAssertThrowsError(try LayerNormLearningSubmission(graph:disabled,parameters:valid,gradientNodeID:"norm"))
        let bytes=try LayerNormLearningSubmission(graph:graph(),parameters:valid,gradientNodeID:"norm").encoded()
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        raw["expected"] = [:] as [String:String]
        XCTAssertThrowsError(try LayerNormLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw)))
        raw.removeValue(forKey:"expected")
        var params=try XCTUnwrap(raw["parameters"] as? [String:Any]);var node=try XCTUnwrap(params["norm"] as? [String:Any]);var tensor=try XCTUnwrap(node["weight"] as? [String:Any]);tensor["extra"]=1;node["weight"]=tensor;params["norm"]=node;raw["parameters"]=params
        XCTAssertThrowsError(try LayerNormLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw)))
    }
}
