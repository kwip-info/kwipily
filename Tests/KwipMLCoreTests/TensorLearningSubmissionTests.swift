import XCTest
@testable import KwipMLCore

final class TensorLearningSubmissionTests:XCTestCase {
    func testSavedSubmissionRevalidatesAndRejectsUnknownFields()throws {
        let graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"o")])
        let original=try TensorLearningSubmission(graph:graph)
        let bytes=try original.encoded()
        XCTAssertEqual(try TensorLearningSubmission.decode(bytes),original)
        var root=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        root["expected"]=["output":[1,2]]
        XCTAssertThrowsError(try TensorLearningSubmission.decode(JSONSerialization.data(withJSONObject:root)))
        root.removeValue(forKey:"expected");root["schemaVersion"]=99
        XCTAssertThrowsError(try TensorLearningSubmission.decode(JSONSerialization.data(withJSONObject:root)))
        root["schemaVersion"]=1
        var savedGraph=try XCTUnwrap(root["graph"] as? [String:Any])
        savedGraph["execution"]=["epochs":100];root["graph"]=savedGraph
        XCTAssertThrowsError(try TensorLearningSubmission.decode(JSONSerialization.data(withJSONObject:root)))
        savedGraph.removeValue(forKey:"execution")
        var nodes=try XCTUnwrap(savedGraph["nodes"] as? [[String:Any]])
        nodes[0]["code"]="print('injected')";savedGraph["nodes"]=nodes;root["graph"]=savedGraph
        XCTAssertThrowsError(try TensorLearningSubmission.decode(JSONSerialization.data(withJSONObject:root)))
        XCTAssertThrowsError(try TensorLearningSubmission.decode(Data(repeating:32,count:2_000_001)))
    }
    func testValidatedCanonicalGraphAndInvalidBoundaries()throws {
        var graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"r",type:"reshape",config:["shape":.array([3.0,2.0])]),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"r"),.init(source:"r",target:"o")])
        let submission=try TensorLearningSubmission(graph:graph)
        XCTAssertEqual(submission.graph.nodes[1].config["shape"],.array([3,2]))
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:submission.encoded()) as? [String:Any])
        XCTAssertEqual(Set(object.keys),["schemaVersion","kind","graph"])
        XCTAssertEqual(object["kind"] as? String,"tensorGraph")
        graph.nodes[0].config=["features":2]
        XCTAssertThrowsError(try TensorLearningSubmission(graph:graph))
        graph.nodes[0].config=[:];graph.edges[0].source="r"
        XCTAssertThrowsError(try TensorLearningSubmission(graph:graph))
    }
}
