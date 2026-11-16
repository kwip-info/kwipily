import XCTest
@testable import KwipMLCore

final class EmbeddingLearningSubmissionTests:XCTestCase {
    func graph()throws->GraphDocument {
        GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"lookup",type:"embedding",config:try EmbeddingLearningTask.gather.configuration.values),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"lookup"),.init(source:"lookup",target:"o")])
    }
    func testRoundTripAndExistingLessonIsolation()throws {
        let graph=try graph(),submission=try EmbeddingLearningSubmission(graph:graph)
        XCTAssertEqual(try EmbeddingLearningSubmission.decode(submission.encoded()),submission)
        XCTAssertThrowsError(try TensorLearningSubmission(graph:graph))
        XCTAssertThrowsError(try LayerNormLearningSubmission(graph:graph,parameters:[:],gradientNodeID:nil))
    }
    func testInvalidPolicyAndAuthoredAnswersReject()throws {
        var graph=try graph();graph.nodes[1].config["freeze"]=true
        XCTAssertThrowsError(try EmbeddingLearningSubmission(graph:graph))
        graph=try self.graph();graph.nodes[1].type="relu";graph.nodes[1].config=[:]
        XCTAssertThrowsError(try EmbeddingLearningSubmission(graph:graph))
        let data=try EmbeddingLearningSubmission(graph:self.graph()).encoded()
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any])
        raw["expected"] = [:] as [String:String]
        XCTAssertThrowsError(try EmbeddingLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw)))
    }
}
