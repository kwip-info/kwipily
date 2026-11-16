import XCTest
@testable import KwipMLCore

final class MatrixSubmissionTests:XCTestCase {
    static func graph()->GraphDocument {
        GraphDocument(nodes:[.init(id:"a",type:"input",name:"User left",config:["fixture":.string("left")]),.init(id:"b",type:"input",name:"User right",config:["fixture":.string("right")]),.init(id:"m",type:"matmul"),.init(id:"o",type:"output")],edges:[.init(source:"a",target:"m",targetPort:"a"),.init(source:"b",target:"m",targetPort:"b"),.init(source:"m",target:"o")])
    }
    func testRoundTripAndOperandOrderIsPreserved()throws {
        var graph=Self.graph()
        let submission=try MatrixLearningSubmission(graph:graph)
        XCTAssertEqual(try MatrixLearningSubmission.decode(submission.encoded()),submission)
        graph.edges[0].targetPort="b";graph.edges[1].targetPort="a"
        XCTAssertEqual(try MatrixLearningSubmission(graph:graph).graph,graph)
    }
    func testRejectsDuplicateBindingIncompleteCycleAndAuthoredAnswers()throws {
        for change in 0..<4 {
            var graph=Self.graph()
            if change == 0 {graph.nodes[1].config["fixture"] = .string("left")}
            if change == 1 {graph.edges.removeLast()}
            if change == 2 {graph.edges[0].source="m"}
            if change == 3 {graph.nodes[2].type="custom_code"}
            XCTAssertThrowsError(try MatrixLearningSubmission(graph:graph))
        }
        var raw=try XCTUnwrap(JSONSerialization.jsonObject(with:MatrixLearningSubmission(graph:Self.graph()).encoded()) as? [String:Any])
        raw["expected"] = [:] as [String:String]
        XCTAssertThrowsError(try MatrixLearningSubmission.decode(JSONSerialization.data(withJSONObject:raw)))
    }
}
