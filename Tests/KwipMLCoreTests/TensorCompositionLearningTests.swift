import XCTest
@testable import KwipMLCore

final class TensorCompositionLearningTests:XCTestCase {
    func graph(_ task:TensorCompositionLearningTask)->GraphDocument {
        var nodes=[GraphNode(id:"left",type:"input",name:"Renamed input",config:["fixture":.string("left")])]
        var edges:[GraphEdge]=[]
        if task.isBinary {nodes.append(.init(id:"right",type:"input",name:"Other source",config:["fixture":.string("right")]));edges=[.init(id:"a",source:"left",target:"op",targetPort:"a"),.init(id:"b",source:"right",target:"op",targetPort:"b")]} else {edges=[.init(id:"a",source:"left",target:"op")]}
        nodes += [.init(id:"op",type:task.componentID,name:"User label",config:task.configuration),.init(id:"output",type:"output",name:"Measured")]
        edges.append(.init(id:"o",source:"op",target:"output"))
        return .init(name:"Different user labels",nodes:nodes,edges:edges)
    }
    func testEveryTaskPlanSeededIndependentFixturesAndRoundTrip()throws {
        for task in TensorCompositionLearningTask.allCases {
            let a=try TensorCompositionAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:27),b=try TensorCompositionAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:28)
            XCTAssertEqual(a.id,"assessment."+task.rawValue);XCTAssertEqual(a.fixtures.count,4);XCTAssertNotEqual(a.fixtures,b.fixtures)
            XCTAssertEqual(a.criteria.count,task.isBinary ? 3:2)
            let submission=try TensorCompositionLearningSubmission(graph:graph(task),task:task)
            XCTAssertEqual(try TensorCompositionLearningSubmission.decode(submission.encoded(),task:task),submission)
            XCTAssertThrowsError(try TensorCompositionLearningSubmission.decode(submission.encoded(),task:task == .permute ? .concat:.permute))
        }
    }
    func testScalarOutputAndWeightedInfluenceKnownCoordinates()throws {
        let left=LearningTensor(shape:[1,2,3],values:[1,2,3,4,5,6]),pattern=LearningTensor(shape:[1,3],values:[2,4,6])
        let add=try TensorCompositionLessonReference(left:left,right:pattern,task:.add)
        XCTAssertEqual(add.output.values,[0,0,0,3,3,3])
        let g=try add.gradients(for:.init(shape:[1,2,3],values:[1,2,3,4,5,6]))
        XCTAssertEqual(g.left.values,[1,2,3,4,5,6]);XCTAssertEqual(g.right?.values,[-2.5,-3.5,-4.5])
        let cat=try TensorCompositionLessonReference(left:left,right:.init(shape:[1,1,3],values:[7,8,9]),task:.concat)
        XCTAssertEqual(cat.output.values,[1,2,3,4,5,6,7,8,9])
        let stack=try TensorCompositionLessonReference(left:left,right:.init(shape:[1,2,3],values:[7,8,9,10,11,12]),task:.stack)
        XCTAssertEqual(stack.output.values,[1,2,3,7,8,9,4,5,6,10,11,12])
    }
    func testPermuteNonInvolutiveCoordinateInverseAndObservations()throws {
        let input=LearningTensor(shape:[2,2,3,4],values:(0..<48).map(Double.init))
        let reference=try TensorCompositionLessonReference(left:input,task:.permute)
        XCTAssertEqual(reference.output.shape,[2,3,4,2]);XCTAssertEqual(Array(reference.output.values.prefix(8)),[0,12,1,13,2,14,3,15])
        let upstream=LearningTensor(shape:reference.output.shape,values:(0..<48).map {Double($0%7-3)})
        let influence=try reference.gradients(for:upstream)
        for n in 0..<2 {for a in 0..<2 {for b in 0..<3 {for c in 0..<4 {
            XCTAssertEqual(influence.left.values[((n*2+a)*3+b)*4+c],upstream.values[((n*3+b)*4+c)*2+a])
        }}}}
        XCTAssertNil(influence.right)
    }
    func testMalformedSubmissionsCannotEarnChecks()throws {
        for task in TensorCompositionLearningTask.allCases {
            var g=graph(task);g.nodes[0].config=["fixture":.string("answer")]
            XCTAssertThrowsError(try TensorCompositionLearningSubmission(graph:g,task:task))
            g=graph(task);g.edges.append(g.edges[0]);XCTAssertThrowsError(try TensorCompositionLearningSubmission(graph:g,task:task))
            g=graph(task);g.nodes.append(.init(id:"unused",type:"reshape",config:["shape":.array([2,3])]))
            XCTAssertThrowsError(try TensorCompositionLearningSubmission(graph:g,task:task))
            var raw=try JSONDecoder().decode(JSONValue.self,from:TensorCompositionLearningSubmission(graph:graph(task),task:task).encoded())
            if case .object(var envelope)=raw {envelope["expected"] = .object([:]);raw = .object(envelope)}
            XCTAssertThrowsError(try TensorCompositionLearningSubmission.decode(JSONEncoder().encode(raw),task:task))
        }
    }
    func testReferenceRefusesWrongShapeNonfiniteAndExtraOperand()throws {
        XCTAssertThrowsError(try TensorCompositionLessonReference(left:.init(shape:[1,2,3],values:[1,2,3,4,5,6]),task:.permute))
        XCTAssertThrowsError(try TensorCompositionLessonReference(left:.init(shape:[1,2,3],values:[.nan,2,3,4,5,6]),right:.init(shape:[1,3],values:[1,2,3]),task:.add))
        let input=LearningTensor(shape:[1,2,3,4],values:Array(repeating:1,count:24))
        XCTAssertThrowsError(try TensorCompositionLessonReference(left:input,right:input,task:.permute))
    }
}
