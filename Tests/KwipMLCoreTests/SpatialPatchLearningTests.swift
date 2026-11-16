import XCTest
@testable import KwipMLCore
final class SpatialPatchLearningTests:XCTestCase {
    func graph(_ task:SpatialPatchLearningTask)->GraphDocument {.init(name:"User patches",nodes:[.init(id:"x",type:"input"),.init(id:"op",type:task.componentID,config:task.configuration),.init(id:"out",type:"output")],edges:[.init(id:"a",source:"x",target:"op"),.init(id:"b",source:"op",target:"out")])}
    func testFourIndependentSeededPlansAndCanonicalSubmissions()throws {
        for task in SpatialPatchLearningTask.allCases {
            let a=try SpatialPatchAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41),b=try SpatialPatchAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:173)
            XCTAssertNotEqual(a.fixtures,b.fixtures);XCTAssertEqual(a.fixtures.map {$0.inputs["x"]!.shape[0]},[1,2,3,1]);XCTAssertEqual(a.criteria.map(\.id),["output","inputGradient"])
            let s=try SpatialPatchLearningSubmission(graph:graph(task),task:task)
            XCTAssertEqual(try SpatialPatchLearningSubmission.decode(s.encoded(),task:task),s)
            XCTAssertTrue(a.fixtures.allSatisfy {Set($0.inputs.keys) == ["x","upstream"] && Set($0.expected.keys) == ["output","inputGradient"]})
        }
    }
    func testUnfoldPackingOrderOverlapAndUnvisitedRowsHaveIndependentInfluence()throws {
        let input=LearningTensor(shape:[1,2,3,4],values:(0..<24).map(Double.init))
        let reference=try SpatialPatchLessonReference(input:input,task:.unfoldOverlap)
        XCTAssertEqual(Array(reference.output.values.prefix(12)),[0,1,2,4,5,6,1,2,3,5,6,7])
        let gradient=try reference.inputGradient(for:.init(shape:[1,8,6],values:Array(repeating:1,count:48)))
        XCTAssertEqual(gradient.values,[1,2,2,1,2,4,4,2,1,2,2,1,1,2,2,1,2,4,4,2,1,2,2,1])
        let spaced=try SpatialPatchLessonReference(input:input,task:.unfoldSpaced)
        XCTAssertEqual(spaced.output.shape,reference.output.shape);XCTAssertNotEqual(spaced.output.values,reference.output.values)
        let g=try spaced.inputGradient(for:.init(shape:spaced.output.shape,values:Array(repeating:1,count:48)))
        for channel in 0..<2 {for row in [0,2] {for col in 0..<4 {XCTAssertEqual(g.values[(channel*3+row)*4+col],0)}}}
        XCTAssertTrue(g.values.contains {$0>1})
    }
    func testFoldAddsOverlapIgnoresPaddingAndLeavesHolesZero()throws {
        let ones=LearningTensor(shape:[1,8,20],values:Array(repeating:1,count:160)),fold=try SpatialPatchLessonReference(input:ones,task:.foldOverlap)
        XCTAssertEqual(fold.output.values,Array(repeating:4,count:24),"Fold adds contributions; it does not average them")
        let influence=try fold.inputGradient(for:.init(shape:[1,2,3,4],values:Array(repeating:1,count:24)))
        XCTAssertTrue(influence.values.contains(0));XCTAssertTrue(influence.values.contains(1));XCTAssertEqual(influence.values.reduce(0,+),96)
        let holes=try SpatialPatchLessonReference(input:.init(shape:[1,8,4],values:Array(repeating:1,count:32)),task:.foldHoles)
        for channel in 0..<2 {for row in 0..<5 {for col in 0..<5 {XCTAssertEqual(holes.output.values[(channel*5+row)*5+col],row == 2 || col == 2 ? 0:1)}}}
        XCTAssertEqual(try holes.inputGradient(for:.init(shape:[1,2,5,5],values:Array(repeating:1,count:50))).values,Array(repeating:1,count:32))
    }
    func testMalformedStateUnknownFieldsColumnMismatchAndPartialDraftBoundaries()throws {
        for task in SpatialPatchLearningTask.allCases {
            XCTAssertNoThrow(try SpatialPatchLearningSubmission.validatedGraph(GraphDocument(),task:task,complete:false));XCTAssertThrowsError(try SpatialPatchLearningSubmission(graph:GraphDocument(),task:task))
            var g=graph(task);g.edges.append(g.edges[0]);XCTAssertThrowsError(try SpatialPatchLearningSubmission(graph:g,task:task))
            g=graph(task);g.nodes[1].config["normalize"]=true;XCTAssertThrowsError(try SpatialPatchLearningSubmission(graph:g,task:task))
            var raw=try JSONDecoder().decode(JSONValue.self,from:SpatialPatchLearningSubmission(graph:graph(task),task:task).encoded())
            if case .object(var values)=raw {values["parameters"] = .object([:]);raw = .object(values)}
            XCTAssertThrowsError(try SpatialPatchLearningSubmission.decode(JSONEncoder().encode(raw),task:task))
        }
        var wrong=graph(.foldOverlap);wrong.nodes[1].config["output_size"] = .array([5,5]);XCTAssertThrowsError(try SpatialPatchLearningSubmission(graph:wrong,task:.foldOverlap))
        XCTAssertThrowsError(try SpatialPatchLessonReference(input:.init(shape:[1,8,20],values:Array(repeating:.nan,count:160)),task:.foldOverlap))
    }
}
