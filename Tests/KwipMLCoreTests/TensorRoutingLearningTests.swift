import XCTest
@testable import KwipMLCore

final class TensorRoutingLearningTests:XCTestCase {
    func graph(_ task:TensorRoutingLearningTask)->GraphDocument {
        var nodes=task.sourceRoles.map {GraphNode(id:$0,type:"input",name:"User source",config:["fixture":.string($0)])}
        nodes += [.init(id:"op",type:task.componentID,name:"User block",config:task.configuration),.init(id:"out",type:"output",name:"Measured")]
        var edges:[GraphEdge]=[]
        func connect(_ a:String,_ port:String,_ b:String,_ input:String) {edges.append(.init(id:"e\(edges.count)",source:a,sourcePort:port,target:b,targetPort:input))}
        if task == .gather {connect("x","out","op","source");connect("indices","out","op","indices");connect("op","out","out","in")}
        else if task == .einsumCombine {connect("left","out","op","a");connect("right","out","op","b");connect("op","out","out","in")}
        else if task == .einsumDiagonal {connect("x","out","op","a");connect("op","out","out","in")}
        else if task == .chunk {connect("x","out","op","in");connect("op","part4","out","in")}
        else {
            nodes += [.init(id:"first",type:"concat",config:["dim":2]),.init(id:"second",type:"concat",config:["dim":2])]
            let order=task == .splitSize ? [2,0,1]:[2,1,0]
            connect("x","out","op","in");connect("op","part\(order[0])","first","a");connect("op","part\(order[1])","first","b");connect("first","out","second","a");connect("op","part\(order[2])","second","b");connect("second","out","out","in")
        }
        return .init(name:"Editable label",nodes:nodes,edges:edges)
    }
    func testSixIndependentPlansAndStrictNamedPortRoundTrips()throws {
        for task in TensorRoutingLearningTask.allCases {
            let a=try TensorRoutingAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41),b=try TensorRoutingAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:173)
            XCTAssertEqual(a.fixtures.count,4);XCTAssertNotEqual(a.fixtures,b.fixtures);XCTAssertEqual(a.criteria.count,task == .einsumCombine ? 3:2)
            XCTAssertEqual(a.fixtures.map {$0.inputs[task.sourceRoles[0]]!.shape[0]},[1,2,3,1])
            XCTAssertFalse(a.criteria.contains {$0.id == "indicesGradient"})
            let s=try TensorRoutingLearningSubmission(graph:graph(task),task:task)
            XCTAssertEqual(try TensorRoutingLearningSubmission.decode(s.encoded(),task:task),s)
            var raw=try JSONDecoder().decode(JSONValue.self,from:s.encoded())
            if case .object(var envelope)=raw {envelope["seed"]=1;raw = .object(envelope)}
            XCTAssertThrowsError(try TensorRoutingLearningSubmission.decode(JSONEncoder().encode(raw),task:task))
        }
    }
    func testKnownSplitChunkAndGatherCoordinatesAndRepeatedInfluence()throws {
        let table=LearningTensor(shape:[1,2,5],values:(0..<10).map(Double.init))
        let size=try TensorRoutingLessonReference(inputs:["x":table],task:.splitSize)
        XCTAssertEqual(size.output.values,[4,0,1,2,3,9,5,6,7,8])
        XCTAssertEqual(try size.gradients(for:.init(shape:[1,2,5],values:(1...10).map(Double.init)))["inputGradient"]?.values,[2,3,4,5,1,7,8,9,10,6])
        let sections=try TensorRoutingLessonReference(inputs:["x":table],task:.splitSections)
        XCTAssertEqual(sections.output.values,[4,1,2,3,0,9,6,7,8,5])
        let chunk=try TensorRoutingLessonReference(inputs:["x":.init(shape:[1,2,13],values:(0..<26).map(Double.init))],task:.chunk)
        XCTAssertEqual(chunk.output.values,[12,25]);let cg=try chunk.gradients(for:.init(shape:[1,2,1],values:[2,-3]))["inputGradient"]!.values
        XCTAssertEqual(cg[12],2);XCTAssertEqual(cg[25],-3);XCTAssertEqual(cg.filter {$0 != 0}.count,2)
        let gather=try TensorRoutingLessonReference(inputs:["x":.init(shape:[1,2,4],values:(1...8).map(Double.init)),"indices":.init(shape:[1,2,3],values:[0,0,3,2,1,2])],task:.gather)
        XCTAssertEqual(gather.output.values,[1,1,4,7,6,7])
        XCTAssertEqual(try gather.gradients(for:.init(shape:[1,2,3],values:[1,2,3,4,5,6]))["inputGradient"]?.values,[3,0,0,3,0,5,10,0])
    }
    func testKnownTableProductBothSourcesAndDiagonalInfluence()throws {
        let multiply=try TensorRoutingLessonReference(inputs:["left":.init(shape:[1,2,3],values:[1,2,3,4,5,6]),"right":.init(shape:[1,3,2],values:[7,8,9,10,11,12])],task:.einsumCombine)
        XCTAssertEqual(multiply.output.values,[58,64,139,154])
        let gradients=try multiply.gradients(for:.init(shape:[1,2,2],values:[1,2,3,4]))
        XCTAssertEqual(gradients["leftGradient"]?.values,[23,29,35,53,67,81]);XCTAssertEqual(gradients["rightGradient"]?.values,[13,18,17,24,21,30])
        let diagonal=try TensorRoutingLessonReference(inputs:["x":.init(shape:[2,3,3],values:(0..<18).map(Double.init))],task:.einsumDiagonal)
        XCTAssertEqual(diagonal.output.values,[0,4,8,9,13,17])
        XCTAssertEqual(try diagonal.gradients(for:.init(shape:[2,3],values:[1,2,3,4,5,6]))["inputGradient"]?.values,[1,0,0,0,2,0,0,0,3,4,0,0,0,5,0,0,0,6])
    }
    func testPartialDraftAndUnavailableActualPartRefusals()throws {
        let empty=GraphDocument(name:"Not yet built",nodes:[],edges:[])
        XCTAssertEqual(try TensorRoutingLearningSubmission.validatedGraph(empty,task:.chunk,complete:false),empty)
        XCTAssertThrowsError(try TensorRoutingLearningSubmission(graph:empty,task:.chunk))
        var bad=graph(.chunk);bad.edges[1].sourcePort="part5"
        XCTAssertThrowsError(try TensorRoutingLearningSubmission(graph:bad,task:.chunk))
        for task in TensorRoutingLearningTask.allCases {
            bad=graph(task);bad.edges.append(bad.edges[0]);XCTAssertThrowsError(try TensorRoutingLearningSubmission(graph:bad,task:task))
            bad=graph(task);bad.nodes.append(.init(id:"unused",type:"reshape",config:["shape":.array([2,5])]))
            XCTAssertThrowsError(try TensorRoutingLearningSubmission(graph:bad,task:task))
        }
    }
    func testReferenceRefusesPrivateFixtureTamperingAndSelectors()throws {
        for selector in [-1.0,4.0,0.5] {XCTAssertThrowsError(try TensorRoutingLessonReference(inputs:["x":.init(shape:[1,2,4],values:Array(repeating:1,count:8)),"indices":.init(shape:[1,2,3],values:[selector,0,1,2,1,2])],task:.gather))}
        XCTAssertThrowsError(try TensorRoutingLessonReference(inputs:["x":.init(shape:[1,2,5],values:Array(repeating:.nan,count:10))],task:.splitSize))
        XCTAssertThrowsError(try TensorRoutingLessonReference(inputs:["x":.init(shape:[1,3,3],values:Array(repeating:1,count:9)),"answer":.init(shape:[1],values:[1])],task:.einsumDiagonal))
    }
}
