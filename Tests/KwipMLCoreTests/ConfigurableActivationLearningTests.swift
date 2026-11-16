import XCTest
@testable import KwipMLCore

final class ConfigurableActivationLearningTests:XCTestCase {
    func testStableSignedValuesZeroInfluenceAndShapes()throws {
        let x=LearningTensor(shape:[1,2,3],values:[-1000,-1,0,1,2,1000])
        for task in ConfigurableActivationLearningTask.allCases {
            let ref=try ConfigurableActivationLessonReference(input:x,task:task)
            XCTAssertEqual(ref.output.shape,x.shape);XCTAssertTrue(ref.output.wellFormed)
            let dx=try ref.inputGradient(for:.init(shape:x.shape,values:Array(repeating:1,count:6)))
            XCTAssertEqual(dx.shape,x.shape);XCTAssertTrue(dx.wellFormed)
            switch task {
            case .silu:XCTAssertEqual(ref.output.values[0],0,accuracy:1e-12);XCTAssertEqual(ref.output.values[5],1000,accuracy:1e-12);XCTAssertEqual(dx.values[2],0.5)
            case .leakyReLU:XCTAssertEqual(ref.output.values[0],-200);XCTAssertEqual(dx.values[2],0.2)
            case .elu:XCTAssertEqual(ref.output.values[0],-2);XCTAssertEqual(dx.values[2],2)
            }
        }
    }
    func testVersionedIndependentPlansCanonicalSubmissionAndWrongIdentity()throws {
        for task in ConfigurableActivationLearningTask.allCases {
            let plan=try ConfigurableActivationAssessment.plan(task:task,runtimeVersion:"2.8",seed:42)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue);XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertEqual(Set(plan.fixtures[0].expected.keys),["output","inputGradient"])
            let graph=GraphDocument(name:"Signed fields",nodes:[.init(id:"x",type:"input",name:"Fields",x:0,y:0),.init(id:"a",type:task.componentID,name:"My transform",x:200,y:0,config:task.configuration),.init(id:"y",type:"output",name:"Results",x:400,y:0)],edges:[.init(id:"one",source:"x",target:"a"),.init(id:"two",source:"a",target:"y")])
            let encoded=try ConfigurableActivationLearningSubmission(graph:graph,task:task).encoded()
            XCTAssertEqual(try ConfigurableActivationLearningSubmission.decode(encoded,task:task).encoded(),encoded)
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            var wrong=graph;wrong.nodes[1].type="relu";wrong.nodes[1].config=[:]
            XCTAssertThrowsError(try ConfigurableActivationLearningSubmission(graph:wrong,task:task))
        }
    }
}
