import XCTest
@testable import KwipMLCore

final class ConvolutionLearningTests: XCTestCase {
    func testAllThreeTasksUseIndependentSharedWindowOutputAndInfluence() throws {
        for task in ConvolutionLearningTask.allCases.filter({!$0.isTransposed}) {
            let d=task.dimensions
            let inputShape=[1,2]+Array(repeating:4,count:d), kernelShape=[2,2]+Array(repeating:2,count:d)
            let input=LearningTensor(shape:inputShape,values:Array(repeating:1,count:inputShape.reduce(1,*)))
            let kernel=LearningTensor(shape:kernelShape,values:Array(repeating:1,count:kernelShape.reduce(1,*)))
            let reference=try ConvolutionLessonReference(input:input,kernel:kernel,bias:.init(shape:[2],values:[1,-1]),dimensions:d)
            XCTAssertEqual(reference.output.shape,[1,2]+Array(repeating:3,count:d))
            let spatial=Int(pow(3.0,Double(d))), window=Int(pow(2.0,Double(d)))
            XCTAssertEqual(Array(reference.output.values.prefix(spatial)),Array(repeating:Double(2*window+1),count:spatial))
            let influence=try reference.gradients(for:.init(shape:reference.output.shape,values:Array(repeating:1,count:reference.output.values.count)))
            XCTAssertEqual(influence.bias.values,[Double(spatial),Double(spatial)])
            XCTAssertTrue(influence.kernel.values.allSatisfy {$0 == Double(spatial)})
            XCTAssertEqual(influence.input.values.first,2)
            XCTAssertEqual(influence.input.values.last,2)
        }
    }
    func testTaskIdentityPrivateFixturesCriteriaAndCanonicalSubmission() throws {
        for task in ConvolutionLearningTask.allCases.filter({!$0.isTransposed}) {
            let plan=try ConvolutionAssessment.plan(task:task,runtimeVersion:"test",seed:41)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            XCTAssertEqual(Set(plan.criteria.map(\.id)),["output","inputGradient","kernelGradient","biasGradient"])
            XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertNotEqual(plan.fixtures,try ConvolutionAssessment.plan(task:task,runtimeVersion:"test",seed:42).fixtures)
            let graph=GraphDocument(name:"My shared window",nodes:[.init(id:"a",type:"input",name:"Data",x:0,y:0),.init(id:"b",type:task.componentID,name:"Window",x:100,y:0,config:task.configuration),.init(id:"c",type:"output",name:"Result",x:200,y:0)],edges:[.init(id:"ab",source:"a",target:"b"),.init(id:"bc",source:"b",target:"c")])
            let submission=try ConvolutionLearningSubmission(graph:graph,task:task)
            XCTAssertEqual(try ConvolutionLearningSubmission.decode(submission.encoded(),task:task),submission)
            var invalid=graph;invalid.nodes[1].type="linear"
            XCTAssertThrowsError(try ConvolutionLearningSubmission(graph:invalid,task:task))
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
            let fields=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(plan.publicTask(mode:.assessment))) as? [String:Any])
            XCTAssertNil(fields["fixtures"]);XCTAssertNil(fields["fixtureSeed"])
        }
    }
}

extension ConvolutionLearningTests {
    func testKernelInfluenceIsAVisibleSharedParameterTensorWithoutObservationSlicing() {
        let tensor=LearningTensor(shape:[2,2,2],values:Array(repeating:1,count:8))
        XCTAssertTrue(TensorMeasuredOutput.kernelGradient.isParameter)
        XCTAssertEqual(TensorMeasuredOutput.kernelGradient.displayTensor(tensor,observation:1),tensor)
        let fixture=TensorFormativeFixture(id:"public",input:.init(shape:[1,2,4],values:Array(repeating:0,count:8)),upstream:.init(shape:[1,2,3],values:Array(repeating:1,count:6)),measured:["kernelGradient":tensor],expected:["kernelGradient":tensor])
        XCTAssertTrue(TensorMeasuredOutput.available(in:fixture).contains(.kernelGradient))
    }
}
