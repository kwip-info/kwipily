import XCTest
@testable import KwipMLCore

final class TransposedConvolutionLearningTests:XCTestCase {
    func testIndependentScatterAddsOverlapAndExtraExtentWithoutReconstructingInput()throws {
        for d in 1...3 {
            let xshape=[1,2]+Array(repeating:3,count:d),wshape=[2,2]+Array(repeating:3,count:d)
            let input=LearningTensor(shape:xshape,values:Array(repeating:1,count:xshape.reduce(1,*))),kernel=LearningTensor(shape:wshape,values:Array(repeating:1,count:wshape.reduce(1,*)))
            let ref=try TransposedConvolutionLessonReference(input:input,kernel:kernel,bias:.init(shape:[2],values:[1,-1]),dimensions:d)
            XCTAssertEqual(ref.output.shape,[1,2]+Array(repeating:8,count:d))
            XCTAssertEqual(ref.output.values.first,3)
            let spatial=Int(pow(8.0,Double(d)))
            XCTAssertEqual(ref.output.values[spatial-1],1,"Extra extent with no input contribution still contains bias")
            XCTAssertEqual(ref.output.values.last,-1)
            let influence=try ref.gradients(for:.init(shape:ref.output.shape,values:Array(repeating:1,count:ref.output.values.count)))
            XCTAssertEqual(influence.bias.values,[Double(spatial),Double(spatial)])
            XCTAssertTrue(influence.kernel.values.allSatisfy {$0 == pow(3.0,Double(d))})
            XCTAssertTrue(influence.input.values.allSatisfy {$0 == 2*pow(3.0,Double(d))})
        }
    }
    func testDistinctTasksBudgetAndCanonicalSubmission()throws {
        for task in [ConvolutionLearningTask.transposedOne,.transposedTwo,.transposedThree] {
            XCTAssertTrue(task.isTransposed)
            let plan=try ConvolutionAssessment.plan(task:task,runtimeVersion:"test",seed:42)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            XCTAssertEqual(plan.budget.maximumTensorElements,16384)
            XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertEqual(Set(plan.criteria.map(\.id)),["output","inputGradient","kernelGradient","biasGradient"])
            XCTAssertEqual(task.sampleShape,[2]+Array(repeating:3,count:task.dimensions))
            let graph=GraphDocument(name:"My scatter windows",nodes:[.init(id:"a",type:"input",name:"Source",x:0,y:0),.init(id:"b",type:task.componentID,name:"Windows",x:100,y:0,config:task.configuration),.init(id:"c",type:"output",name:"Result",x:200,y:0)],edges:[.init(id:"ab",source:"a",target:"b"),.init(id:"bc",source:"b",target:"c")])
            let submission=try ConvolutionLearningSubmission(graph:graph,task:task)
            XCTAssertEqual(try ConvolutionLearningSubmission.decode(submission.encoded(),task:task),submission)
            var wrong=graph;wrong.nodes[1].type="conv\(task.dimensions)d"
            XCTAssertThrowsError(try ConvolutionLearningSubmission(graph:wrong,task:task))
        }
    }
}
