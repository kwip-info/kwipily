import XCTest
@testable import KwipMLCore

final class PoolingLearningTests:XCTestCase {
    func testNegativeMaxWindowsCeilingFirstTiesAndSharedInfluence()throws {
        for task in [PoolingLearningTask.maxOne,.maxTwo,.maxThree] {
            let shape=[1]+task.sampleShape
            let input=LearningTensor(shape:shape,values:Array(repeating:-2,count:shape.reduce(1,*)))
            let reference=try PoolingLessonReference(input:input,task:task)
            XCTAssertEqual(reference.output.shape,[1,2]+Array(repeating:3,count:task.dimensions))
            XCTAssertTrue(reference.output.values.allSatisfy {$0 == -2},"Padding must not introduce zero winners")
            XCTAssertEqual(reference.indices?.values.first,0)
            XCTAssertEqual(reference.indices?.values.last,Double(Int(pow(4.0,Double(task.dimensions)))-1))
            let influence=try reference.inputGradient(for:.init(shape:reference.output.shape,values:Array(repeating:1,count:reference.output.values.count)))
            XCTAssertEqual(influence.values.reduce(0,+),Double(reference.output.values.count))
        }
    }
    func testAveragePartialWindowCountsOnlyDeclaredPadding()throws {
        let ref=try PoolingLessonReference(input:.init(shape:[1,2,6],values:Array(repeating:1,count:12)),task:.averageOne)
        XCTAssertEqual(ref.output.shape,[1,2,4])
        XCTAssertEqual(ref.output.values,[2.0/3,1,1,0.5,2.0/3,1,1,0.5])
        XCTAssertNil(ref.indices)
        let gradient=try ref.inputGradient(for:.init(shape:[1,2,4],values:Array(repeating:1,count:8)))
        XCTAssertEqual(gradient.values[5],1.0/3+0.5,accuracy:1e-12)
    }
    func testAllFourVersionedPlansCanonicalGraphAndTypedIndices()throws {
        for task in PoolingLearningTask.allCases {
            let plan=try PoolingAssessment.plan(task:task,runtimeVersion:"test",seed:12)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertEqual(Set(plan.criteria.map(\.id)),task.isAverage ? ["output","inputGradient"]:["output","inputGradient","indices"])
            let graph=GraphDocument(name:"My windows",nodes:[.init(id:"a",type:"input",name:"Fields",x:0,y:0),.init(id:"b",type:task.componentID,name:"Pool",x:100,y:0,config:task.configuration),.init(id:"c",type:"output",name:"Values",x:200,y:0)],edges:[.init(id:"ab",source:"a",target:"b"),.init(id:"bc",source:"b",target:"c")])
            let submission=try PoolingLearningSubmission(graph:graph,task:task)
            XCTAssertEqual(try PoolingLearningSubmission.decode(submission.encoded(),task:task),submission)
            XCTAssertTrue(TensorLearningTask.operations.contains(task.rawValue))
        }
        XCTAssertFalse(TensorMeasuredOutput.indices.isParameter)
        XCTAssertEqual(TensorMeasuredOutput.indices.title,"Winning input positions")
    }
}
