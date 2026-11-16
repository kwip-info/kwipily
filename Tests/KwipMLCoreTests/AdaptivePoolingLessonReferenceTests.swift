import XCTest
@testable import KwipMLCore

final class AdaptivePoolingLessonReferenceTests:XCTestCase {
    func testAverageWindowsDeclaredPaddingAndCeilingExtension()throws {
        let reference=try AdaptivePoolingLessonReference(input:.init(shape:[1,2,6,6],values:Array(repeating:1,count:72)),task:.averageTwo)
        XCTAssertEqual(reference.output.shape,[1,2,4,3]);XCTAssertNil(reference.indices)
        let channel=[4.0/9,2.0/3,1.0/3,2.0/3,1,0.5,2.0/3,1,0.5,1.0/3,0.5,0.25]
        for (actual,expected) in zip(reference.output.values,channel+channel) {XCTAssertEqual(actual,expected,accuracy:1e-12)}
        let gradient=try reference.inputGradient(for:.init(shape:reference.output.shape,values:Array(repeating:1,count:24)))
        XCTAssertEqual(gradient.values[35],1.0/6+1.0/4,accuracy:1e-12)
    }
    func testAdaptiveUnevenBinsAndMeasuredSharedInfluence()throws {
        let input=LearningTensor(shape:[1,2,5],values:[0,1,2,3,4,10,11,12,13,14])
        let reference=try AdaptivePoolingLessonReference(input:input,task:.adaptiveAverageOne)
        XCTAssertEqual(reference.output.values,[0.5,2,3.5,10.5,12,13.5]);XCTAssertEqual(reference.output.shape,[1,2,3])
        let gradient=try reference.inputGradient(for:.init(shape:reference.output.shape,values:[1,-2,3,1,-2,3]))
        let channel=[0.5,0.5-2.0/3,-2.0/3,1.5-2.0/3,1.5]
        for (actual,expected) in zip(gradient.values,channel+channel) {XCTAssertEqual(actual,expected,accuracy:1e-12)}
    }
    func testAdaptiveMaximumFirstTiesReadingOrderAndRepeatedInfluence()throws {
        let reference=try AdaptivePoolingLessonReference(input:.init(shape:[2,2,5,4],values:Array(repeating:-2,count:80)),task:.adaptiveMaximumTwo)
        XCTAssertEqual(reference.output.shape,[2,2,3,4]);XCTAssertTrue(reference.output.values.allSatisfy {$0 == -2})
        XCTAssertEqual(reference.indices?.values,Array(repeating:[0.0,1,2,3,4,5,6,7,12,13,14,15],count:4).flatMap {$0})
        let gradient=try reference.inputGradient(for:.init(shape:reference.output.shape,values:Array(repeating:1,count:48)))
        XCTAssertEqual(gradient.values.reduce(0,+),48)
    }
    func testKeepWidthAndSeparateBatchesChannels()throws {
        let values=[Double](repeating:1,count:20)+[Double](repeating:3,count:20)+[Double](repeating:-2,count:20)+[Double](repeating:7,count:20)
        let reference=try AdaptivePoolingLessonReference(input:.init(shape:[2,2,5,4],values:values),task:.adaptiveAverageTwo)
        XCTAssertEqual(reference.output.values,Array(repeating:1.0,count:12)+Array(repeating:3.0,count:12)+Array(repeating:-2.0,count:12)+Array(repeating:7.0,count:12))
        XCTAssertNil(reference.indices)
    }
    func testStrictReferenceInputsAndInfluence()throws {
        for task in AdaptivePoolingLearningTask.allCases {
            let shape=[1]+task.sampleShape
            XCTAssertThrowsError(try AdaptivePoolingLessonReference(input:.init(shape:shape,values:[0]),task:task))
            XCTAssertThrowsError(try AdaptivePoolingLessonReference(input:.init(shape:[3]+task.sampleShape,values:Array(repeating:0,count:3*task.sampleShape.reduce(1,*))),task:task))
            let ref=try AdaptivePoolingLessonReference(input:.init(shape:shape,values:Array(repeating:0,count:shape.reduce(1,*))),task:task)
            XCTAssertThrowsError(try ref.inputGradient(for:.init(shape:[1],values:[1])))
            XCTAssertThrowsError(try ref.inputGradient(for:.init(shape:ref.output.shape,values:Array(repeating:.nan,count:ref.output.values.count))))
        }
    }
    func testFourVersionedPlansDistinctObjectivesAndSeededPrivateCases()throws {
        for task in AdaptivePoolingLearningTask.allCases {
            let plan=try AdaptivePoolingAssessment.plan(task:task,runtimeVersion:"test",seed:12)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue);XCTAssertEqual(plan.fixtures.count,4)
            XCTAssertEqual(Set(plan.criteria.map(\.id)),task.isAverage ? ["output","inputGradient"]:["output","inputGradient","indices"])
            XCTAssertEqual(plan.fixtures.map {$0.inputs["x"]!.shape[0]},[1,2,1,2])
            XCTAssertNotEqual(plan.fixtures[1].inputs["x"]?.values,plan.fixtures[3].inputs["x"]?.values)
            XCTAssertEqual(task.configuration["output_width"],task.dimensions == 2 && task != .averageTwo ? .null:nil)
        }
    }
}
