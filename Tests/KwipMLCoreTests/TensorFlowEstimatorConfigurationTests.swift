import XCTest
@testable import KwipMLCore

final class TensorFlowEstimatorConfigurationTests:XCTestCase {
    func testExplicitCPUSettingsAndInvalidTypes()throws {
        let settings=TensorFlowEstimatorConfiguration.defaultSettings
        XCTAssertNoThrow(try TensorFlowEstimatorConfiguration.validate(settings))
        for (key,value) in [("device",JSONValue.string("mps")),("seed",.bool(true)),("epochs",.integer(0)),("batchSize",.integer(100001)),("learningRate",.number(0))] {
            var wrong=settings;wrong[key]=value
            XCTAssertThrowsError(try TensorFlowEstimatorConfiguration.validate(wrong))
        }
        var unknown=settings;unknown["resume"] = .bool(true)
        XCTAssertThrowsError(try TensorFlowEstimatorConfiguration.validate(unknown))
    }
    func testFoundationGraphGate()throws {
        let registry=try ComponentRegistry.builtIn()
        XCTAssertNoThrow(try TensorFlowEstimatorConfiguration.validateGraph(ExampleFlows.parallelClassifier,registry:registry))
        var wrong=ExampleFlows.parallelClassifier
        wrong.nodes[1].type="gelu"
        XCTAssertThrowsError(try TensorFlowEstimatorConfiguration.validateGraph(wrong,registry:registry))
    }
}

extension TensorFlowEstimatorConfigurationTests {
    func testNativeExampleCompilesAndLessonIsVersioned()throws {
        let document=try WorkflowExamples.preparedTensorFlowClassifier()
        let dataRegistry=try DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))
        let plan=try WorkflowEstimatorCompiler.compile(document,estimator:WorkflowExamples.tensorFlowEstimatorAddress,modelRegistry:ComponentRegistry.builtIn(),dataRegistry:dataRegistry)
        XCTAssertEqual(plan.configuration.backendID,"tensorflow")
        XCTAssertNotNil(plan.model)
        XCTAssertEqual(plan.training,.init(node:"fit",port:"training"))
        XCTAssertEqual(plan.validation,.init(node:"fit",port:"validation"))
        let lesson=try TensorFlowFoundationLesson.bundled()
        XCTAssertEqual(lesson.contentVersion,"1.0.0")
        XCTAssertEqual(lesson.checks.count,4)
    }
}
