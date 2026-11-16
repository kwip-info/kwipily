import XCTest
@testable import KwipMLCore

final class PartitionClassifierExamplesTests:XCTestCase {
    func testExamplesCompileWithoutConnectingProtectedTestDataToTraining()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        for temporal in [false,true] {
            let document=try temporal ? WorkflowExamples.preparedTemporalSplitClassifier(device:"cpu") : WorkflowExamples.preparedGroupSplitClassifier(device:"cpu")
            let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:13)
            let split=try XCTUnwrap(document.nodes.first {$0.address.localID == "split"})
            guard case .dataOperator(let op,let config)=split.payload else {return XCTFail("Missing splitter")}
            XCTAssertEqual(op,"kwip.data.partition")
            XCTAssertEqual(config["strategy"],.string(temporal ? "time":"entity"))
            if temporal {XCTAssertEqual(config["entityPolicy"],.string("requireDisjoint"))}
            else {XCTAssertEqual(config["groupBy"],.array([.string("subject_id")]))}
            let consumers=document.edges.filter {$0.from.address == split.address && $0.from.port == "test"}
            XCTAssertEqual(consumers.count,2,"Protected Test may be prepared using training-only state and explicitly evaluated")
            for consumer in consumers {
                let target=try XCTUnwrap(document.nodes.first {$0.address == consumer.to.address})
                XCTAssertEqual(consumer.to.port,"test")
                switch target.payload {
                case .evaluateTest:break
                case .dataOperator(let op,let config):
                    XCTAssertEqual(op,"kwip.data.preprocess");XCTAssertEqual(config["mode"],.string("partitions"))
                default:XCTFail("Protected Test cannot enter training or validation mapping")
                }
            }
            XCTAssertEqual(plan.featureColumns,["signal","context"])
            XCTAssertEqual(plan.targetColumn,"label")
            XCTAssertEqual(plan.settings.epochs,3)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        }
    }
    func testPartitionExamplesProvideUnboundExplicitFrozenEvaluationWithoutChangingTraining()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        for temporal in [false,true] {
            let document=try temporal ? WorkflowExamples.preparedTemporalSplitClassifier(device:"mps") : WorkflowExamples.preparedGroupSplitClassifier(device:"cpu")
            XCTAssertEqual(document.schemaVersion,5)
            let frozen=try XCTUnwrap(document.nodes.first {if case .frozenModel=$0.payload{return true};return false})
            let evaluate=try XCTUnwrap(document.nodes.first {if case .evaluateTest=$0.payload{return true};return false})
            XCTAssertEqual(document.nodes.filter {if case .frozenModel=$0.payload{return true};return false}.count,1)
            XCTAssertEqual(document.nodes.filter {if case .evaluateTest=$0.payload{return true};return false}.count,1)
            XCTAssertEqual(frozen.payload,.frozenModel(config:[:]),"Bundled examples cannot contain a user's private checkpoint binding")
            let plan=try WorkflowEvaluationCompiler.compile(document,evaluate:evaluate.address,modelRegistry:.builtIn(),dataRegistry:registry)
            XCTAssertEqual(plan.frozenModel,frozen.address)
            XCTAssertEqual(plan.testEndpoint,.init(node:"split",port:"test"))
            XCTAssertEqual(plan.testLineage.role,"test")
            XCTAssertEqual(plan.membershipValidation,.exactArchivedProtectedPopulation)
            XCTAssertEqual(plan.settings.device,temporal ? "mps":"cpu")
            XCTAssertTrue(document.edges.contains {$0.from.address==frozen.address && $0.from.port=="model" && $0.to.address==evaluate.address && $0.to.port=="model" && $0.kind == .frozenModel})
            XCTAssertFalse(document.edges.contains {$0.from.address==evaluate.address || $0.to.address==frozen.address},"Evaluation is explicit, never a training successor")
            let training=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:13)
            var trainingOnly=document
            let evaluationAddresses:Set<WorkflowNodeAddress>=[frozen.address,evaluate.address]
            trainingOnly.nodes.removeAll {evaluationAddresses.contains($0.address)}
            trainingOnly.edges.removeAll {evaluationAddresses.contains($0.from.address) || evaluationAddresses.contains($0.to.address)}
            XCTAssertEqual(training,try WorkflowCompiler.compile(trainingOnly,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:13))
            XCTAssertEqual(training.training,.init(node:"fit",port:"training"))
            XCTAssertEqual(training.validation,.init(node:"fit",port:"validation"))
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        }
    }

}
