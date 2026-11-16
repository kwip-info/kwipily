import XCTest
@testable import KwipMLCore

final class WorkflowTrainMappingTests:XCTestCase {
    private var root:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testMigrationPreservesPreparedPlanAndRewiresTables()throws {
        let source=try WorkflowModelBoundary.consolidate(WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedParallelClassifier(device:"cpu")))
        let registry=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let before=try WorkflowCompiler.compileHistoricalEvidence(source,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:15)
        let next=try WorkflowTrainMapping.migrate(source)
        XCTAssertEqual(next.schemaVersion,18)
        XCTAssertFalse(next.nodes.contains {if case .trainingData=$0.payload {return true};return false})
        let train=try XCTUnwrap(next.nodes.first {$0.address==WorkflowExamples.trainAddress})
        let ports=try WorkflowPortContracts.resolveForAuthoring(train,document:next,modelRegistry:registry,dataRegistry:data)
        XCTAssertEqual(ports.inputs.map(\.id),["model","training","validation"])
        XCTAssertEqual(ports.inputs.map(\.kind),[.modelDefinition,.table,.table])
        let after=try WorkflowCompiler.compile(next,train:train.address,modelRegistry:registry,dataRegistry:data,contractVersion:15)
        XCTAssertEqual(before.featureColumns,after.featureColumns);XCTAssertEqual(before.targetColumn,after.targetColumn)
        XCTAssertEqual(before.dataflow,after.dataflow);XCTAssertEqual(before.settings,after.settings)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(next)),next)
        XCTAssertEqual(try WorkflowTrainMapping.migrate(next),next)
    }
    func testOrphanAndUnsupportedConsumerRefuseAtomically()throws {
        var source=try WorkflowModelBoundary.consolidate(WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedParallelClassifier(device:"cpu")))
        source.edges.removeAll {$0.kind == .trainingData}
        let original=source
        XCTAssertThrowsError(try WorkflowTrainMapping.migrate(source));XCTAssertEqual(source,original)
    }
    func testSharedMappingCopiesExactSettingsToEachTrain()throws {
        var source=try WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedParallelClassifier(device:"cpu"))
        let second=WorkflowNodeAddress(domain:.control,localID:"second-train")
        var clone=try XCTUnwrap(source.nodes.first {$0.address==WorkflowExamples.trainAddress});clone.address=second;source.nodes.append(clone)
        let link=try XCTUnwrap(source.edges.first {$0.kind == .trainingData})
        source.edges.append(.init(id:"second-mapping",from:link.from,to:.init(address:second,port:"data"),kind:.trainingData))
        let next=try WorkflowTrainMapping.migrate(source)
        let configs=next.nodes.compactMap {node->[String:JSONValue]? in if case .train(let config)=node.payload {return config};return nil}
        XCTAssertEqual(configs.count,2);XCTAssertEqual(configs[0]["dataMapping"],configs[1]["dataMapping"])
        XCTAssertEqual(next.edges.filter {$0.to.address==second}.map(\.to.port).sorted(),["training","validation"])
        var conflict=source
        conflict.edges.append(.init(id:"conflict",from:.init(address:.init(domain:.data,localID:"fit"),port:"test"),to:.init(address:second,port:"training"),kind:.table))
        XCTAssertThrowsError(try WorkflowTrainMapping.migrate(conflict))
    }
    func testPartitionRoleReversalAndDowngradeRefuse()throws {
        var source=try WorkflowTrainMapping.migrate(WorkflowTrainRunSetup.migrate(WorkflowExamples.preparedParallelClassifier(device:"cpu")))
        let registry=try ComponentRegistry.builtIn(),data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        for i in source.edges.indices where source.edges[i].to.address==WorkflowExamples.trainAddress && ["training","validation"].contains(source.edges[i].to.port) {
            source.edges[i].to.port=source.edges[i].to.port=="training" ? "validation":"training"
        }
        XCTAssertThrowsError(try WorkflowCompiler.compile(source,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:15))
        source.schemaVersion=17;XCTAssertThrowsError(try WorkflowCodec.encode(source))
    }
    func testLegacyGuidedStagesRemainResumableAndNewCourseHasNoMappingNode()throws {
        for version in [3,4] {
            let plan=try WorkflowGuidedBuildPlan.parallelClassifier(version:version)
            let old=try plan.document(completedSteps:plan.steps.count-1)
            XCTAssertEqual(try WorkflowTrainMapping.migrate(old),old)
        }
        let current=try WorkflowGuidedBuildPlan.parallelClassifier()
        XCTAssertEqual(current.version,5)
        XCTAssertFalse(current.steps.contains {$0.address.localID=="mapping"})
    }
}
