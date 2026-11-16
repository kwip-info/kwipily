import XCTest
@testable import KwipMLCore
final class PrebuiltModelTests:XCTestCase {
    func testEveryCatalogDefaultDecodesAndLegacyRemainsReadable()throws {
        XCTAssertEqual(PrebuiltModelConfiguration.algorithms.count,9)
        for id in PrebuiltModelConfiguration.algorithms.keys {let c=try WorkflowEstimatorConfiguration.decode(PrebuiltModelConfiguration.defaults(id));XCTAssertTrue(c.isPrebuilt);XCTAssertEqual(c.algorithm,id)}
        var old=WorkflowEstimatorConfiguration.sklearnDefaults;old["featureColumns"] = .array([.string("a"),.string("b")]);old["targetColumn"] = .string("y")
        XCTAssertFalse(try WorkflowEstimatorConfiguration.decode(old).isPrebuilt)
    }
    func testVariableFeaturesAndTaskRefusals()throws {
        var c=PrebuiltModelConfiguration.defaults("dbscan");c["operation"] = .string("predict");XCTAssertThrowsError(try WorkflowEstimatorConfiguration.decode(c))
        c=PrebuiltModelConfiguration.defaults("kmeans");c["targetColumn"] = .string("y");XCTAssertThrowsError(try WorkflowEstimatorConfiguration.decode(c))
        c=PrebuiltModelConfiguration.defaults("ridge");c["featureColumns"] = .array([.string("x"),.string("x")]);XCTAssertThrowsError(try WorkflowEstimatorConfiguration.decode(c))
    }
    func testPracticeAndModelPorts()throws {
        let d=try PrebuiltModelsPractice.document();XCTAssertEqual(d.nodes.count,6)
        let node=try XCTUnwrap(d.nodes.first{$0.address==PrebuiltModelsPractice.estimator})
        let ports=try WorkflowPortContracts.resolve(node,document:d,modelRegistry:ComponentRegistry.builtIn(),dataRegistry:DataFlowOperatorRegistry.load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertTrue(ports.outputs.contains{$0.id=="profiles"})
    }
    func testDerivedTrainingCodeKeepsThreeFeatureClusteringContract()throws {
        let d=try PrebuiltModelsPractice.document(),root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let plan=try WorkflowCodePlan.compile(d,scope:.training,selectedTargets:[PrebuiltModelsPractice.estimator],modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(plan.trainings.first?.configuration?["task"],.string("prebuilt"))
        XCTAssertEqual(plan.trainings.first?.prepared.featureColumns,["usage","recency","breadth"])
        XCTAssertEqual(plan.trainings.first?.prepared.sampleShape,[3])
        try plan.validate()
        if let path=ProcessInfo.processInfo.environment["KWIPML_PREBUILT_CODE_FIXTURE"] {try plan.encoded().write(to:URL(fileURLWithPath:path))}
    }

    func testFitOnlySwitchRefusesConnectedModelAndPreservesSource()throws {
        let d=try PrebuiltModelsPractice.document(),root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        var config=PrebuiltModelConfiguration.defaults("dbscan");config["featureColumns"] = .array([.string("usage"),.string("recency"),.string("breadth")])
        XCTAssertThrowsError(try WorkflowEditing.apply(.configure(address:PrebuiltModelsPractice.estimator,config:config),to:d,selection:.init(),modelRegistry:.builtIn(),dataRegistry:registry))
        XCTAssertEqual(d.edges.count,5)
    }

}
