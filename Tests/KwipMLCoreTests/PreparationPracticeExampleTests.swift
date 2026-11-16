import XCTest
@testable import KwipMLCore

final class PreparationPracticeExampleTests: XCTestCase {
    func testConfiguredPracticeCompilesThreeFeaturesAndSharedTrainingOnlyStates() throws {
        let example = PreparedWorkflowExample.preparationPractice
        XCTAssertEqual(example.observationsResource, "preparation-practice-observations")
        let document = try example.document(device: "cpu")
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = try ComponentRegistry.builtIn()
        let plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
            modelRegistry: registry, dataRegistry: .load(from: repository.appendingPathComponent("runner/dataflow-operators.json")))
        XCTAssertEqual(plan.featureColumns, ["amount", "onehot:[\"region\",\"north\"]", "onehot:[\"region\",\"south\"]"])
        XCTAssertEqual(try WorkflowMappingRequirements.inputContract(plan.model.graph, registry: registry).layout.sample.dimensions, [3])
        XCTAssertEqual(plan.sourceAddresses, [WorkflowExamples.sourceAddress])
        let prepare=try XCTUnwrap(plan.dataflow.nodes.first {$0.id == "prepare"})
        XCTAssertEqual(prepare.operatorID,"kwip.data.preprocess")
        let config=try PreprocessConfiguration(values:prepare.config)
        XCTAssertEqual(config.mode,"partitions")
        let amount=try XCTUnwrap(config.columns.first {$0.column == "amount"})
        XCTAssertEqual(amount.missing,"mean")
        let region=try XCTUnwrap(config.columns.first {$0.column == "region"})
        XCTAssertEqual(region.encoding,"oneHot");XCTAssertEqual(region.unknownPolicy,"ignore")
        XCTAssertEqual(region.vocabularyOrder,"lexicographic");XCTAssertEqual(region.maxCategories,2)
        XCTAssertEqual(Set(plan.dataflow.edges.filter {$0.to.node == "prepare"}.map(\.from)),Set(["training","validation","test"].map {DataFlowEndpoint(node:"split",port:$0)}))
        XCTAssertEqual(plan.training,.init(node:"prepare",port:"training"))
        XCTAssertEqual(plan.validation,.init(node:"prepare",port:"validation"))
        let encoded = try WorkflowCodec.encode(document)
        XCTAssertFalse(String(decoding: encoded, as: UTF8.self).contains("/Users/"))
    }
    func testPublicResourceCarriesMissingValuesAndHeldOutOnlyCategory() throws {
        let source = try String(contentsOf: PreparationPracticeExample.observationsURL(), encoding: .utf8)
        let lines = source.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines.count, 17)
        XCTAssertEqual(lines.first, "observed_at,amount,region,label")
        XCTAssertEqual(lines.filter { $0.contains(",,") }.count, 3)
        XCTAssertTrue(lines[9].contains(",remote,"))
        XCTAssertFalse(lines[1...8].contains { $0.contains(",remote,") })
    }
}
