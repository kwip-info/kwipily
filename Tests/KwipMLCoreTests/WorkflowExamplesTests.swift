import XCTest
import CryptoKit
@testable import KwipMLCore

final class WorkflowExamplesTests: XCTestCase {
    func testTensorLayoutExampleUsesSharedDataAndControlPipeline()throws {
        let document=try WorkflowExamples.preparedTensorLayoutClassifier(device:"cpu")
        let plan=try compile(document)
        XCTAssertEqual(plan.sourceAddresses,[WorkflowExamples.sourceAddress])
        XCTAssertEqual(plan.settings.epochs,3)
        XCTAssertEqual(plan.featureColumns,["signal","context"])
        XCTAssertEqual(Set(plan.model.graph.nodes.map(\.type)),["input","linear","reshape","transpose","flatten","output"])
        XCTAssertTrue(document.definitions.isEmpty)
        XCTAssertTrue(GraphValidator.validate(plan.model.graph,registry:try .builtIn()).isValid)
    }
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func compile(_ document: WorkflowDocument) throws -> WorkflowExecutionPlan {
        try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress, modelRegistry: .builtIn(),
            dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testBundledWorkflowCompilesExplicitTrainingOnlyFitAndPinnedParallelModel() throws {
        let document = try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let plan = try compile(document)
        XCTAssertEqual(plan.sourceAddresses,[WorkflowExamples.sourceAddress])
        XCTAssertEqual(plan.training,.init(node:"fit",port:"training"))
        XCTAssertEqual(plan.validation,.init(node:"fit",port:"validation"))
        XCTAssertEqual(plan.featureColumns,["signal","context"]);XCTAssertEqual(plan.targetColumn,"label")
        XCTAssertEqual(plan.settings.epochs,3);XCTAssertEqual(plan.settings.batchSize,32)
        XCTAssertEqual(plan.settings.seed,42);XCTAssertEqual(plan.settings.learningRate,0.03);XCTAssertEqual(plan.settings.device,"cpu")
        XCTAssertNil(plan.settings.datasetCSV);XCTAssertNil(plan.settings.preparedTraining)
        let fit=try XCTUnwrap(plan.dataflow.nodes.first {$0.operatorID == "kwip.data.preprocess"})
        let preparation=try PreprocessConfiguration(values:fit.config)
        XCTAssertEqual(preparation.mode,"partitions")
        XCTAssertEqual(preparation.columns.map(\.column),["signal","context"])
        XCTAssertTrue(preparation.columns.allSatisfy {$0.scaling == "standard" && $0.missing == "error"})
        XCTAssertEqual(Set(plan.dataflow.edges.filter {$0.to.node == fit.id}.map(\.from)),Set(["training","validation","test"].map {DataFlowEndpoint(node:"split",port:$0)}))
        // Views inspect prepared roles in the authored graph; they are terminal
        // sinks, outside the upstream dependencies required by Train.
        XCTAssertTrue(document.edges.contains {
            $0.from == .init(address:.init(domain:.data,localID:"fit"),port:"validation") &&
            $0.to == .init(address:.init(domain:.data,localID:"validation"),port:"table")
        })
        let inspection:Set<WorkflowNodeAddress>=Set(["training","validation"].map { .init(domain:.data,localID:$0) })
        XCTAssertFalse(document.edges.contains {inspection.contains($0.from.address)})
        XCTAssertTrue(Set(plan.dataflow.nodes.map(\.id)).isDisjoint(with:["training","validation"]))
        let expected = try CompositeExpander.expand(CompositeExamples.parallelClassifier,registry:.builtIn())
        let actualDigest = SHA256.hash(data:try GraphSemantics.canonicalData(expected.graph,registry:.builtIn())).map{String(format:"%02x",$0)}.joined()
        XCTAssertEqual(plan.model.semanticSHA256,actualDigest)
        XCTAssertEqual(document.definitions,CompositeExamples.parallelClassifier.definitions)
        let validation = GraphValidator.validate(plan.model.graph,registry:try .builtIn())
        XCTAssertTrue(validation.isValid)
        XCTAssertEqual(validation.widths["input"],2);XCTAssertEqual(validation.widths["classifier"],2)
        XCTAssertEqual(plan.model.graph.nodes.filter{$0.type == "linear"}.count,3)
        let parameterCount = plan.model.graph.nodes.filter{$0.type == "linear"}.reduce(0) { total,node in
            let input = Int(node.config["in_features"]!.doubleValue!), output = Int(node.config["out_features"]!.doubleValue!)
            return total + input * output + output
        }
        XCTAssertEqual(parameterCount,82)
        XCTAssertNotNil(plan.model.provenance)
    }
    func testFactoryIdentityConfigurationAndPortableSourceHaveNoLocalBinding() throws {
        let projectID = UUID().uuidString
        let document = try WorkflowExamples.preparedParallelClassifier(projectID:projectID,device:"auto")
        XCTAssertEqual(document.projectID,projectID);XCTAssertEqual(document.revision,0)
        XCTAssertEqual(try WorkflowExamples.preparedParallelClassifier(projectID:projectID,device:"auto"),document)
        XCTAssertNotEqual(try WorkflowExamples.preparedParallelClassifier().projectID,try WorkflowExamples.preparedParallelClassifier().projectID)
        XCTAssertEqual(try compile(document).settings.device,"auto")
        XCTAssertEqual(try compile(WorkflowExamples.preparedParallelClassifier()).settings.device,"auto")
        XCTAssertThrowsError(try WorkflowExamples.preparedParallelClassifier(projectID:"invalid"))
        XCTAssertThrowsError(try WorkflowExamples.preparedParallelClassifier(device:"cuda"))
        let bytes = try WorkflowCodec.encode(document)
        XCTAssertEqual(try WorkflowCodec.decode(bytes),document)
        let text = String(decoding:bytes,as:UTF8.self)
        XCTAssertFalse(text.contains("/Users/"));XCTAssertFalse(text.contains("sourceBindings"));XCTAssertFalse(text.contains(".csv"))
        XCTAssertTrue(document.nodes.allSatisfy{ !$0.name.isEmpty && $0.x.isFinite && $0.y.isFinite })
        XCTAssertEqual(document.nodes.filter{$0.address.domain == .model}.count,4)
        XCTAssertEqual(document.nodes.filter{$0.address.domain == .data}.count,6)
    }
    func testCSVFixtureHasExactMappingBalancedFiniteDistinctObservations() throws {
        let url = repo.appendingPathComponent("Sources/KwipMLApp/Resources/" + WorkflowExamples.observationsResourceName)
        let text = try String(contentsOf:url,encoding:.utf8)
        let rows = text.split(whereSeparator: \.isNewline).map{ $0.split(separator:",",omittingEmptySubsequences:false).map(String.init) }
        XCTAssertEqual(rows.first,["signal","context","label"])
        XCTAssertEqual(rows.count,65)
        var labels:[Int] = [], features = Set<String>()
        for row in rows.dropFirst() {
            XCTAssertEqual(row.count,3)
            XCTAssertTrue(try XCTUnwrap(Double(row[0])).isFinite)
            XCTAssertTrue(try XCTUnwrap(Double(row[1])).isFinite)
            let label = try XCTUnwrap(Int(row[2]));XCTAssertTrue([0,1].contains(label));labels.append(label)
            XCTAssertTrue(features.insert(row[0]+","+row[1]).inserted,"Features must not overlap across partitions")
        }
        XCTAssertEqual(labels.filter{$0 == 0}.count,32);XCTAssertEqual(labels.filter{$0 == 1}.count,32)
        XCTAssertEqual(try compile(WorkflowExamples.preparedParallelClassifier()).featureColumns,Array(try XCTUnwrap(rows.first).prefix(2)))
    }
}
