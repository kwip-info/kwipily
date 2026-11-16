import XCTest
@testable import KwipMLCore
final class PreprocessContractTests:XCTestCase {
    private var root:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testModesPortsPracticeAndOptionalPartitions()throws {
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let definition=try XCTUnwrap(registry.operatorDefinition(id:"kwip.data.preprocess"))
        let partitioned=definition.configured(PreprocessConfiguration().values)
        XCTAssertEqual(partitioned.inputs.map(\.id),["training","validation","test"])
        XCTAssertEqual(partitioned.inputs.filter(\.required).map(\.id),["training"])
        XCTAssertEqual(partitioned.authoringOutputs.map(\.id),["training","validation","test"])
        XCTAssertEqual(definition.executionOutputs(PreprocessConfiguration().values,connectedInputs:["training"]).map(\.id),["training","state"])
        let table=definition.configured(PreprocessConfiguration(mode:"table").values)
        XCTAssertEqual(table.inputs.map(\.id),["table"]);XCTAssertEqual(table.authoringOutputs.map(\.id),["table"])
        let document=try PreprocessPracticeExample.document()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document)
        let graph=try WorkflowDataPreparationPlan.compile(document,registry:registry)
        XCTAssertEqual(graph.nodes.filter{$0.operatorID=="kwip.data.preprocess"}.count,1)
        var optional=document
        optional.nodes.removeAll{$0.address.localID=="test-view"}
        optional.edges.removeAll{$0.to.port=="test" || $0.to.address.localID=="test-view"}
        _=try WorkflowDataPreparationPlan.compile(optional,registry:registry)
        optional.edges.removeAll{$0.to.address==PreprocessPracticeExample.address && $0.to.port=="training"}
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(optional,registry:registry))
    }
    func testSinglePreprocessFeedsModelMappingAndProtectedEvaluation()throws {
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let practice=try PreprocessPracticeExample.document()
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        document.nodes.removeAll{$0.address.domain == .data}
        document.edges.removeAll{$0.from.address.domain == .data || $0.to.address.domain == .data}
        document.nodes += practice.nodes;document.edges += practice.edges
        let mapping=WorkflowNodeAddress(domain:.control,localID:"mapping")
        let index=try XCTUnwrap(document.nodes.firstIndex{$0.address==mapping})
        document.nodes[index].payload = .trainingData(config:["featureColumns":.array([.string("x"),.string("time")]),"targetColumn":.string("target")])
        for role in ["training","validation"] {document.edges.append(.init(from:.init(address:PreprocessPracticeExample.address,port:role),to:.init(address:mapping,port:role),kind:.table))}
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(plan.training,.init(node:"preprocess",port:"training"));XCTAssertEqual(plan.validation,.init(node:"preprocess",port:"validation"))
        var invalid=document
        let edge=try XCTUnwrap(invalid.edges.firstIndex{$0.to.address==PreprocessPracticeExample.address && $0.to.port=="validation"})
        invalid.edges[edge].from.port="training"
        XCTAssertThrowsError(try WorkflowCompiler.compile(invalid,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry))
        var frozen=practice;frozen.schemaVersion=5
        let model=WorkflowNodeAddress(domain:.control,localID:"frozen"),evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        frozen.nodes += [.init(address:model,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:["device":.string("cpu"),"batchSize":2,"previewLimit":3]))]
        frozen.edges += [.init(from:.init(address:model,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),.init(from:.init(address:PreprocessPracticeExample.address,port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
        let evaluation=try WorkflowEvaluationCompiler.compile(frozen,evaluate:evaluate,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(evaluation.testLineage.states,[PreprocessPracticeExample.address])
        XCTAssertEqual(evaluation.canonicalTestEndpoint,.init(node:"partition",port:"test"))
    }
    func testTypedConfigurationRejectsLearningInFixedModeAndRoundTrips()throws {
        let config=PreprocessPracticeExample.configuration
        XCTAssertEqual(try PreprocessConfiguration(values:config.values),config)
        var fixed=config;fixed.mode="table";XCTAssertThrowsError(try fixed.validate())
        var invalid=config;invalid.columns.append(invalid.columns[0]);XCTAssertThrowsError(try invalid.validate())
        invalid=config;invalid.columns[1].scaling="standard";XCTAssertThrowsError(try invalid.validate())
        invalid=config;invalid.columns[1].maxCategories=129;XCTAssertThrowsError(try invalid.validate())
    }
    func testCleanupOptionsLegacyDecodingAndNoLearningInTableMode()throws {
        var column=PreprocessConfiguration.Column(column:"name")
        column.unicodeNormalization="NFKC";column.removeControls=true;column.removePunctuation=true
        column.removeDigits=true;column.whitespace="collapse";column.casePolicy="lower";column.emptyAfterCleanup="missing"
        let config=PreprocessConfiguration(mode:"table",columns:[column])
        XCTAssertEqual(try PreprocessConfiguration(values:config.values),config)
        var legacy=config.values
        if case .array(var rows)=legacy["columns"],case .object(var row)=rows[0] {
            for key in ["unicodeNormalization","removeControls","removePunctuation","removeDigits","whitespace","case","emptyAfterCleanup"] {row.removeValue(forKey:key)}
            rows[0] = .object(row);legacy["columns"] = .array(rows)
        }
        XCTAssertFalse(try PreprocessConfiguration(values:legacy).columns[0].hasTextCleanup)
        XCTAssertEqual(try PreprocessConfiguration(values:legacy).values,legacy,"Opening legacy controls must not create a pending draft.")
        var partitioned=config;partitioned.mode="partitions"
        XCTAssertThrowsError(try partitioned.validate())
    }

}
