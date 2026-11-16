import XCTest
@testable import KwipMLCore

final class DatasetLoaderWorkflowCompilerTests:XCTestCase {
    private func document()throws->WorkflowDocument {
        var doc=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        doc.schemaVersion=13
        let dataset=WorkflowNodeAddress(domain:.control,localID:"chosenRows")
        let loader=WorkflowNodeAddress(domain:.control,localID:"groups")
        var values=DataLoaderConfiguration.defaults;values["batch_size"]=3;values["shuffle"]=true;values["seed"]=99
        doc.nodes += [.init(address:dataset,payload:.datasetRecipe(type:"dataset_source",configurationVersion:1,config:["kind":.string("subset"),"indices":.array([4,1,3,0]),"max_rows":100000])),.init(address:loader,payload:.batchRecipe(type:"data_loader",configurationVersion:1,config:values))]
        doc.edges += [.init(from:.init(address:dataset,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"dataset"),kind:.datasetRecipe),.init(from:.init(address:loader,port:"out"),to:.init(address:WorkflowExamples.trainAddress,port:"loader"),kind:.batchRecipe)]
        return doc
    }
    private func compile(_ doc:WorkflowDocument,version:Int=12)throws->WorkflowExecutionPlan {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compile(RetiredTrainingRecipeMigration.migrate(doc),train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),contractVersion:version)
    }
    func testSelectedRecipesKeepSourceAuthorityAndOwnEffectiveBatchSize()throws {
        let doc=try document(),plan=try compile(doc)
        XCTAssertEqual(plan.datasetRecipe?.configuration.indices,[4,1,3,0])
        XCTAssertEqual(plan.datasetRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(plan.loaderRecipe?.configuration.batchSize,3)
        XCTAssertEqual(plan.loaderRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(plan.settings.batchSize,3)
        XCTAssertEqual(plan.featureColumns,["signal","context"])
        XCTAssertFalse(plan.model.graph.nodes.contains {$0.type=="dataset_source" || $0.type=="data_loader"})
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)),doc)
    }
    func testWrongKindOlderCompilerAndShuffledIterableRefuse()throws {
        var doc=try document()
        XCTAssertThrowsError(try compile(doc,version:11))
        let original=doc
        doc.edges[doc.edges.count-1].kind = .datasetRecipe
        XCTAssertThrowsError(try compile(doc))
        doc=original
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address.localID=="chosenRows"})
        doc.nodes[index].payload = .datasetRecipe(type:"dataset_source",configurationVersion:1,config:["kind":.string("boundedIterable"),"indices":.array([]),"max_rows":20])
        XCTAssertThrowsError(try compile(doc))
    }
    func testAbsentRecipesKeepLegacyBatchAndConnections()throws {
        let doc=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let old=try compile(doc,version:11),current=try compile(doc)
        XCTAssertEqual(old,current)
        XCTAssertNil(current.datasetRecipe);XCTAssertNil(current.loaderRecipe)
    }
}
