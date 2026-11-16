import XCTest
@testable import KwipMLCore

final class WorkflowTrainDataLoadingMigrationTests:XCTestCase {
    private var train:WorkflowNodeAddress {WorkflowExamples.trainAddress}
    private var dataset:WorkflowNodeAddress {.init(domain:.control,localID:"legacyDataset")}
    private var loader:WorkflowNodeAddress {.init(domain:.control,localID:"legacyLoader")}
    private var registry:DataFlowOperatorRegistry {get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))}}
    private func legacy()throws->WorkflowDocument {
        var document=try WorkflowExamples.preparedParallelClassifier(device:"cpu");document.schemaVersion=13
        let values:[String:JSONValue]=["kind":.string("subset"),"indices":.array([4,1,0]),"max_rows":100000]
        var batching=DataLoaderConfiguration.defaults;batching["batch_size"]=7;batching["shuffle"] = .bool(true);batching["seed"]=123
        document.nodes.append(.init(address:dataset,name:"Keep selected rows",x:140,y:50,payload:.datasetRecipe(type:DatasetConfiguration.componentID,configurationVersion:1,config:values)))
        document.nodes.append(.init(address:loader,name:"Group rows",x:140,y:200,payload:.batchRecipe(type:DataLoaderConfiguration.componentID,configurationVersion:1,config:batching)))
        document.edges.append(.init(id:"dataset-link",from:.init(address:dataset,port:"out"),to:.init(address:train,port:"dataset"),kind:.datasetRecipe))
        document.edges.append(.init(id:"loader-link",from:.init(address:loader,port:"out"),to:.init(address:train,port:"loader"),kind:.batchRecipe))
        return document
    }
    private func compile(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {try WorkflowCompiler.compile(document,train:train,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:12)}
    func testMigrationPreservesNormalizedExecutionAndUnrelatedSourceExactly()throws {
        let original=try legacy(),before=try WorkflowCompiler.compileHistoricalEvidence(original,train:train,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:12)
        let migrated=try WorkflowTrainDataLoading.migrateLegacyRecipes(original),after=try compile(migrated)
        XCTAssertEqual(migrated.projectID,original.projectID);XCTAssertEqual(migrated.revision,original.revision)
        XCTAssertEqual(migrated.name,original.name)
        XCTAssertEqual(migrated.nodes.filter {$0.address != train},original.nodes.filter {$0.address != train && $0.address != dataset && $0.address != loader})
        XCTAssertEqual(migrated.edges,original.edges.filter {$0.from.address != dataset && $0.from.address != loader})
        XCTAssertEqual(after.datasetRecipe?.configuration,before.datasetRecipe?.configuration)
        XCTAssertEqual(after.loaderRecipe?.configuration,before.loaderRecipe?.configuration)
        XCTAssertEqual(after.settings,before.settings)
        XCTAssertEqual(after.dataflow,before.dataflow);XCTAssertEqual(after.model,before.model)
        XCTAssertEqual(after.datasetRecipe?.source,train);XCTAssertEqual(after.loaderRecipe?.source,train)
        XCTAssertEqual(after.inlineDataLoadingRoles,["dataset","loader"])
        XCTAssertTrue(before.inlineDataLoadingRoles.isEmpty)
        XCTAssertEqual(try WorkflowTrainDataLoading.migrateLegacyRecipes(migrated),migrated)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(original)),original,"Historical decoding never migrates recipe nodes.")
    }
    func testSharedRecipesCopyToEveryTrainAndPreserveOtherTrainSettings()throws {
        var original=try legacy()
        let second=WorkflowNodeAddress(domain:.control,localID:"secondTrain")
        original.nodes.append(.init(address:second,name:"Second run",payload:.train(config:["trainingMode":.string("fineTune"),"objectiveScalarization":.string("sum")])))
        let shared=original.edges.filter {$0.to.address==train && ($0.kind == .datasetRecipe || $0.kind == .batchRecipe)}
        original.edges += shared.map {.init(id:$0.id+"-second",from:$0.from,to:.init(address:second,port:$0.to.port),kind:$0.kind)}
        let migrated=try WorkflowTrainDataLoading.migrateLegacyRecipes(original)
        let first=try XCTUnwrap(migrated.nodes.first {$0.address==train}),other=try XCTUnwrap(migrated.nodes.first {$0.address==second})
        guard case .train(let a)=first.payload,case .train(let b)=other.payload else {return XCTFail("Missing Train")}
        XCTAssertEqual(a["datasetAccess"],b["datasetAccess"]);XCTAssertEqual(a["batching"],b["batching"])
        XCTAssertEqual(b["trainingMode"],.string("fineTune"));XCTAssertEqual(b["objectiveScalarization"],.string("sum"))
        XCTAssertFalse(migrated.nodes.contains {$0.address==dataset || $0.address==loader})
    }
    func testOrphanMalformedEdgeAndInlineConflictFailWithoutMutatingSource()throws {
        for scenario in ["orphan","wrongPort","wrongKind","inline","duplicate","incoming"] {
            var original=try legacy()
            switch scenario {
            case "orphan":original.edges.removeAll {$0.from.address==dataset}
            case "wrongPort":original.edges[original.edges.count-1].to.port="dataset"
            case "wrongKind":original.edges[original.edges.count-1].kind = .table
            case "inline":
                let i=try XCTUnwrap(original.nodes.firstIndex {$0.address==train})
                original.nodes[i].payload = .train(config:["datasetAccess":.object(DatasetConfiguration.defaults)])
            case "duplicate":
                var duplicate=try XCTUnwrap(original.edges.first {$0.id=="dataset-link"});duplicate.id="duplicate-link";original.edges.append(duplicate)
            default:original.edges.append(.init(from:.init(address:train,port:"model"),to:.init(address:dataset,port:"in"),kind:.datasetRecipe))
            }
            let unchanged=original
            XCTAssertThrowsError(try WorkflowTrainDataLoading.migrateLegacyRecipes(original),scenario)
            XCTAssertEqual(original,unchanged,scenario)
        }
    }
    func testInlineBatchingDecoderCanonicalizesOnlyItsTimeoutValue()throws {
        var original=try WorkflowExamples.preparedParallelClassifier(device:"cpu");original.schemaVersion=13
        let index=try XCTUnwrap(original.nodes.firstIndex {$0.address==train})
        original.nodes[index].payload = .train(config:["trainingMode":.string("fineTune"),"batching":.object(DataLoaderConfiguration.defaults)])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(original)),original)
        var integer=DataLoaderConfiguration.defaults;integer["timeout"] = .integer(0)
        original.nodes[index].payload = .train(config:["trainingMode":.string("fineTune"),"batching":.object(integer)])
        let decoded=try WorkflowCodec.decode(WorkflowCodec.encode(original))
        guard case .train(let config)=decoded.nodes[index].payload,case .object(let canonical)=config["batching"] else {return XCTFail("Missing batching")}
        XCTAssertEqual(canonical["timeout"],.number(0));XCTAssertEqual(config["trainingMode"],.string("fineTune"))
        XCTAssertNil(config["datasetAccess"])
    }
}
