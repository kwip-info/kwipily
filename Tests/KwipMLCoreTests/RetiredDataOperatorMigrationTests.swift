import XCTest
@testable import KwipMLCore

final class RetiredDataOperatorMigrationTests:XCTestCase {
    private func node(_ id:String,_ op:String,_ config:[String:JSONValue])->DataFlowNode {.init(id:id,operatorID:"kwip.data."+op,config:config)}
    private func edge(_ a:String,_ ap:String,_ b:String,_ bp:String)->DataFlowEdge {.init(from:.init(node:a,port:ap),to:.init(node:b,port:bp))}
    func testExactColumnUpgradeKeepsNamesAndSourceOrderWithKnownSchema()throws {
        let old=DataFlowDocument(nodes:[node("cast","cast",["columns":.object(["x":.string("float64")]),"missingPolicy":.string("preserve")])])
        XCTAssertThrowsError(try RetiredDataOperatorMigration.migrate(old))
        let schema:[[String:JSONValue]]=[["name":.string("id"),"dtype":.string("string")],["name":.string("x"),"dtype":.string("string")],["name":.string("tag"),"dtype":.string("string")]]
        let current=try RetiredDataOperatorMigration.migrate(old,knownInputSchemas:["cast":schema])
        XCTAssertEqual(current.nodes[0].operatorID,"kwip.data.columns")
        guard case .array(let rows)=current.nodes[0].config["rows"] else {return XCTFail("Missing current column grid")}
        XCTAssertEqual(rows.compactMap {if case .object(let r)=$0 {return r["source"]?.stringValue};return nil},["id","x","tag"])
        XCTAssertEqual(try RetiredDataOperatorMigration.migrate(current),current)
        XCTAssertEqual(old.nodes[0].operatorID,"kwip.data.cast","Pure migration never changes the supplied source.")
    }
    func testAmbiguousSplitAndUnknownReplacementAreExplicitRepair()throws {
        let split=DataFlowDocument(nodes:[node("split","split",["seed":42,"validationPercent":20])])
        XCTAssertThrowsError(try RetiredDataOperatorMigration.migrate(split)) {error in XCTAssertTrue(error.localizedDescription.contains("Original source is unchanged"))}
        let formula=DataFlowDocument(nodes:[node("formula","calculate",["expression":.string("[x]+1"),"outputColumn":.string("x"),"dtype":.string("float64"),"replaceExisting":true,"nullPolicy":.string("preserve")])])
        XCTAssertThrowsError(try RetiredDataOperatorMigration.migrate(formula))
    }
    func testFormulaAndFilterUseCurrentControlsAndKeepNodeIdentity()throws {
        let old=DataFlowDocument(nodes:[node("calculate","calculate",["expression":.string("[x]*2"),"outputColumn":.string("twice"),"dtype":.string("float64"),"replaceExisting":false,"nullPolicy":.string("error")]),node("filter","filterExpression",["expression":.string("[twice]>1"),"nullPolicy":.string("nonmatching")])],edges:[edge("calculate","table","filter","table")])
        let current=try RetiredDataOperatorMigration.migrate(old)
        XCTAssertEqual(current.nodes.map(\.id),old.nodes.map(\.id));XCTAssertEqual(current.edges,old.edges)
        XCTAssertEqual(current.nodes.map(\.operatorID),["kwip.data.formula","kwip.data.filterRows"])
        XCTAssertEqual(current.nodes[1].config["mode"],.string("expression"))
        XCTAssertEqual(try DataFlowCodec.decodeGraph(DataFlowCodec.encodeGraph(current)),current)
    }
    func testFittedPairFusionRewiresExternalConsumersAndPreservesOtherWorkflowMetadata()throws {
        let source=WorkflowNodeAddress(domain:.data,localID:"partition"),fit=WorkflowNodeAddress(domain:.data,localID:"fit"),train=WorkflowNodeAddress(domain:.data,localID:"training"),validation=WorkflowNodeAddress(domain:.data,localID:"validation"),mapping=WorkflowNodeAddress(domain:.control,localID:"mapping")
        let nodes:[WorkflowNode]=[
            .init(address:source,name:"Known partitions",x:10,y:20,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":20,"testPercent":20])),
            .init(address:fit,name:"Scale selected measurement",x:30,y:40,payload:.dataOperator(operatorID:"kwip.data.standardScalerFit",config:["columns":.array([.string("x")])])),
            .init(address:train,x:0,y:0,payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:validation,x:0,y:0,payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:mapping,x:50,y:60,payload:.trainingData(config:["featureColumns":.array([.string("x")]),"targetColumn":.string("target")]))]
        var edges:[WorkflowEdge]=[.init(id:"fit-source",from:.init(address:source,port:"training"),to:.init(address:fit,port:"training"),kind:.table)]
        for (address,role) in [(train,"training"),(validation,"validation")] {
            edges += [.init(id:role+"-source",from:.init(address:source,port:role),to:.init(address:address,port:"table"),kind:.table),.init(id:role+"-state",from:.init(address:fit,port:"state"),to:.init(address:address,port:"state"),kind:.fittedState),.init(id:role+"-mapping",from:.init(address:address,port:"table"),to:.init(address:mapping,port:role),kind:.table)]
        }
        let original=WorkflowDocument(projectID:"migration-fixture",revision:7,name:"Exact saved graph",nodes:nodes,edges:edges)
        XCTAssertTrue(RetiredDataOperatorMigration.requiresUpgrade(original))
        let current=try RetiredDataOperatorMigration.migrate(original)
        XCTAssertFalse(RetiredDataOperatorMigration.requiresUpgrade(current))
        XCTAssertEqual(current.projectID,original.projectID);XCTAssertEqual(current.revision,original.revision);XCTAssertEqual(current.name,original.name)
        let kept=try XCTUnwrap(current.nodes.first {$0.address==fit});XCTAssertEqual(kept.name,nodes[1].name);XCTAssertEqual(kept.x,30);XCTAssertEqual(kept.y,40)
        guard case .dataOperator(let id,let config)=kept.payload else {return XCTFail("Missing current Preprocess")}
        XCTAssertEqual(id,"kwip.data.preprocess");XCTAssertEqual(try PreprocessConfiguration(values:config).columns[0].scaling,"standard")
        for role in ["training","validation"] {
            let edge=try XCTUnwrap(current.edges.first {$0.id==role+"-mapping"})
            XCTAssertEqual(edge.from,.init(address:fit,port:role));XCTAssertEqual(edge.to,.init(address:mapping,port:role))
        }
        XCTAssertEqual(try RetiredDataOperatorMigration.migrate(current),current)
        XCTAssertTrue(RetiredDataOperatorMigration.requiresUpgrade(original))
    }

    private var registry:DataFlowOperatorRegistry {get throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try .load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
    }}
    func testLegacyDatasetPartitionEdgesUpgradeToCompilableInternalAndExternalTableEdges()throws {
        func a(_ id:String,_ domain:WorkflowDomain = .data)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
        let source=a("source"),partition=a("partition"),fit=a("fit"),mapping=a("mapping",.control),frozen=a("frozen",.control),evaluate=a("evaluate",.control)
        var nodes:[WorkflowNode]=[
            .init(address:source,name:"Original source",x:7,y:9,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(";")])),
            .init(address:partition,name:"Original groups",x:17,y:19,payload:.dataOperator(operatorID:"kwip.data.groupSplit",config:["groupBy":.array([.string("group")]),"nullKeys":.string("error"),"seed":42,"allocationUnit":.string("groups"),"validationPercent":20,"testPercent":20])),
            .init(address:fit,name:"Original scaler",x:27,y:29,payload:.dataOperator(operatorID:"kwip.data.standardScalerFit",config:["columns":.array([.string("x")])])),
            .init(address:mapping,payload:.trainingData(config:["featureColumns":.array([.string("x")]),"targetColumn":.string("label")])),
            .init(address:frozen,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:[:])),
            .init(address:a("unrelated-input",.model),payload:.modelPrimitive(type:"input",configurationVersion:1,config:["features":1])),
            .init(address:a("unrelated-output",.model),payload:.modelPrimitive(type:"output",configurationVersion:1,config:[:]))]
        var edges:[WorkflowEdge]=[
            .init(id:"source-to-partition",from:.init(address:source,port:"table"),to:.init(address:partition,port:"table"),kind:.table),
            .init(id:"fit-source",from:.init(address:partition,port:"training"),to:.init(address:fit,port:"training"),kind:.dataset),
            .init(id:"test-to-evaluation",from:.init(address:partition,port:"test"),to:.init(address:evaluate,port:"test"),kind:.dataset),
            .init(id:"frozen-to-evaluation",from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),
            .init(id:"unrelated-tensor",from:.init(address:a("unrelated-input",.model),port:"out"),to:.init(address:a("unrelated-output",.model),port:"in"),kind:.tensor)]
        for role in ["training","validation"] {
            let transform=a(role)
            nodes.append(.init(address:transform,payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])))
            edges += [.init(id:role+"-source",from:.init(address:partition,port:role),to:.init(address:transform,port:"table"),kind:.dataset),.init(id:role+"-state",from:.init(address:fit,port:"state"),to:.init(address:transform,port:"state"),kind:.fittedState),.init(id:role+"-mapping",from:.init(address:transform,port:"table"),to:.init(address:mapping,port:role),kind:.table)]
        }
        var original=WorkflowDocument(projectID:UUID().uuidString,revision:11,name:"Dataset edge migration",nodes:nodes,edges:edges)
        original.schemaVersion=5
        let before=original,current=try RetiredDataOperatorMigration.migrate(original)
        let region=try WorkflowCompiler.dependencies(current,terminal:mapping,modelRegistry:.builtIn(),dataRegistry:registry)
        let lineage=try WorkflowDataLineage.resolve(region,dataRegistry:registry)
        for role in ["training","validation"] {
            guard case .rows(let rows)=lineage[.init(address:fit,port:role)] else {return XCTFail("Missing migrated role")}
            XCTAssertEqual(rows.role,role);XCTAssertEqual(rows.source,source);XCTAssertEqual(rows.split,partition);XCTAssertEqual(rows.states,[fit])
        }
        let evaluation=try WorkflowEvaluationCompiler.compile(current,evaluate:evaluate,modelRegistry:.builtIn(),dataRegistry:registry)
        XCTAssertEqual(evaluation.canonicalTestEndpoint,.init(node:"partition",port:"test"))
        XCTAssertEqual(evaluation.testLineage.role,"test");XCTAssertEqual(evaluation.membershipValidation,.exactArchivedProtectedPopulation)
        for id in ["fit-source","test-to-evaluation"] {XCTAssertEqual(current.edges.first {$0.id==id}?.kind,.table)}
        for id in ["source-to-partition","frozen-to-evaluation","unrelated-tensor"] {XCTAssertEqual(current.edges.first {$0.id==id},before.edges.first {$0.id==id})}
        for id in ["training-mapping","validation-mapping"] {XCTAssertNotNil(current.edges.first {$0.id==id})}
        XCTAssertEqual(current.nodes.first {$0.address==source},before.nodes.first {$0.address==source})
        XCTAssertEqual(current.projectID,before.projectID);XCTAssertEqual(current.revision,before.revision);XCTAssertEqual(current.name,before.name)
        XCTAssertEqual(original,before);XCTAssertEqual(try RetiredDataOperatorMigration.migrate(current),current)
    }
    private func chainedFits()->DataFlowDocument {
        var nodes=[node("s","sourceCSV",[:]),node("p","temporalSplit",["timeColumn":.string("time"),"timeEncoding":.string("epochMilliseconds"),"trainingEnd":2,"validationEnd":4,"embargoMilliseconds":0,"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")]),node("impute","imputerFit",["columns":.array([.string("x")]),"strategy":.string("mean"),"emptyPolicy":.string("error")]),node("scale","standardScalerFit",["columns":.array([.string("x")])])]
        var edges=[edge("s","table","p","table"),edge("p","training","impute","training"),edge("training-imputed","table","scale","training")]
        for role in ["training","validation","test"] {
            nodes += [node(role+"-imputed","transform",[:]),node(role+"-scaled","transform",[:]),node(role+"-view","view",[:])]
            edges += [edge("p",role,role+"-imputed","table"),edge("impute","state",role+"-imputed","state"),edge(role+"-imputed","table",role+"-scaled","table"),edge("scale","state",role+"-scaled","state"),edge(role+"-scaled","table",role+"-view","table")]
        }
        return .init(nodes:nodes,edges:edges)
    }
    func testTwoStageFitUpgradeIsIndependentOfSerializedNodeOrder()throws {
        let ordered=chainedFits();var reversed=ordered;reversed.nodes.reverse();let before=reversed
        let expected=try RetiredDataOperatorMigration.migrate(ordered),actual=try RetiredDataOperatorMigration.migrate(reversed)
        XCTAssertEqual(Dictionary(uniqueKeysWithValues:actual.nodes.map {($0.id,$0)}),Dictionary(uniqueKeysWithValues:expected.nodes.map {($0.id,$0)}))
        XCTAssertEqual(Set(actual.edges),Set(expected.edges))
        XCTAssertEqual(actual.nodes.map(\.id),before.nodes.filter {$0.operatorID != "kwip.data.transform"}.map(\.id))
        XCTAssertEqual(reversed,before);XCTAssertEqual(try RetiredDataOperatorMigration.migrate(actual),actual)
        XCTAssertEqual(Set(actual.nodes.filter {$0.operatorID=="kwip.data.preprocess"}.map(\.id)),["impute","scale"])
    }
    func testCyclicOrUnprovableFitChainRefusesWithoutPartialUpgrade()throws {
        for cycle in [true,false] {
            var original=chainedFits()
            let index=try XCTUnwrap(original.edges.firstIndex {$0.to == .init(node:"impute",port:"training")})
            original.edges[index].from = .init(node:cycle ? "training-scaled":"s",port:"table")
            original.nodes.reverse();let before=original
            XCTAssertThrowsError(try RetiredDataOperatorMigration.migrate(original)) {error in
                XCTAssertTrue(cycle ? error.localizedDescription.lowercased().contains("cycle"):error.localizedDescription.contains("Original source is unchanged"),error.localizedDescription)
            }
            XCTAssertEqual(original,before)
        }
    }
}
