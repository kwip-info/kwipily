import XCTest
@testable import KwipMLCore

final class WorkflowCompilerTests:XCTestCase {
    /// These saved-source fixtures prove historical compiler contracts, including
    /// old fitted-state wires and schema2/3 continuation behavior. They do not
    /// authorize execution with retired tools in the current authoring catalog.
    private func capturedDataRegistry()throws->DataFlowOperatorRegistry {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var registry=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let retired=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/retired-data-operators-v1.json"))
        registry.operators += retired.operators
        return registry
    }
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    private func fixture(block:Bool=false)->WorkflowDocument {
        let base=ExampleFlows.parallelClassifier
        let source=CompositeExamples.parallelClassifier
        var nodes:[WorkflowNode]=[
            .init(address:address(.data,"input"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),
            .init(address:address(.data,"cast"),payload:.dataOperator(operatorID:"kwip.data.cast",config:["columns":.object(["a":.string("float64"),"b":.string("float64"),"label":.string("int64")])])),
            .init(address:address(.data,"split"),payload:.dataOperator(operatorID:"kwip.data.split",config:["seed":.integer(42),"validationPercent":.integer(20),"stratifyColumn":.string("label")])),
            .init(address:address(.data,"fit"),payload:.dataOperator(operatorID:"kwip.data.standardScalerFit",config:["columns":.array([.string("a"),.string("b")])])),
            .init(address:address(.data,"training"),payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:address(.data,"validation"),payload:.dataOperator(operatorID:"kwip.data.transform",config:[:])),
            .init(address:address(.control,"model"),payload:.modelDefinition),
            .init(address:address(.control,"mapping"),payload:.trainingData(config:["featureColumns":.array([.string("a"),.string("b")]),"targetColumn":.string("label")])),
            .init(address:address(.control,"setup"),payload:.runSetup(config:["device":.string("cpu"),"epochs":.integer(3),"batchSize":.integer(32),"seed":.integer(17),"learningRate":.number(0.02)])),
            .init(address:address(.control,"train"),payload:.train(config:[:]))]
        let primitives=block ? source.root.nodes : base.nodes
        nodes += primitives.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelPrimitive(type:$0.type,configurationVersion:block ? (source.root.configurationVersion ?? 1) : base.schemaVersion,config:$0.config))}
        if block {nodes += source.root.instances.map {.init(address:address(.model,$0.id),name:$0.name,x:$0.x,y:$0.y,payload:.modelBlock(reference:$0.definition,overrides:$0.overrides))}}
        var edges=[edge(.data,"input","table",.data,"cast","table",.table),edge(.data,"cast","table",.data,"split","table",.table),edge(.data,"split","training",.data,"fit","training",.dataset),edge(.data,"split","training",.data,"training","table",.dataset),edge(.data,"split","validation",.data,"validation","table",.dataset),edge(.data,"fit","state",.data,"training","state",.fittedState),edge(.data,"fit","state",.data,"validation","state",.fittedState),edge(.data,"training","table",.control,"mapping","training",.table),edge(.data,"validation","table",.control,"mapping","validation",.table),edge(.model,"output","definition",.control,"model","source",.modelDefinition),edge(.control,"model","model",.control,"train","model",.modelDefinition),edge(.control,"mapping","data",.control,"train","data",.trainingData),edge(.control,"setup","configuration",.control,"train","configuration",.runConfiguration)]
        edges += (block ? source.root.edges : base.edges).map {.init(id:$0.id,from:.init(address:address(.model,$0.source),port:$0.sourcePort),to:.init(address:address(.model,$0.target),port:$0.targetPort),kind:.tensor)}
        return WorkflowDocument(projectID:UUID().uuidString,revision:7,name:"Measured workflow",nodes:nodes,edges:edges,definitions:block ? source.definitions : [])
    }
    private func compile(_ document:WorkflowDocument)throws->WorkflowExecutionPlan {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCompiler.compileHistoricalEvidence(document,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:capturedDataRegistry())
    }
    func testMappingRequirementsDescribeEveryConsumerWithoutRequiringDataOrSetup()throws {
        var doc=fixture()
        doc.nodes.removeAll {$0.address.domain == .data || $0.address == address(.control,"setup")}
        doc.edges.removeAll {$0.from.address.domain == .data || $0.to.address.domain == .data || $0.from.address == address(.control,"setup")}
        let second=address(.control,"secondTrain")
        doc.nodes.append(.init(address:second,name:"Second run",payload:.train(config:[:])))
        doc.edges.append(edge(.control,"mapping","data",.control,"secondTrain","data",.trainingData))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try capturedDataRegistry()
        let result=try WorkflowMappingRequirements.resolve(doc,mapping:address(.control,"mapping"),modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(result.count,2)
        XCTAssertEqual(result[0].train,address(.control,"train"))
        XCTAssertEqual(result[0].contract,try TypedTensorContract(sample:[2],dtype:.float32))
        XCTAssertNil(result[0].problem)
        XCTAssertEqual(result[1].train,second)
        XCTAssertEqual(result[1].label,"Second run")
        XCTAssertNil(result[1].contract)
        XCTAssertNotNil(result[1].problem)
        doc.edges.append(edge(.control,"model","model",.control,"secondTrain","model",.modelDefinition))
        let connected=try WorkflowMappingRequirements.resolve(doc,mapping:address(.control,"mapping"),modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertEqual(connected.map(\.contract),[result[0].contract,result[0].contract])
        let modelEdge=try XCTUnwrap(doc.edges.firstIndex {$0.to.address==second && $0.to.port=="model"})
        doc.edges[modelEdge].kind = .table
        let malformed=try WorkflowMappingRequirements.resolve(doc,mapping:address(.control,"mapping"),modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertNil(malformed[1].contract);XCTAssertNotNil(malformed[1].problem)
        doc.edges.removeAll {$0.from.address==address(.control,"mapping")}
        XCTAssertTrue(try WorkflowMappingRequirements.resolve(doc,mapping:address(.control,"mapping"),modelRegistry:.builtIn(),dataRegistry:data).isEmpty)
    }
    func testMappingCountMustMatchModelSampleShape()throws {
        var doc=fixture()
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address==address(.control,"mapping")})
        doc.nodes[index].payload = .trainingData(config:["featureColumns":.array([.string("a")]),"targetColumn":.string("label")])
        XCTAssertThrowsError(try compile(doc)) { error in
            XCTAssertTrue(error.localizedDescription.contains("requires 2"),error.localizedDescription)
            XCTAssertEqual((error as? WorkflowCompilerError)?.node,self.address(.control,"mapping"))
        }
    }
    func testIntegerMappingUsesFlattenedMultidimensionalSampleContract()throws {
        var doc=fixture()
        var graph=EmbeddingGraphTests().graph()
        graph.nodes[0].config["shape"] = .array([1,3])
        graph.nodes[3].config["in_features"] = 9
        doc.nodes.removeAll {$0.address.domain == .model}
        doc.edges.removeAll {$0.from.address.domain == .model || $0.to.address.domain == .model}
        doc.nodes += graph.nodes.map {.init(address:address(.model,$0.id),payload:.modelPrimitive(type:$0.type,configurationVersion:2,config:$0.config))}
        doc.edges += graph.edges.map {edge(.model,$0.source,$0.sourcePort,.model,$0.target,$0.targetPort,.tensor)}
        doc.edges.append(edge(.model,"output","definition",.control,"model","source",.modelDefinition))
        let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address==address(.control,"mapping")})
        doc.nodes[index].payload = .trainingData(config:["featureColumns":.array(["third","first","second"].map(JSONValue.string)),"targetColumn":.string("label")])
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try EmbeddingGraphTests().registry()
        let data=try capturedDataRegistry()
        let plan=try WorkflowCompiler.compileHistoricalEvidence(doc,train:address(.control,"train"),modelRegistry:registry,dataRegistry:data)
        XCTAssertEqual(plan.featureColumns,["third","first","second"])
        let guidance=try WorkflowMappingRequirements.resolve(doc,mapping:address(.control,"mapping"),modelRegistry:registry,dataRegistry:data)
        XCTAssertEqual(guidance.first?.contract,try TypedTensorContract(sample:[1,3],dtype:.int64))
        XCTAssertNil(guidance.first?.problem)
        doc.nodes[index].payload = .trainingData(config:["featureColumns":.array(["first","second"].map(JSONValue.string)),"targetColumn":.string("label")])
        XCTAssertThrowsError(try WorkflowCompiler.compileHistoricalEvidence(doc,train:address(.control,"train"),modelRegistry:registry,dataRegistry:data)) { error in
            XCTAssertTrue(error.localizedDescription.contains("requires 3"),error.localizedDescription)
        }
    }
    func testVersionThreeRetainsFreshPlanButNeverSilentlyIgnoresContinuationMode() throws {
        let original = fixture()
        let baseline = try compile(original)
        var migrated = try WorkflowCodec.migrateToVersion3(original)
        XCTAssertEqual(try compile(migrated), baseline)
        let index = try XCTUnwrap(migrated.nodes.firstIndex { $0.address == address(.control, "train") })
        for mode in [WorkflowTrainingMode.exactResume, .fineTune] {
            migrated.nodes[index].payload = .train(config: ["trainingMode": .string(mode.rawValue)])
            XCTAssertThrowsError(try compile(migrated)) { error in
                XCTAssertEqual((error as? WorkflowCompilerError)?.code, .configuration)
                XCTAssertTrue(error.localizedDescription.contains("not be run as fresh"))
            }
        }
    }
    func testCompilerVersionThreePreservesLifecycleWithoutChangingPhaseInputs()throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try capturedDataRegistry()
        let original=fixture(),baseline=try compile(original)
        for mode in WorkflowTrainingMode.allCases {
            var doc=try WorkflowCodec.migrateToVersion3(original)
            let index=try XCTUnwrap(doc.nodes.firstIndex {$0.address==address(.control,"train")})
            doc.nodes[index].payload = .train(config:["trainingMode":.string(mode.rawValue)])
            let plan=try WorkflowCompiler.compileHistoricalEvidence(doc,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:3)
            XCTAssertEqual(plan.trainingMode,mode)
            XCTAssertEqual(plan.dataflow,baseline.dataflow)
            XCTAssertEqual(plan.model,baseline.model)
            XCTAssertEqual(plan.settings,baseline.settings)
            if mode != .fresh {XCTAssertThrowsError(try compile(doc))}
        }
        XCTAssertThrowsError(try WorkflowCompiler.compileHistoricalEvidence(original,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:99))
    }
    func testDeclaredPortsRemainAvailableForIncompleteAuthoring()throws {
        let node=WorkflowNode(address:address(.model,"unfinished"),payload:.modelPrimitive(type:"linear",configurationVersion:2,config:[:]))
        let doc=WorkflowDocument(projectID:UUID().uuidString,nodes:[node])
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let ports=try WorkflowPortContracts.resolve(node,document:doc,modelRegistry:.builtIn(),dataRegistry:capturedDataRegistry())
        XCTAssertEqual(ports.inputs.map(\.id),["in"]);XCTAssertEqual(ports.outputs.map(\.id),["out"])
        XCTAssertEqual(ports.inputs.first?.kind,.tensor);XCTAssertEqual(ports.inputs.first?.maximumConnections,1)
        expect(doc,.invalidSelection)
    }
    func testSelectedExecutionHonorsDeclaredOutputConnectionLimit()throws {
        let doc=fixture()
        _ = try compile(doc)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var registry=try capturedDataRegistry()
        let index=try XCTUnwrap(registry.operators.firstIndex{$0.id=="kwip.data.standardScalerFit"})
        registry.operators[index].outputs[0].maximumConnections=1
        XCTAssertThrowsError(try WorkflowCompiler.compileHistoricalEvidence(doc,train:address(.control,"train"),modelRegistry:.builtIn(),dataRegistry:registry)) { error in
            let failure=error as? WorkflowCompilerError
            XCTAssertEqual(failure?.code,.invalidPort)
            XCTAssertEqual(failure?.node,self.address(.data,"fit"))
            XCTAssertEqual(failure?.port,"state")
        }
    }
    func testSlicesTypedDependenciesAndPreservesPrimitiveIDsAndExplicitMapping()throws {
        var doc=fixture()
        doc.nodes.append(.init(address:address(.data,"unrelated"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:[:])))
        let plan=try compile(doc)
        XCTAssertEqual(plan.projectID,doc.projectID);XCTAssertEqual(plan.revision,7)
        XCTAssertEqual(Set(plan.dataflow.nodes.map(\.id)),["input","cast","split","fit","training","validation"])
        XCTAssertEqual(plan.sourceAddresses,[address(.data,"input")])
        XCTAssertEqual(plan.training,.init(node:"training",port:"table"));XCTAssertEqual(plan.validation,.init(node:"validation",port:"table"))
        XCTAssertEqual(plan.featureColumns,["a","b"]);XCTAssertEqual(plan.targetColumn,"label")
        XCTAssertEqual(plan.settings.device,"cpu");XCTAssertEqual(plan.settings.seed,17);XCTAssertEqual(plan.settings.batchSize,32)
        XCTAssertEqual(plan.model.graph.nodes,ExampleFlows.parallelClassifier.nodes)
        XCTAssertEqual(plan.model.graph.edges,ExampleFlows.parallelClassifier.edges)
        XCTAssertNil(plan.settings.preparedTraining,"Artifact paths do not exist until preparation completes")
        XCTAssertEqual(try compile(doc),plan,"Compilation is deterministic")
    }
    func testLegacyModelConfigurationRemainsVersionOneWithoutInjectedBias()throws {
        var doc=fixture()
        for index in doc.nodes.indices {
            if case .modelPrimitive(let type,_,var config)=doc.nodes[index].payload {
                if type == "linear" {config.removeValue(forKey:"bias")}
                doc.nodes[index].payload = .modelPrimitive(type:type,configurationVersion:1,config:config)
            }
        }
        let plan=try compile(doc)
        XCTAssertEqual(plan.model.graph.schemaVersion,1)
        XCTAssertTrue(plan.model.graph.nodes.filter{$0.type=="linear"}.allSatisfy{$0.config["bias"]==nil})
        XCTAssertEqual(plan.model.graph.edges,ExampleFlows.parallelClassifier.edges)
        XCTAssertEqual(plan.model.graph.nodes.map(\.id),ExampleFlows.parallelClassifier.nodes.map(\.id))
    }
    func testNestedBlockUsesExistingExpansionAndRetainsDefinitionPins()throws {
        let doc=fixture(block:true),plan=try compile(doc)
        var source=CompositeExamples.parallelClassifier;source.name=doc.name;source.execution=plan.settings
        let expected=try ModelExecutionSnapshotBuilder.build(.blockModel(.init(projectID:doc.projectID,revision:doc.revision,source:source)),registry:.builtIn())
        XCTAssertEqual(plan.model.graph,expected.graph);XCTAssertEqual(plan.model.origins,expected.origins)
        XCTAssertEqual(plan.model.provenance?.consumedDefinitions,expected.provenance?.consumedDefinitions)
        XCTAssertEqual(plan.model.semanticSHA256,expected.semanticSHA256)
    }
    private func expect(_ document:WorkflowDocument,_ code:WorkflowCompilerError.Code,file:StaticString=#filePath,line:UInt=#line) {
        XCTAssertThrowsError(try compile(document),file:file,line:line) {error in
            XCTAssertEqual((error as? WorkflowCompilerError)?.code,code,error.localizedDescription,file:file,line:line)
        }
    }
    func testLeakageAndFalsePortKindsAreRejectedBeforeAnyDataExists()throws {
        var doc=fixture();_ = try compile(doc)
        let fit=try XCTUnwrap(doc.edges.firstIndex {$0.to.address==address(.data,"fit")})
        doc.edges[fit].from.port="validation"
        expect(doc,.lineage)
        doc=fixture();let state=try XCTUnwrap(doc.edges.firstIndex {$0.to.address==address(.data,"training") && $0.to.port=="state"})
        doc.edges[state].kind = .table;expect(doc,.portKind)
        doc=fixture();let mapped=try XCTUnwrap(doc.edges.firstIndex {$0.to.address==address(.control,"mapping") && $0.to.port=="training"})
        doc.edges[mapped].from.address=address(.data,"validation");expect(doc,.lineage)
    }
    func testMissingRequiredCycleUnknownConfigurationAndIncompatibleStateChainFail()throws {
        var doc=fixture();_ = try compile(doc)
        doc.edges.removeAll {$0.to.address==address(.control,"train") && $0.to.port=="configuration"}
        expect(doc,.requiredInput)
        doc=fixture()
        doc.edges.removeAll {$0.to.address==address(.data,"cast") && $0.to.port=="table"}
        doc.edges.append(edge(.data,"training","table",.data,"cast","table",.table))
        expect(doc,.cycle)
        doc=fixture();let index=try XCTUnwrap(doc.nodes.firstIndex{$0.address==address(.data,"cast")})
        doc.nodes[index].payload = .dataOperator(operatorID:"kwip.data.cast",config:["columns":.object(["a":.string("float64")]),"unknown":.bool(true)])
        expect(doc,.configuration)
        doc=fixture();doc.nodes[index].payload = .dataOperator(operatorID:"kwip.data.unknown",config:[:])
        expect(doc,.unsupportedOperator)
        doc=fixture();let setup=try XCTUnwrap(doc.nodes.firstIndex{$0.address==address(.control,"setup")})
        doc.nodes[setup].payload = .runSetup(config:["epochs":.number(3.5)])
        expect(doc,.configuration)
        doc=fixture();doc.edges.removeAll {$0.to.address==address(.control,"mapping") && $0.to.port=="validation"}
        doc.edges.append(edge(.data,"split","validation",.control,"mapping","validation",.dataset))
        expect(doc,.lineage)
    }
    func testUnresolvedDraftAndMixedRootConfigurationVersionsVetoCompilation()throws {
        var doc=fixture(block:true)
        let definition=try XCTUnwrap(doc.definitions.first)
        doc.drafts=[.init(projectID:doc.projectID,instancePath:["features"],baseReference:definition.reference,baseRevision:doc.revision,proposed:definition)]
        expect(doc,.unresolvedDraft)
        doc=fixture();let index=try XCTUnwrap(doc.nodes.firstIndex{$0.address==address(.model,"input")})
        if case .modelPrimitive(let type,_,let config)=doc.nodes[index].payload {doc.nodes[index].payload = .modelPrimitive(type:type,configurationVersion:1,config:config)}
        expect(doc,.mixedModelVersions)
    }
    func testPreparationOutputBranchIsIncludedInTrainingPlan()throws {
        var doc=fixture()
        let sink=address(.data,"save")
        doc.nodes.append(.init(address:sink,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string("training.csv"),"separator":.string(","),"columns":.array([]),"overwrite":.bool(false)])))
        doc.edges.append(edge(.data,"training","table",.data,"save","table",.table))
        let plan=try compile(doc)
        XCTAssertTrue(plan.dataflow.nodes.contains {$0.id=="save"})
        XCTAssertTrue(plan.dataflow.edges.contains {$0.from.node=="training" && $0.to.node=="save"})
        XCTAssertEqual(plan.sourceAddresses,[address(.data,"input")])
    }
    func testInspectionViewsAreIncludedOnlyByNewCapturedCompilerContract()throws {
        let document=try ModelOutputPracticeExample.document(device:"cpu")
        let registry=try capturedDataRegistry()
        let old=try WorkflowCompiler.compileHistoricalEvidence(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:15)
        XCTAssertFalse(old.dataflow.nodes.contains {$0.id=="training" || $0.id=="validation"})
        let current=try WorkflowCompiler.compileHistoricalEvidence(document,train:WorkflowExamples.trainAddress,modelRegistry:.builtIn(),dataRegistry:registry,contractVersion:16)
        XCTAssertTrue(current.dataflow.nodes.contains {$0.id=="training"})
        XCTAssertTrue(current.dataflow.nodes.contains {$0.id=="validation"})
        XCTAssertEqual(current.training,old.training)
        XCTAssertEqual(current.validation,old.validation)
        XCTAssertEqual(current.model,old.model)
    }
    func testUnrelatedOutputCannotBeSilentlySkippedByTraining()throws {
        var doc=fixture()
        doc.nodes.append(.init(address:address(.data,"other"),payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])))
        doc.nodes.append(.init(address:address(.data,"save"),payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string("out.csv"),"separator":.string(","),"columns":.array([]),"overwrite":.bool(false)])))
        doc.edges.append(edge(.data,"other","table",.data,"save","table",.table))
        XCTAssertThrowsError(try compile(doc)) {error in XCTAssertTrue(error.localizedDescription.contains("Unrelated outputs"))}
    }

}
