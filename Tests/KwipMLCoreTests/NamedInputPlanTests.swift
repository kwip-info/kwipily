import XCTest
@testable import KwipMLCore

final class NamedInputPlanTests:XCTestCase {
    func testNamedNumericTokenRoundTripAndFutureAdapterRefusal()throws {
        let inputs:[NamedInputPlan.Tensor]=[
            .init(name:"measurements",nodeID:"numbers",sampleShape:[2],columns:["a","b"]),
            .init(name:"tokens",nodeID:"words",representation:"tokens",dtype:"int64",sampleShape:[128],columns:["tokens"],lengthPolicy:"pad",paddingValue:0,vocabularySize:256)]
        let p=NamedInputPlan(inputs:inputs,roles:[.init(name:"target",dtype:"int64",sampleShape:[],columns:["label"])])
        try p.validate();XCTAssertEqual(try NamedInputPlan(values:p.values),p)
        var future=p;future.inputs[0].representation="image"
        XCTAssertThrowsError(try future.validate());XCTAssertNoThrow(try future.validate(executable:false))
        var duplicate=p;duplicate.inputs.append(inputs[0]);XCTAssertThrowsError(try duplicate.validate())
        var overflow=p;overflow.inputs[0].sampleShape=Array(repeating:4096,count:7);XCTAssertThrowsError(try overflow.validate())
        var padding=p;padding.inputs[1].paddingValue=256;XCTAssertThrowsError(try padding.validate())
    }
}

extension NamedInputPlanTests {
    func testTextPracticeConditionalPortsAndActualTrainingCompile()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),models=try ComponentRegistry.builtIn()
        let plain=try TextEncodingPracticeExample.document()
        let node=try XCTUnwrap(plain.nodes.first {if case .dataOperator(let id,_ )=$0.payload {return id==TextEncodingConfiguration.operatorID};return false})
        let supplied=try WorkflowPortContracts.resolve(node,document:plain,modelRegistry:models,dataRegistry:registry)
        XCTAssertEqual(supplied.inputs.map(\.id),["table"])
        XCTAssertTrue(supplied.outputs.contains {$0.id=="vocabulary"})
        let trained=try TextEncodingPracticeExample.trainingDocument()
        XCTAssertEqual(trained.schemaVersion,19)
        let plan=try WorkflowCompiler.compile(trained,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:registry,contractVersion:15)
        XCTAssertEqual(plan.namedInputPlan?.inputs.first?.name,"tokens")
        XCTAssertEqual(plan.model.graph.namedInputs,["tokens":"input"])
        XCTAssertEqual(plan.namedInputPlan?.inputs.first?.sampleShape,[5])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(trained)),trained)
        let code=try WorkflowCodePlan.compile(trained,scope:.workflow,modelRegistry:models,dataRegistry:registry)
        XCTAssertFalse(code.trainings.isEmpty)
    }
}


extension NamedInputPlanTests {
    func testNamedWorkflowRunSettingsRemainEditableWithoutChangingMapping()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json")),models=try ComponentRegistry.builtIn()
        var document=try TextEncodingPracticeExample.trainingDocument()
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address==WorkflowExamples.trainAddress})
        guard case .train(let config)=document.nodes[index].payload else {return XCTFail("Missing Train")}
        let before=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:registry,contractVersion:15)
        var settings=try XCTUnwrap(WorkflowTrainRunSetup.configuration(config,schemaVersion:document.schemaVersion))
        settings["epochs"]=4
        let changed=try WorkflowTrainRunSetup.setting(settings,in:config,schemaVersion:document.schemaVersion)
        XCTAssertEqual(changed["dataMapping"],config["dataMapping"])
        document.nodes[index].payload = .train(config:changed)
        let after=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:models,dataRegistry:registry,contractVersion:15)
        XCTAssertEqual(after.settings.epochs,4)
        XCTAssertEqual(after.namedInputPlan,before.namedInputPlan)
        var beforeArchitecture=before.model.graph,afterArchitecture=after.model.graph
        beforeArchitecture.execution=nil;afterArchitecture.execution=nil
        XCTAssertEqual(afterArchitecture,beforeArchitecture)
    }
}
