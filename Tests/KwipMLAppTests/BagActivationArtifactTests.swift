import XCTest
import Darwin
@testable import KwipMLCore
@testable import KwipMLApp

final class BagActivationArtifactTests: XCTestCase {
    struct Fixture {
        let root: URL
        let selection: WorkflowPreparedSelection
        let mapping: BagInputMapping
        let requirement: BagWorkflowInputRequirement
        let manifest: [String: JSONValue]
        let predictions: Data
        let target: BagArtifactTarget
        let device: ContractDevice
        let graph: GraphDocument
        let registry: ComponentRegistry
        let event: [String: JSONValue]
        let activations: Data
    }
    func fixture(objective: Bool = false, capture: Bool = true, activation: Bool = true, device: String = "cpu", empty: Bool = false) throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let root = folder.resolvingSymlinksInPath()
        let capturedRegistry=try registry()
        try WorkflowPhaseArtifacts.encode(capturedRegistry).write(to:root.appendingPathComponent("registry.json"))
        let process = Process(), log = Pipe()
        process.executableURL = repo.appendingPathComponent(".venv/bin/python"); process.currentDirectoryURL = repo
        process.arguments = ["-B", "-c", Self.script, root.path, objective ? "yes" : "no", capture ? "yes" : "no", activation ? "yes" : "no", device, empty ? "yes":"no"]
        process.standardOutput = log; process.standardError = log
        try process.run(); let deadline = Date().addingTimeInterval(40)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.025) }
        if process.isRunning { _ = Darwin.kill(process.processIdentifier, SIGKILL) }; process.waitUntilExit()
        let output = String(decoding: log.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, output)
        let manifest = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: root.appendingPathComponent("training/artifacts/manifest.json")))
        let summary = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
        func decode<T: Decodable>(_ type: T.Type, _ value: JSONValue) throws -> T { try JSONDecoder().decode(type, from: WorkflowPhaseArtifacts.encode(value)) }
        let reference = try decode(PreparedDataReference.self, XCTUnwrap(summary["reference"]))
        let mapping = try BagInputMapping(value: XCTUnwrap(summary["mapping"]))
        guard case .object(let config) = summary["sourceConfiguration"] else { throw WorkflowPhaseError("Missing actual source configuration") }
        let requirement = BagWorkflowInputRequirement(sourceNodeID: "source", configuration: try BagSourceConfiguration(values: config), numEmbeddings: 5, embeddingDim: 2)
        let selection: WorkflowPreparedSelection, target: BagArtifactTarget
        if objective {
            let configuration = try LossConfiguration.decodeRecipe(XCTUnwrap(summary["recipe"]))
            let roles = try decode(LossTargetMapping.self, XCTUnwrap(summary["targetMapping"]))
            let selected = PreparedObjectiveSelection(reference: reference, objectiveRecipe: configuration.recipeValue, targetMapping: roles)
            let recipe = try LossRecipe(configuration: configuration, source: .init(domain: .control, localID: "objective"), targetMapping: roles)
            selection = .objective(selected); target = .objective(selected, recipe)
        } else {
            selection = .legacy(.init(manifestPath: reference.manifestPath, manifestSHA256: reference.manifestSHA256, graphDigest: reference.graphDigest, trainingNode: reference.trainingNode, trainingPort: reference.trainingPort, validationNode: reference.validationNode, validationPort: reference.validationPort, featureColumns: mapping.indicesColumns, targetColumn: "label")); target = .binary("label")
        }
        return Fixture(root: root, selection: selection, mapping: mapping, requirement: requirement, manifest: manifest, predictions: try Data(contentsOf: root.appendingPathComponent("training/artifacts/predictions.json")), target: target, device: device == "mps" ? .mps : .cpu, graph:try decode(GraphDocument.self,XCTUnwrap(summary["graph"])),registry:capturedRegistry,event:try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:root.appendingPathComponent("completed.json"))),activations:try Data(contentsOf:root.appendingPathComponent("training/artifacts/activations.json")))
    }
    func context(_ f: Fixture, capture: Bool = true, activation: Bool = true) throws -> WorkflowBagArtifactContext {
        let evidence = try WorkflowBagPreparedEvidence.preparedEvidence(root: f.root, selection: f.selection, mapping: f.mapping, requirement: f.requirement)
        return .init(requirement: f.requirement, mapping: f.mapping, preparation: evidence, runID: try XCTUnwrap(f.manifest["runID"]?.stringValue), device: f.device, outputContract: try .init(sample: [2], dtype: .float32), target: f.target, policy: .init(activationValues: activation, predictionRows: capture), explicitSampleCapture: true)
    }
    private func preview(_ f: Fixture, context c: WorkflowBagArtifactContext) throws -> BagPredictionPreview {
        try WorkflowBagArtifacts.predictions(data: f.predictions, metadata: XCTUnwrap(f.manifest["predictionPreview"]), context: c)
    }
    private func decode(_ f:Fixture,_ c:WorkflowBagArtifactContext)throws->JSONValue {
        try WorkflowActivationArtifact.decode(f.activations,metadata:XCTUnwrap(f.manifest["activationPreview"]),runID:c.runID,graphHash:try XCTUnwrap(f.manifest["graphHash"]?.stringValue),graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c)
    }
    func testActualBinaryCPUAndMPSCoordinateActivationsAndLiveCompletion() throws {
        for device in ["cpu","mps"] {
            let f=try fixture(device:device);defer {try? FileManager.default.removeItem(at:f.root)}
            let c=try context(f)
            _ = try decode(f,c)
            try WorkflowModelTrainingExecution.verifyPresentation(event:f.event,graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c)
        }
    }
    func testActualEmptyPackedCoordinatesAllowOnlyProvenSourceZeroExtent() throws {
        let f=try fixture(empty:true);defer {try? FileManager.default.removeItem(at:f.root)}
        let c=try context(f);let previews=try decode(f,c)
        guard case .object(let nodes)=previews,case .object(let source)=nodes["source"],case .object(let outputs)=source["outputs"],case .object(let ids)=outputs["indices"],case .object(let bag)=nodes["bag"] else {return XCTFail("Actual coordinate preview missing")}
        XCTAssertEqual(ids["shape"],.array([0]));XCTAssertEqual(ids["values"],.array([]))
        XCTAssertEqual(bag["shape"],.array([8,2]))
    }
    func testActualObjectiveCompletionRequiresBoundSharedContextAndIndependentOmission() throws {
        let f=try fixture(objective:true,activation:false);defer {try? FileManager.default.removeItem(at:f.root)}
        let c=try context(f,activation:false),objective=try WorkflowBagArtifacts.objectiveEvaluationContext(context:c,expectedEpochs:1)
        _ = try decode(f,c)
        try WorkflowModelTrainingExecution.verifyPresentation(event:f.event,graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c,objectiveEvaluationContext:objective)
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:f.event,graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c))
        var changed=f.event;changed["accuracy"] = 0.5
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:changed,graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c,objectiveEvaluationContext:objective))
    }
    func testCoordinateMutationsMissingAuthorityAndOrdinaryZeroShapesRefuse() throws {
        let f=try fixture();defer {try? FileManager.default.removeItem(at:f.root)};let c=try context(f)
        let raw=try JSONDecoder().decode([String:JSONValue].self,from:f.activations)
        guard case .object(let original)=raw["previews"] else {return XCTFail("Actual previews absent")}
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(original,graph:f.graph,sampleCapture:c.policy,bagContext:c))
        for field in ["shape","dtype","values"] {
            var nodes=original
            if case .object(var source)=nodes["source"],case .object(var outputs)=source["outputs"],case .object(var ids)=outputs["indices"] {
                ids[field] = .null;outputs["indices"] = .object(ids);source["outputs"] = .object(outputs);nodes["source"] = .object(source)
            }
            XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(nodes,graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c),field)
        }
        var nodes=original
        if case .object(var bag)=nodes["bag"] {bag["shape"] = .array([0,2]);nodes["bag"] = .object(bag)}
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(nodes,graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c))
        var changed=f.event;changed["evaluationCount"] = 7
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:changed,graph:f.graph,sampleCapture:c.policy,capturedRegistry:f.registry,bagContext:c))
    }
    private func registry()throws->ComponentRegistry {
        var registry=try ComponentRegistry.builtIn()
        // Independent descriptor fixtures retain the shared field gate: these are
        // real declared controls, not empty templates that bypass configuration validation.
        func choice(_ key:String,_ value:String,_ options:[String])->ComponentField {
            .init(key:key,title:key,kind:.choice,defaultValue:.string(value),min:nil,max:nil,
                  choices:options.map {.init(value:$0,title:$0)})
        }
        func flag(_ key:String)->ComponentField {
            .init(key:key,title:key,kind:.boolean,defaultValue:false,min:nil,max:nil)
        }
        let sourceFields:[ComponentField]=[
            choice("layout","fixed",["fixed","packed"]),
            .init(key:"maximum_items_per_bag",title:"Maximum items",defaultValue:4,min:1,max:128),
            flag("per_sample_weights"),flag("include_last_offset")]
        let tableFields:[ComponentField]=[
            .init(key:"num_embeddings",title:"Table rows",defaultValue:16,min:1,max:100_000_000),
            .init(key:"embedding_dim",title:"Dimensions",defaultValue:4,min:1,max:4096),
            choice("mode","mean",["mean","sum","max"]),
            .init(key:"padding_idx",title:"Padding",defaultValue:.null,min:-100_000_000,max:99_999_999,allowsNull:true),
            .init(key:"max_norm",title:"Maximum norm",kind:.number,defaultValue:.null,min:1.1754943508222875e-38,max:3.4028234663852886e38,allowsNull:true),
            .init(key:"norm_type",title:"Norm order",kind:.number,defaultValue:2,min:1.1754943508222875e-38,max:3.4028234663852886e38),
            flag("scale_grad_by_freq"),flag("sparse"),flag("include_last_offset"),flag("freeze"),
            choice("layout","fixed",["fixed","packed"]),flag("per_sample_weights")]
        registry.components.removeAll {["bag_input","embedding_bag"].contains($0.id)}
        registry.components += [
            .init(id:"bag_input",title:"Bag input",inputs:[],outputs:["indices","offsets","per_sample_weights"],fields:sourceFields),
            .init(id:"embedding_bag",title:"Embedding bag",inputs:["indices","offsets","per_sample_weights"],outputs:["out"],fields:tableFields)]
        return registry
    }

    private static let script = #"""
import hashlib,json,shutil,sys
from pathlib import Path
from runner import engine
from runner.tests.test_prepared_bag_loader import fixture,options,load
from runner.tests.test_bag_source_graph import graph,registry
from runner.loss_configuration import canonical_configuration
root=Path(sys.argv[1]);objective,capture,activation=[v=='yes' for v in sys.argv[2:5]];device=sys.argv[5]
source=fixture(root/'source')
if sys.argv[6]=='yes':
 from runner.dataflow_cli import run_dataflow
 csv=root/'source/bags.csv';lines=csv.read_text().splitlines();rows=[line.split(',') for line in lines[1:]]
 for row in rows:row[4]='0'
 csv.write_text(lines[0]+'\n'+'\n'.join(','.join(row) for row in rows)+'\n')
 source=root/'empty-prepared';run_dataflow(root/'source/flow.json',{'s':str(csv)},source,lambda _:None)
args=options(source)
recipe=canonical_configuration('loss_mse',{'reduction':'none','target_broadcast':True})
mapping={'schemaVersion':1,'componentID':'loss_mse','inputColumns':['i1','i2'],'roles':[{'role':'target','columns':['desired'],'dtype':'float32','sampleShape':[]}],'tripletSourceGroups':[]}
if objective:
 recipe={'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':recipe}
 admitted=load(source,target_column=None,objective_mapping=mapping,objective_recipe=recipe,prediction_shape=[2])
else:admitted=load(source)
reg=root/'registry.json';value=graph();value['name']='Actual prepared bag activation';value['nodes'][0]['config']['maximum_items_per_bag']=2
for index,node in enumerate(value['nodes']):node.update(name=node['id'],x=float(index*200),y=0.0)
kwargs=dict(objective_recipe=recipe,target_mapping=mapping) if objective else {}
events=[]
engine.run(value,root/'training/artifacts',emit=events.append,registry=reg,prepared=admitted,device=device,epochs=1,batch_size=3,sample_capture={'schemaVersion':1,'predictionRows':capture,'activationValues':activation},**kwargs)
(root/'completed.json').write_text(json.dumps(next(e for e in events if e['event']=='completed')))
shutil.copytree(source,root/'preparation/artifacts')
raw=(source/'manifest.json').read_bytes();manifest=json.loads(raw)
summary={'graph':value,'reference':{'manifestPath':str(root/'preparation/artifacts/manifest.json'),'manifestSHA256':hashlib.sha256(raw).hexdigest(),'graphDigest':manifest['graphDigest'],'trainingNode':'p','trainingPort':'training','validationNode':'p','validationPort':'validation'},'mapping':args['mapping'],'sourceConfiguration':args['source_config'],'recipe':recipe,'targetMapping':mapping}
(root/'fixture.json').write_text(json.dumps(summary))
"""#
}
