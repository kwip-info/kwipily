import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BagDerivativeResultsTests: XCTestCase {
    private struct Fixture {
        let root:URL, graph:GraphDocument, registry:ComponentRegistry
        let policy:GradientPolicyConfiguration, identity:WorkflowGradientPolicyIdentity
        let preparation:VerifiedBagPreparation, requirement:BagWorkflowInputRequirement
        let manifest:[String:JSONValue], artifact:Data, device:ContractDevice
    }
    private func fixture(layout:String="packed",terminal:Bool=true,frozen:Bool=false,mode:String="inherit",device:String="cpu") throws -> Fixture {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent("bag-derivative-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let process=Process(),pipe=Pipe();process.executableURL=repo.appendingPathComponent(".venv/bin/python");process.currentDirectoryURL=repo
        process.arguments=["-B","-c",Self.script,root.path,layout,terminal ? "yes":"no",frozen ? "yes":"no",mode,device]
        process.standardOutput=pipe;process.standardError=pipe;try process.run();let output=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        guard process.terminationStatus==0 else {throw WorkflowPhaseError(String(decoding:output,as:UTF8.self))}
        func read<T:Decodable>(_ path:String,_ type:T.Type)throws->T {try JSONDecoder().decode(type,from:Data(contentsOf:root.appendingPathComponent(path)))}
        let manifest=try read("run/manifest.json",[String:JSONValue].self),graph=try read("run/graph.json",GraphDocument.self),summary=try read("fixture.json",[String:JSONValue].self)
        let registry=try BagWorkflowArchiveTests().registry()
        let requirement=try WorkflowMappingRequirements.bagInputRequirement(graph,registry:registry)
        let ref=try JSONDecoder().decode(PreparedDataReference.self,from:WorkflowPhaseArtifacts.encode(XCTUnwrap(summary["reference"])))
        let selection=WorkflowPreparedSelection.legacy(.init(manifestPath:ref.manifestPath,manifestSHA256:ref.manifestSHA256,graphDigest:ref.graphDigest,trainingNode:ref.trainingNode,trainingPort:ref.trainingPort,validationNode:ref.validationNode,validationPort:ref.validationPort,featureColumns:["i1","i2"],targetColumn:"label"))
        let mapping=try BagInputMapping(value:XCTUnwrap(summary["mapping"]))
        let preparation=try WorkflowBagPreparedEvidence.preparedEvidence(root:root,selection:selection,mapping:mapping,requirement:requirement)
        guard case .object(let settings)=manifest["settings"] else {throw WorkflowPhaseError("Fixture settings")}
        let policy=try GradientPolicyConfiguration.decodeRecipe(XCTUnwrap(settings["gradientPolicyRecipe"]))
        let flags=try settings["trainabilityRecipe"].map {try TrainabilityConfiguration.decodeRecipe($0)}
        let identity=try WorkflowGradientPolicyVerifier.validate(manifest:manifest,gradientPolicy:policy,trainability:flags,derivativeCapture:.init(values:true),graph:graph,trainingRows:preparation.training.count,batchSize:32,context:.preparedWorkflow,capturedRegistry:registry)
        return .init(root:root,graph:graph,registry:registry,policy:policy,identity:identity,preparation:preparation,requirement:requirement,manifest:manifest,artifact:try Data(contentsOf:root.appendingPathComponent("run/derivative-results.json")),device:device=="mps" ? .mps:.cpu)
    }
    private func context(_ f:Fixture)throws->DerivativeResultsContext {
        let bag=try WorkflowBagDerivativeContext(requirement:f.requirement,preparation:f.preparation)
        let geometry=try WorkflowDerivativeGeometry.resolve(policy:f.policy,graph:f.graph,registry:f.registry,context:.preparedWorkflow,identity:f.identity,bag:bag)
        return try .init(policy:f.policy,capture:.init(values:true),runID:XCTUnwrap(f.manifest["runID"]?.stringValue),device:f.device,torchVersion:runtime(f)["torch"]!.stringValue!,graph:Data(contentsOf:f.root.appendingPathComponent("run/graph.json")),checkpoint:Data(contentsOf:f.root.appendingPathComponent("run/checkpoint.pt")),preparationManifest:Data(contentsOf:URL(fileURLWithPath:f.preparation.selection.manifestPath)),preparationGraphDigest:f.preparation.selection.graphDigest,partitionRowIDs:["training":f.preparation.training.rowIDs,"validation":f.preparation.validation.rowIDs],geometry:geometry,bag:bag)
    }
    private func runtime(_ f:Fixture)throws->[String:JSONValue] {guard case .object(let runtime)=f.manifest["runtime"] else {throw WorkflowPhaseError("Fixture runtime")};return runtime}
    private func verify(_ f:Fixture)throws->DerivativeResults {try DerivativeResults.verify(f.artifact,metadata:XCTUnwrap(f.manifest["derivativeResults"]),context:context(f))}
    func testActualPreparedCountsBeyondPreviewAndPacketDerivatives() throws {
        let f=try fixture();defer {try? FileManager.default.removeItem(at:f.root)}
        XCTAssertEqual(f.preparation.training.count,240);XCTAssertEqual(f.preparation.training.prefix.count,200)
        XCTAssertEqual(f.preparation.training.itemCounts.count,240);XCTAssertEqual(f.preparation.validation.itemCounts.count,80)
        let result=try verify(f);XCTAssertEqual(result.requests.count,2)
        XCTAssertEqual(result.requests[0].targets[1].shape,[3])
        XCTAssertEqual(result.requests[1].targets[2].shape,[0]);XCTAssertEqual(result.requests[1].targets[2].values,[])
        XCTAssertEqual(result.requests[1].targets[0].reportedCause,"integerInput")
    }
    func testFixedAndNoTerminalGeometryRetainsActualCoordinateShapes() throws {
        for (layout,terminal) in [("fixed",false),("packed",false)] {
            let f=try fixture(layout:layout,terminal:terminal);defer {try? FileManager.default.removeItem(at:f.root)}
            let result=try verify(f)
            XCTAssertEqual(result.requests[0].targets[0].shape,layout=="fixed" ? [2,2]:[f.preparation.training.itemCounts[201]+f.preparation.training.itemCounts[202]])
            if layout=="packed" {XCTAssertEqual(result.requests[0].targets[1].shape,[2])}
        }
    }
    func testMPSActualPacketArtifactUsesSameIndependentGeometry() throws {
        let f=try fixture(device:"mps");defer {try? FileManager.default.removeItem(at:f.root)}
        XCTAssertEqual(try verify(f).device,.mps)
    }
    func testFrozenTableFlagsAndExplicitOverridesMatchActualNativeOwners() throws {
        for mode in ["inherit","frozen","trainable"] {
            let f=try fixture(frozen:true,mode:mode);defer {try? FileManager.default.removeItem(at:f.root)}
            guard case .object(let owner)=f.identity.fullOwnership.first(where:{if case .object(let value)=$0 {return value["node"] == .string("bag")};return false}) else {return XCTFail("Missing bag owner")}
            XCTAssertEqual(owner["requiresGrad"],.bool(mode=="trainable"))
            XCTAssertEqual(owner["name"],.string("layers.bag.weight"));XCTAssertEqual(owner["shape"],.array([5,2]))
            let result=try verify(f);XCTAssertEqual(result.requests[0].targets.last?.reportedCause,mode=="trainable" ? nil:"frozenParameter")
        }
    }
    func testRepinnedSchemaSourceCountsAndZeroForgeriesRefuse() throws {
        let f=try fixture();defer {try? FileManager.default.removeItem(at:f.root)};let c=try context(f)
        let original=try JSONDecoder().decode([String:JSONValue].self,from:f.artifact)
        for mutation in ["version","null","modelInput","counts","offsets","parameterZero","outputZero","floatInteger"] {
            var root=original
            if mutation=="version" {root["schemaVersion"]=1}
            else if mutation=="null" {root["modelInput"] = .null}
            else if mutation=="modelInput",case .object(var input)=root["modelInput"] {input["sourceID"] = .string("foreign");root["modelInput"] = .object(input)}
            else if case .array(var requests)=root["requests"],case .object(var row)=requests[0] {
                if mutation=="counts",case .object(var sample)=row["sample"] {
                    var forged=try XCTUnwrap(c.bag).selectedCounts(f.policy.requests[0])
                    forged[0]=(forged[0]+1) % (f.requirement.configuration.maximumItemsPerBag+1)
                    sample["bagItemCounts"] = .array(forged.map { .integer(Int64($0)) });row["sample"] = .object(sample)
                }
                else if mutation=="outputZero",case .object(var output)=row["output"] {output["shape"] = .array([0,2]);row["output"] = .object(output)}
                else if case .array(var targets)=row["targets"],case .object(var target)=targets[mutation=="parameterZero" ? targets.count-1:mutation=="offsets" ? 1:0] {
                    if mutation=="floatInteger" {target["status"] = .string("observed");target.removeValue(forKey:"cause");target["values"] = .array([])} else {target["shape"] = .array([0])}
                    targets[mutation=="parameterZero" ? targets.count-1:mutation=="offsets" ? 1:0] = .object(target);row["targets"] = .array(targets)
                }
                requests[0] = .object(row);root["requests"] = .array(requests)
            }
            let bytes=try WorkflowPhaseArtifacts.encode(root)
            guard case .object(var metadata)=f.manifest["derivativeResults"] else {return XCTFail("Missing metadata")}
            metadata["bytes"] = .integer(Int64(bytes.count));metadata["sha256"] = .string(WorkflowPhaseArtifacts.hash(bytes))
            XCTAssertThrowsError(try DerivativeResults.verify(bytes,metadata:.object(metadata),context:c),mutation)
        }
        var missing=c;missing.bag=nil
        XCTAssertThrowsError(try DerivativeResults.verify(f.artifact,metadata:XCTUnwrap(f.manifest["derivativeResults"]),context:missing))
    }
    private static let script = #"""
import hashlib,json,sys
from pathlib import Path
from runner import engine
from runner.dataflow_cli import run_dataflow
from runner.tests.test_prepared_bag_loader import options,load
from runner.tests.test_bag_source_graph import graph,registry
from runner.tests.test_gradient_engine import policy
from runner.tests.test_trainability import recipe as flags,owner
root=Path(sys.argv[1]);layout,terminal,frozen,mode,device=sys.argv[2:];terminal=terminal=='yes';frozen=frozen=='yes'
# Dedicated population oracle:400rows, balancedlabels;60/20/20 gives240/80/80.
# This derivative fixture must keep trainingrow201 beyond the200row display preview.
source=root/'source';source.mkdir();csv=source/'bags.csv'
csv.write_text('i1,i2,w1,w2,count,label,desired\n'+''.join(f'{i%5},{(i+1)%5},{i+.5},{i+1.5},{i%3},{i%2},{i/8}\n' for i in range(400)))
flow=dict(schemaVersion=1,nodes=[
 dict(id='s',operator='kwip.data.sourceCSV',config={'separator':','}),
 dict(id='c',operator='kwip.data.columns',config={
  'rows':[dict(source=name,included=True,outputName=name,targetType=dtype) for name,dtype in [('i1','int64'),('i2','int64'),('w1','float64'),('w2','float64'),('count','int64'),('label','int64'),('desired','float64')]],
  'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}),
 dict(id='p',operator='kwip.data.partition',config={'strategy':'random','seed':7,'validationPercent':20,'testPercent':20,'stratifyColumn':'label'})],
 edges=[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},
        {'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}])
(source/'flow.json').write_text(json.dumps(flow))
source=root/'preparation/artifacts';run_dataflow(root/'source/flow.json',{'s':str(csv)},source,lambda _:None)
args=options(source);args['source_config'].update(layout=layout,include_last_offset=terminal)
if layout=='fixed':args['mapping'].pop('lengthColumn')
admitted=load(source,source_config=args['source_config'],mapping=args['mapping'])
assert len(admitted[0][1])==240 and len(admitted[0][3])==80, 'Dedicated training/validation population changed'
reg=registry.__wrapped__(root);value=graph(layout,True,terminal,freeze=frozen);value['nodes'][0]['config']['maximum_items_per_bag']=2
if frozen:
 value['nodes'].insert(-1,dict(id='head',type='linear',config=dict(in_features=2,out_features=2,bias=True)));value['edges'][-1]['target']='head';value['edges'].append(dict(id='headout',source='head',sourcePort='out',target='result',targetPort='in'))
value['name']='Actual bag derivative';
for node in value['nodes']:node.update(name='',x=0.,y=0.)
targets=[dict(kind='input',node='source',port=port) for port in ['indices']+(['offsets'] if layout=='packed' else [])+['per_sample_weights']]+[dict(kind='parameter',node='bag',parameter='weight')]
def request(name,partition,start,count):return dict(id=name,output=dict(node='bag',port='out'),partition=partition,row_start=start,row_count=count,reduction='sum',upstream=None,targets=targets,unavailable='reportUnavailable')
requests=[request('beyond-preview','training',201,2)]
if layout=='packed':
 packet=admitted[0][2];lengths=(packet.offsets[1:]-packet.offsets[:-1]).tolist() if terminal else ( __import__('torch').cat((packet.offsets[1:],packet.offsets.new_tensor([packet.indices.numel()])))-packet.offsets).tolist();requests.append(request('empty','validation',lengths.index(0),1))
kw=dict(trainability=flags(owner('bag','weight',mode))) if frozen else {}
engine.run(value,root/'run',registry=reg,prepared=admitted,device=device,epochs=1,batch_size=32,gradient_policy=policy(requests=requests),derivative_capture=dict(schemaVersion=1,values=True),**kw)
raw=(source/'manifest.json').read_bytes();manifest=json.loads(raw)
(root/'fixture.json').write_text(json.dumps(dict(reference=dict(manifestPath=str(source/'manifest.json'),manifestSHA256=hashlib.sha256(raw).hexdigest(),graphDigest=manifest['graphDigest'],trainingNode='p',trainingPort='training',validationNode='p',validationPort='validation'),mapping=args['mapping'])))
"""#
}
