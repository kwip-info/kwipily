import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GradientLifecycleVerifierTests:XCTestCase {
    private static var generated:URL?
    override class func tearDown() {if let base=generated {try? FileManager.default.removeItem(at:base)};generated=nil;super.tearDown()}
    private func fixture(_ name:String="parent")throws->([String:JSONValue],GraphDocument,ComponentRegistry,URL) {
        let root=try XCTUnwrap(ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"])
        if Self.generated == nil {
            let base=FileManager.default.temporaryDirectory.appendingPathComponent("gradient-lifecycle-"+UUID().uuidString)
            try FileManager.default.createDirectory(at:base,withIntermediateDirectories:true)
            let actual=base.resolvingSymlinksInPath(),process=Process(),output=Pipe()
            process.executableURL=URL(fileURLWithPath:root).appendingPathComponent(".venv/bin/python");process.currentDirectoryURL=URL(fileURLWithPath:root)
            process.arguments=["-c",Self.script,actual.path];process.standardOutput=output;process.standardError=output
            try process.run();let bytes=output.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
            guard process.terminationStatus == 0 else {throw WorkflowPhaseError(String(decoding:bytes,as:UTF8.self))}
            Self.generated=actual
        }
        let folder=try XCTUnwrap(Self.generated).appendingPathComponent(name)
        let m=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:folder.appendingPathComponent("manifest.json")))
        let g=try JSONDecoder().decode(GraphDocument.self,from:Data(contentsOf:folder.appendingPathComponent("graph.json")))
        let registry=try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:URL(fileURLWithPath:root).appendingPathComponent("Sources/KwipMLCore/Resources/components.json")))
        return(m,g,registry,folder)
    }
    private static let script = #"""
import sys
from pathlib import Path
from runner import engine
from runner.tests.test_loss_engine import model
from runner.tests.test_loss_prepared_data import fixture
from runner import dataflow
from runner.dataflow_cli import run_dataflow
from runner.loss_prepared_data import load_prepared_objective
import hashlib,json
from runner.tests.test_gradient_engine import policy,optimizer
from runner.tests.test_trainability import recipe as flags,owner
from runner.scheduler_configuration import canonical_configuration
base=Path(sys.argv[1]);source=base/'source';source.mkdir()
# Neutral lifecycle population:12training/4validation/4reserved; batch5 yields3
# microbatches and accumulation2 yields2updates. Keep this oracle independent
# of public lesson population changes and another suite's preparation helper.
table,m,r,shape=fixture('loss_mse');columns=[*table['columns'],dict(name='fixture_order',dtype='int64',nullable=False)]
rows=[[v+i*.1 if j<len(m['inputColumns']) else v for j,v in enumerate(table['rows'][i%4])]+[i] for i in range(20)]
csv=source/'source.csv';csv.write_text(','.join(c['name'] for c in columns)+'\n'+''.join(','.join(str(v) for v in row)+'\n' for row in rows))
flow=dict(schemaVersion=1,nodes=[
 dict(id='s',operator='kwip.data.sourceCSV',config={'separator':','}),
 dict(id='c',operator='kwip.data.columns',config={'rows':[dict(source=c['name'],included=True,outputName=c['name'],targetType=c['dtype']) for c in columns],'castFailure':'error','trimText':False,'newColumns':'drop','missingPolicy':'error'}),
 dict(id='p',operator='kwip.data.partition',config={'strategy':'time','timeColumn':'fixture_order','timeEncoding':'epochMilliseconds','trainingEnd':11,'validationEnd':15,'embargoMilliseconds':0,'entityColumns':[],'entityPolicy':'allowAcrossTime'})],
 edges=[{'from':{'node':'s','port':'table'},'to':{'node':'c','port':'table'}},{'from':{'node':'c','port':'table'},'to':{'node':'p','port':'table'}}])
path=source/'dataflow.json';path.write_text(json.dumps(flow));prepared=source/'preparation'
run_dataflow(path,{'s':str(csv)},prepared,lambda _:None)
raw=(prepared/'manifest.json').read_bytes()
data=load_prepared_objective(prepared/'manifest.json','p','training','p','validation',m,r,expected_manifest_sha256=hashlib.sha256(raw).hexdigest(),expected_graph_digest=dataflow.digest(flow),sample_shape=[len(m['inputColumns'])],prediction_shape=shape)
assert len(data[0][0])==12 and len(data[0][2])==4
graph=model('loss_mse')
# Supply real Core document presentation fields before the engine captures the graph.
graph['name']='Gradient lifecycle fixture'
for node in graph['nodes']:node.update(name='',x=0.0,y=0.0)
schedule=dict(schemaVersion=1,configurationVersion=1,componentID='scheduler_one_cycle_lr',config=canonical_configuration('scheduler_one_cycle_lr',dict(planned_total_batches=8,max_lr=.03)))
kw=dict(seed=41,batch_size=5,device='cpu',prepared=data,objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy(),trainability=flags(owner('learn','weight')),scheduler_recipe=schedule)
engine.run(graph,base/'parent',epochs=2,**kw)
engine.run(graph,base/'resume',epochs=4,resume_checkpoint=base/'parent/checkpoint.pt',**kw)
engine.run(graph,base/'child',epochs=1,finetune_checkpoint=base/'parent/checkpoint.pt',**{**kw,'gradient_policy':None,'trainability':None,'scheduler_recipe':None})
engine.run(graph,base/'legacy',epochs=1,**{**kw,'gradient_policy':None,'trainability':None,'scheduler_recipe':None})
implicit={**kw,'optimizer_recipe':None,'trainability':None,'scheduler_recipe':None}
engine.run(graph,base/'implicitParent',epochs=1,**implicit)
engine.run(graph,base/'implicitResume',epochs=2,resume_checkpoint=base/'implicitParent/checkpoint.pt',**implicit)
"""#
    private func object(_ raw:JSONValue?)throws->[String:JSONValue] {guard case .object(let o)=raw else {throw WorkflowPhaseError("Fixture object")};return o}
    private func identity(_ m:[String:JSONValue],_ g:GraphDocument,_ r:ComponentRegistry,selected:Bool=true)throws->WorkflowGradientPolicyIdentity {
        let settings=try object(m["settings"])
        return try WorkflowGradientPolicyVerifier.validate(manifest:m,gradientPolicy:selected ? GradientPolicyConfiguration.decodeRecipe(settings["gradientPolicyRecipe"]!):nil,trainability:selected ? settings["trainabilityRecipe"].map {try TrainabilityConfiguration.decodeRecipe($0)}:nil,derivativeCapture:selected ? .factory:nil,graph:g,trainingRows:12,batchSize:5,context:.explicitObjective,capturedRegistry:r)
    }
    func testActualNativeOwnershipFrozenSubsetAndWindowCadence()throws {
        let(m,g,r,_)=try fixture();let value=try identity(m,g,r)
        XCTAssertEqual(value.microbatchesPerEpoch,3);XCTAssertEqual(value.optimizerUpdatesPerEpoch,2)
        XCTAssertEqual(value.fullOwnership,[.object(["node":.string("learn"),"parameter":.string("weight"),"name":.string("layers.learn.weight"),"shape":.array([2,2]),"dtype":.string("float32"),"requiresGrad":.bool(false)]),.object(["node":.string("learn"),"parameter":.string("bias"),"name":.string("layers.learn.bias"),"shape":.array([2]),"dtype":.string("float32"),"requiresGrad":.bool(true)])])
        XCTAssertEqual(value.optimizerOwnership,[.object(["name":.string("layers.learn.bias"),"shape":.array([2]),"dtype":.string("float32")])])
    }
    func testActualAbsentPolicyRetainsAllLearnedOwners()throws {
        let(m,g,r,_)=try fixture("legacy");let value=try identity(m,g,r,selected:false)
        XCTAssertEqual(value.microbatchesPerEpoch,3);XCTAssertEqual(value.optimizerUpdatesPerEpoch,3);XCTAssertEqual(value.optimizerOwnership.count,2)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(m,mode:.fresh,targetEpochs:1,parent:nil,graph:g,hasObjective:true))
    }
    func testSelectedIdentityRejectsUnrequestedMetadataAndPermission()throws {
        let(m,g,r,_)=try fixture()
        XCTAssertThrowsError(try identity(m,g,r,selected:false))
        let settings=try object(m["settings"]),p=try GradientPolicyConfiguration.decodeRecipe(settings["gradientPolicyRecipe"]!),t=try TrainabilityConfiguration.decodeRecipe(settings["trainabilityRecipe"]!)
        XCTAssertThrowsError(try WorkflowGradientPolicyVerifier.validate(manifest:m,gradientPolicy:p,trainability:t,derivativeCapture:nil,graph:g,trainingRows:12,batchSize:5,context:.explicitObjective,capturedRegistry:r))
    }
    func testNativeOwnershipAndCadenceMutationsRefuse()throws {
        let(m,g,r,_)=try fixture()
        for mutation in ["shape","dtype","flag","order","names","schedule","runtime","recipe","unknown"] {
            var bad=m;var checkpoint=try object(bad["checkpoint"]),contract=try object(checkpoint["contract"])
            var t=try object(contract["trainability"]),p=try object(contract["gradientPolicy"])
            if mutation == "names" {t["optimizerParameterNames"] = .array([.string("layers.learn.weight")])}
            if ["shape","dtype","flag","order"].contains(mutation),case .array(var rows)=t["parameterOwnership"] {
                if mutation == "order" {rows.reverse()} else {var first=try object(rows[0]);first[mutation == "flag" ? "requiresGrad":mutation] = mutation == "shape" ? .array([3,2]):mutation == "dtype" ? .string("int64"):.bool(true);rows[0] = .object(first)}
                t["parameterOwnership"] = .array(rows)
            }
            if mutation == "schedule" {var s=try object(p["schedule"]);s["optimizerUpdatesPerEpoch"] = 3;p["schedule"] = .object(s)}
            if mutation == "unknown" {p["foreign"] = .bool(true)}
            if mutation == "recipe" {p["recipe"] = try GradientPolicyConfiguration(values:["accumulation_steps":1]).recipeValue}
            if mutation == "runtime" {var runtime=try object(bad["runtime"]);runtime["updateCadence"] = .object([:]);bad["runtime"] = .object(runtime)}
            contract["trainability"] = .object(t);contract["gradientPolicy"] = .object(p);checkpoint["contract"] = .object(contract);bad["checkpoint"] = .object(checkpoint)
            XCTAssertThrowsError(try identity(bad,g,r),mutation)
        }
    }
    func testActualFreshResumeAndFineTuneRetainParentPolicyIdentity()throws {
        let(m,g,r,parentFolder)=try fixture();let selected=try identity(m,g,r)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(m,mode:.fresh,targetEpochs:2,parent:nil,graph:g,hasScheduler:true,hasObjective:true,hasGradientPolicy:true,hasTrainability:true,gradientIdentity:selected))
        var files:[String:Data]=[:]
        for name in ["manifest.json","checkpoint.pt","metrics.json","rate-trace.json"] {files[name]=try Data(contentsOf:parentFolder.appendingPathComponent(name))}
        let parent=try WorkflowLifecycleParentRecord.decode(files)
        XCTAssertTrue(parent.evidence.hasScheduler);XCTAssertTrue(parent.evidence.hasObjective)
        XCTAssertTrue(parent.evidence.hasGradientPolicy);XCTAssertTrue(parent.evidence.hasTrainability)
        let(resume,rg,rr,resumeFolder)=try fixture("resume");let resumed=try identity(resume,rg,rr)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(resume,mode:.exactResume,targetEpochs:4,parent:parent.evidence,graph:rg,hasScheduler:true,hasObjective:true,hasGradientPolicy:true,hasTrainability:true,gradientIdentity:resumed))
        let settings=try object(resume["settings"])
        let opt=try OptimizerConfiguration.decodeRecipe(settings["optimizerRecipe"]!),schedule=try SchedulerConfiguration.decodeRecipe(settings["schedulerRecipe"]!)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateOptimizer(resume,expected:nil,graph:rg,expectedConfiguration:opt,context:.explicitObjective,capturedRegistry:rr,gradientIdentity:resumed))
        let trace=try JSONDecoder().decode(JSONValue.self,from:Data(contentsOf:resumeFolder.appendingPathComponent("rate-trace.json")))
        let metrics=try JSONDecoder().decode([[String:JSONValue]].self,from:Data(contentsOf:resumeFolder.appendingPathComponent("metrics.json")))
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateScheduler(resume,expected:nil,optimizer:nil,graph:rg,trace:trace,metrics:metrics,expectedConfiguration:schedule,optimizerConfiguration:opt,context:.explicitObjective,capturedRegistry:rr,gradientIdentity:resumed))
        let(child,cg,cr,_)=try fixture("child");_ = try identity(child,cg,cr,selected:false)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(child,mode:.fineTune,targetEpochs:1,parent:parent.evidence,graph:cg,hasObjective:true))
        let changed=WorkflowGradientPolicyIdentity(gradientPolicy:try GradientPolicyConfiguration(values:["accumulation_steps":1]),trainability:resumed.trainability,fullOwnership:resumed.fullOwnership,optimizerOwnership:resumed.optimizerOwnership,microbatchesPerEpoch:3,optimizerUpdatesPerEpoch:3)
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(resume,mode:.exactResume,targetEpochs:4,parent:parent.evidence,graph:rg,hasScheduler:true,hasObjective:true,hasGradientPolicy:true,hasTrainability:true,gradientIdentity:changed))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(resume,mode:.exactResume,targetEpochs:4,parent:parent.evidence,graph:rg,hasScheduler:true,hasObjective:true))
        var foreign=try object(m["provenance"]),bad=m;foreign["trainability"] = .object([:]);bad["provenance"] = .object(foreign)
        files["manifest.json"]=try WorkflowPhaseArtifacts.encode(bad)
        XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(files))
    }
    func testActualGradientOnlyImplicitAdamExactResumeIsSupported()throws {
        let(parent,pg,pr,folder)=try fixture("implicitParent")
        let first=try identity(parent,pg,pr)
        XCTAssertEqual(first.optimizerOwnership.count,2)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(parent,mode:.fresh,targetEpochs:1,parent:nil,graph:pg,hasObjective:true,hasGradientPolicy:true,gradientIdentity:first))
        var contents:[String:Data]=[:]
        for name in ["manifest.json","checkpoint.pt","metrics.json"] {contents[name]=try Data(contentsOf:folder.appendingPathComponent(name))}
        let stored=try WorkflowLifecycleParentRecord.decode(contents)
        XCTAssertTrue(stored.evidence.hasGradientPolicy);XCTAssertFalse(stored.evidence.hasTrainability)
        let(resume,g,r,_)=try fixture("implicitResume"),selected=try identity(resume,g,r)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(resume,mode:.exactResume,targetEpochs:2,parent:stored.evidence,graph:g,hasObjective:true,hasGradientPolicy:true,gradientIdentity:selected))
        let changed=WorkflowGradientPolicyIdentity(gradientPolicy:try GradientPolicyConfiguration(values:["accumulation_steps":1]),trainability:nil,fullOwnership:selected.fullOwnership,optimizerOwnership:selected.optimizerOwnership,microbatchesPerEpoch:3,optimizerUpdatesPerEpoch:3)
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(resume,mode:.exactResume,targetEpochs:2,parent:stored.evidence,graph:g,hasObjective:true,hasGradientPolicy:true,gradientIdentity:changed))
    }

}
