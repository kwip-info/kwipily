import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DerivativeResultsTests:XCTestCase {
    private static var generatedFixture:URL?
    override class func tearDown() {
        if let url=generatedFixture {try? FileManager.default.removeItem(at:url)}
        generatedFixture=nil;super.tearDown()
    }
    private func generateFixture()throws->URL {
        if let existing=Self.generatedFixture {return existing}
        let root=try XCTUnwrap(ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],"Declare the genuine host development root for this actual-worker test.")
        let base=FileManager.default.temporaryDirectory.appendingPathComponent("derivative-results-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:base,withIntermediateDirectories:true)
        let resolved=base.resolvingSymlinksInPath()
        let process=Process();process.executableURL=URL(fileURLWithPath:root).appendingPathComponent(".venv/bin/python")
        process.currentDirectoryURL=URL(fileURLWithPath:root)
        process.arguments=["-c",Self.fixtureScript,resolved.path]
        let output=Pipe();process.standardOutput=output;process.standardError=output
        try process.run();let log=output.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        guard process.terminationStatus==0 else {throw WorkflowPhaseError("Actual derivative fixture failed: "+String(decoding:log,as:UTF8.self))}
        Self.generatedFixture=resolved;return resolved
    }
    private static let fixtureScript = #"""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from runner import engine
from runner.tests.test_loss_engine import prepared
from runner.tests.test_gradient_engine import policy,optimizer
from runner.tests.test_requested_gradients import request
from runner.tests.test_trainability import recipe as flags,owner
base=Path(sys.argv[1])
graph,r,m,data=prepared(base/'source','loss_mse')
q=request('sum',output={'node':'output','port':'out'},targets=[{'kind':'input','node':'input','port':'out'},{'kind':'parameter','node':'learn','parameter':'weight'}],unavailable='reportUnavailable')
engine.run(graph,base/'run',epochs=1,seed=41,batch_size=5,prepared=data,device='cpu',objective_recipe=r,target_mapping=m,optimizer_recipe=optimizer(),gradient_policy=policy(requests=[q]),trainability=flags(owner('learn','weight')),derivative_capture={'schemaVersion':1,'values':True})
manifest=json.loads((base/'run/manifest.json').read_text())
context={'runID':manifest['runID'],'torchVersion':manifest['runtime']['torch'],'device':'cpu','policy':manifest['settings']['gradientPolicyRecipe'],'rowIDs':{'training':manifest['provenance']['trainingRowIDs'],'validation':manifest['provenance']['validationRowIDs']},'preparationManifestPath':str((base/'source/preparation/manifest.json').resolve()),'preparationGraphDigest':manifest['provenance']['preparation']['graphDigest']}
(base/'context.json').write_text(json.dumps(context))
"""#
    private func fixture()throws->(Data,JSONValue,DerivativeResultsContext) {
        let base=try generateFixture()
        let value=try JSONDecoder().decode(JSONValue.self,from:Data(contentsOf:base.appendingPathComponent("context.json")))
        guard case .object(let c)=value,case .object(let rows)=c["rowIDs"],case .object(let recipe)=c["policy"],case .object(let config)=recipe["config"],case .array(let requests)=config["requests"],case .object(let request)=requests[0],case .array(let targets)=request["targets"] else {throw WorkflowPhaseError("Actual engine fixture is incomplete.")}
        var ids:[String:[String]]=[:]
        for (key,value) in rows {guard case .array(let a)=value else {throw WorkflowPhaseError("Rows")};ids[key]=try a.map {guard let s=$0.stringValue else {throw WorkflowPhaseError("Row ID")};return s}}
        let context=DerivativeResultsContext(policy:try GradientPolicyConfiguration.decodeRecipe(c["policy"]!),capture:.init(values:true),runID:c["runID"]!.stringValue!,device:.cpu,torchVersion:c["torchVersion"]!.stringValue!,graph:try Data(contentsOf:base.appendingPathComponent("run/graph.json")),checkpoint:try Data(contentsOf:base.appendingPathComponent("run/checkpoint.pt")),preparationManifest:try Data(contentsOf:URL(fileURLWithPath:c["preparationManifestPath"]!.stringValue!)),preparationGraphDigest:c["preparationGraphDigest"]!.stringValue!,partitionRowIDs:ids,geometry:[.init(requestID:"effect",outputShape:[2,2],targets:[.init(target:targets[0],name:"input.out",shape:[2,2],dtype:.float32,requiresGrad:true),.init(target:targets[1],name:"layers.learn.weight",shape:[2,2],dtype:.float32,requiresGrad:false)])])
        let manifest=try JSONDecoder().decode(JSONValue.self,from:Data(contentsOf:base.appendingPathComponent("run/manifest.json")))
        guard case .object(let m)=manifest else {throw WorkflowPhaseError("Manifest")}
        return (try Data(contentsOf:base.appendingPathComponent("run/derivative-results.json")),m["derivativeResults"]!,context)
    }
    private func repin(_ data:Data,metadata:JSONValue)throws->JSONValue {
        guard case .object(var m)=metadata else {throw WorkflowPhaseError("Metadata")}
        m["bytes"] = .integer(Int64(data.count));m["sha256"] = .string(WorkflowPhaseArtifacts.hash(data));return .object(m)
    }
    private func changed(_ data:Data,_ mutate:(inout [String:Any])->Void)throws->Data {
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any]);mutate(&object)
        return try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys,.withoutEscapingSlashes])
    }
    func testActualEngineObservedAndFrozenResultsAreTyped()throws {
        let (data,metadata,c)=try fixture();let result=try DerivativeResults.verify(data,metadata:metadata,context:c)
        XCTAssertEqual(result.runID,c.runID);XCTAssertEqual(result.requests.count,1)
        XCTAssertEqual(result.requests[0].targets[0].status,.observed);XCTAssertEqual(result.requests[0].targets[0].values?.count,4)
        XCTAssertEqual(result.requests[0].targets[1].status,.unavailable);XCTAssertNil(result.requests[0].targets[1].values)
        XCTAssertEqual(result.requests[0].targets[1].reportedCause,"frozenParameter")
    }
    func testIndependentPermissionAndActualSourceBytesAreRequired()throws {
        let (data,metadata,c)=try fixture()
        func replacing(capture:DerivativeCapturePolicy,graph:Data)->DerivativeResultsContext {
            .init(policy:c.policy,capture:capture,runID:c.runID,device:c.device,torchVersion:c.torchVersion,graph:graph,checkpoint:c.checkpoint,preparationManifest:c.preparationManifest,preparationGraphDigest:c.preparationGraphDigest,partitionRowIDs:c.partitionRowIDs,geometry:c.geometry)
        }
        XCTAssertThrowsError(try DerivativeResults.verify(data,metadata:metadata,context:replacing(capture:.factory,graph:c.graph)))
        XCTAssertThrowsError(try DerivativeResults.verify(data,metadata:metadata,context:replacing(capture:c.capture,graph:c.graph+Data(" ".utf8))))
        let oversized=Data(repeating:32,count:8_388_609)
        XCTAssertThrowsError(try DerivativeResults.verify(oversized,metadata:repin(oversized,metadata:metadata),context:c))
    }
    func testTamperedBytesAndForeignMetadataRefuse()throws {
        let (data,metadata,c)=try fixture()
        XCTAssertThrowsError(try DerivativeResults.verify(data+Data(" ".utf8),metadata:metadata,context:c))
        guard case .object(var m)=metadata else {return XCTFail()};m["foreign"] = .bool(true)
        XCTAssertThrowsError(try DerivativeResults.verify(data,metadata:.object(m),context:c))
    }
    func testStaleIdentityAndUnrequestedEvidenceRefuse()throws {
        let (data,metadata,c)=try fixture()
        for key in ["runID","device","torchVersion","phase","dtype"] {
            let bad=try changed(data){$0[key]="foreign"};XCTAssertThrowsError(try DerivativeResults.verify(bad,metadata:repin(bad,metadata:metadata),context:c),key)
        }
        let extra=try changed(data){$0["foreign"]=true};XCTAssertThrowsError(try DerivativeResults.verify(extra,metadata:repin(extra,metadata:metadata),context:c))
        let source=try changed(data){var s=$0["source"] as! [String:Any];s["checkpointSHA256"]=String(repeating:"a",count:64);$0["source"]=s};XCTAssertThrowsError(try DerivativeResults.verify(source,metadata:repin(source,metadata:metadata),context:c))
    }
    func testRequestedRowsOrderGeometryAndStatusesAreExact()throws {
        let (data,metadata,c)=try fixture()
        for kind in 0..<8 {
            let bad=try changed(data){o in
                var requests=o["requests"] as! [[String:Any]];var q=requests[0]
                if kind==0 {var s=q["sample"] as! [String:Any];s["rowIDsSHA256"]=String(repeating:"b",count:64);q["sample"]=s}
                if kind==1 {var s=q["sample"] as! [String:Any];s["rowStart"]=1;q["sample"]=s}
                if kind==2 {var out=q["output"] as! [String:Any];out["shape"]=[2,3];q["output"]=out}
                var targets=q["targets"] as! [[String:Any]]
                if kind==3 {targets.reverse()}
                if kind==4 {targets[0]["values"]=[1,2,3]}
                if kind==5 {targets[1]["cause"]="recordingDisabled"}
                if kind==6 {targets[1]["status"]="observed";targets[1]["values"]=[0,0,0,0];targets[1].removeValue(forKey:"cause")}
                if kind==7 {targets[0]["values"]=[true,0,0,0]}
                q["targets"]=targets;requests[0]=q;o["requests"]=requests
            }
            XCTAssertThrowsError(try DerivativeResults.verify(bad,metadata:repin(bad,metadata:metadata),context:c),"mutation \(kind)")
        }
    }
    func testMeasuredZerosRemainObservedRatherThanUnavailable()throws {
        let (data,metadata,c)=try fixture()
        let zero=try changed(data){o in var q=o["requests"] as! [[String:Any]];var t=q[0]["targets"] as! [[String:Any]];t[0]["values"]=[0,0,0,0];q[0]["targets"]=t;o["requests"]=q}
        let result=try DerivativeResults.verify(zero,metadata:repin(zero,metadata:metadata),context:c)
        XCTAssertEqual(result.requests[0].targets[0].status,.observed);XCTAssertEqual(result.requests[0].targets[0].values,[0,0,0,0])
    }
}
