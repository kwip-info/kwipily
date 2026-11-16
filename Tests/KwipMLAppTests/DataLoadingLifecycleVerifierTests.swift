import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataLoadingLifecycleVerifierTests:XCTestCase {
    private static var generated:URL?
    override class func tearDown() {if let url=generated {try? FileManager.default.removeItem(at:url)};generated=nil;super.tearDown()}
    private func object(_ raw:JSONValue?)throws->[String:JSONValue] {guard case .object(let value)=raw else {throw WorkflowPhaseError("Missing actual fixture object.")};return value}
    private func fixture(_ name:String)throws->([String:JSONValue],WorkflowFailedGradientEvidence.PreparedRowCounts) {
        let root=URL(fileURLWithPath:try XCTUnwrap(ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"]))
        if Self.generated==nil {
            let folder=FileManager.default.temporaryDirectory.appendingPathComponent("data-loading-lifecycle-"+UUID().uuidString)
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
            let process=Process(),pipe=Pipe();process.executableURL=root.appendingPathComponent(".venv/bin/python");process.currentDirectoryURL=root
            process.arguments=["-c",Self.script,folder.path];process.standardOutput=pipe;process.standardError=pipe
            try process.run();let output=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
            guard process.terminationStatus==0 else {throw WorkflowPhaseError(String(decoding:output,as:UTF8.self))};Self.generated=folder
        }
        let base=try XCTUnwrap(Self.generated)
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:base.appendingPathComponent(name+"/manifest.json")))
        let rows=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:base.appendingPathComponent("rows.json")))
        func ids(_ key:String)throws->[String] {guard case .array(let values)=rows[key] else {throw WorkflowPhaseError("Missing actual IDs")};return try values.map {try XCTUnwrap($0.stringValue)}}
        let training=try ids("training"),validation=try ids("validation")
        return(manifest,.init(trainingRows:training.count,validationRows:validation.count,trainingTableSHA256:try XCTUnwrap(rows["trainingPin"]?.stringValue),validationTableSHA256:try XCTUnwrap(rows["validationPin"]?.stringValue),trainingRowIDs:training,validationRowIDs:validation))
    }
    private static let script = #"""
import sys,json
from pathlib import Path
from runner import engine
from runner.tests.test_dataset_loader_engine import setup
from runner.tests.test_dataset_loader_configuration import recipe
base=Path(sys.argv[1]);graph,kw=setup(base)
selected=dict(batch_size=2,dataset_recipe=recipe(kind='subset',indices=[4,1,0,7,3]),loader_recipe=recipe('data_loader',batch_size=2,shuffle=True))
engine.run(graph,base/'parent',epochs=1,**selected,**kw)
engine.run(graph,base/'resume',epochs=2,resume_checkpoint=base/'parent/checkpoint.pt',**selected,**kw)
changed={**selected,'loader_recipe':recipe('data_loader',batch_size=2,shuffle=True,seed=43)}
engine.run(graph,base/'child',epochs=1,finetune_checkpoint=base/'parent/checkpoint.pt',**changed,**kw)
engine.run(graph,base/'unselectedChild',epochs=1,batch_size=2,finetune_checkpoint=base/'parent/checkpoint.pt',**kw)
p=kw['prepared'][1];inventory=json.loads((base/'source/preparation/manifest.json').read_text())
pins={a['port']:a['sha256'] for a in inventory['artifacts'] if a.get('nodeID')=='p' and a.get('kind')=='table'}
(base/'rows.json').write_text(json.dumps(dict(training=p['trainingRowIDs'],validation=p['validationRowIDs'],trainingPin=pins['training'],validationPin=pins['validation'])))
"""#
    private func identity(_ manifest:[String:JSONValue],_ rows:WorkflowFailedGradientEvidence.PreparedRowCounts)throws->WorkflowDataLoadingIdentity? {
        let settings=try object(manifest["settings"])
        let source=try settings["datasetRecipe"].map {DatasetRecipe(configuration:try DatasetConfiguration.decodeRecipe($0),source:.init(domain:.control,localID:"dataset"))}
        let loader=try settings["loaderRecipe"].map {DataLoaderRecipe(configuration:try DataLoaderConfiguration.decodeRecipe($0),source:.init(domain:.control,localID:"loader"))}
        return try WorkflowDataLoadingVerifier.validate(manifest:manifest,datasetRecipe:source,loaderRecipe:loader,prepared:rows,capturedBatchSize:try XCTUnwrap(settings["requestedBatchSize"]?.integerValue),expectedEpochs:try XCTUnwrap(settings["epochs"]?.integerValue))
    }
    private func parent(_ manifest:[String:JSONValue],identity:WorkflowDataLoadingIdentity?)throws->WorkflowLifecycleParentEvidence {
        let checkpoint=try object(manifest["checkpoint"])
        return .init(workerRunID:try XCTUnwrap(manifest["runID"]?.stringValue),checkpointSHA256:try XCTUnwrap(checkpoint["sha256"]?.stringValue),completedEpoch:1,hasObjective:true,dataLoadingIdentity:identity)
    }
    func testActualSelectedFreshResumeAndFineTunePreserveExactLineageOrder()throws {
        let (first,rows)=try fixture("parent"),selected=try XCTUnwrap(identity(first,rows)),parent=try parent(first,identity:selected)
        XCTAssertTrue(parent.hasDataLoading)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(first,mode:.fresh,targetEpochs:1,parent:nil,hasObjective:true,dataLoadingIdentity:selected))
        let (resumed,r)=try fixture("resume"),resumedIdentity=try XCTUnwrap(identity(resumed,r))
        XCTAssertEqual(resumedIdentity,selected)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(resumed,mode:.exactResume,targetEpochs:2,parent:parent,hasObjective:true,dataLoadingIdentity:resumedIdentity))
        let (child,c)=try fixture("child"),childIdentity=try XCTUnwrap(identity(child,c))
        XCTAssertNotEqual(childIdentity,selected)
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(child,mode:.fineTune,targetEpochs:1,parent:parent,hasObjective:true,dataLoadingIdentity:childIdentity))
        let (unselected,u)=try fixture("unselectedChild")
        XCTAssertNil(try identity(unselected,u))
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validate(unselected,mode:.fineTune,targetEpochs:1,parent:parent,hasObjective:true))
    }
    func testMissingChangedIdentityLineageAndDeviceCannotAcquireResume()throws {
        let (first,rows)=try fixture("parent"),selected=try XCTUnwrap(identity(first,rows)),parent=try parent(first,identity:selected)
        let (resumed,r)=try fixture("resume"),current=try XCTUnwrap(identity(resumed,r))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(resumed,mode:.exactResume,targetEpochs:2,parent:parent,hasObjective:true))
        let (child,c)=try fixture("child"),changed=try XCTUnwrap(identity(child,c))
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(resumed,mode:.exactResume,targetEpochs:2,parent:parent,hasObjective:true,dataLoadingIdentity:changed))
        var forged=resumed,lineage=try object(resumed["lineage"])
        lineage["restoredState"] = .array(["parameters","optimizer","randomGenerator","epoch","metrics","dataLoading","objective","objectiveBuffers"].map(JSONValue.string));forged["lineage"] = .object(lineage)
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(forged,mode:.exactResume,targetEpochs:2,parent:parent,hasObjective:true,dataLoadingIdentity:current))
        forged=resumed;var runtime=try object(forged["runtime"]);runtime["device"] = .string("mps");forged["runtime"] = .object(runtime)
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validate(forged,mode:.exactResume,targetEpochs:2,parent:parent,hasObjective:true,dataLoadingIdentity:current))
    }
}
