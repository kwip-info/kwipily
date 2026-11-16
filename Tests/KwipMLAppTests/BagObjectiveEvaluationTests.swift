import XCTest
import Darwin
@testable import KwipMLCore
@testable import KwipMLApp

final class BagObjectiveEvaluationTests: XCTestCase {
    private struct Fixture {
        let root: URL
        let bag: WorkflowBagArtifactContext
        let manifest: [String: JSONValue]
        let metrics: Data
    }
    private func fixture(classified: Bool = false, device: ContractDevice = .cpu, activation: Bool = true) throws -> Fixture {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let temporary=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:true)
        let root=temporary.resolvingSymlinksInPath(),process=Process(),log=Pipe()
        process.executableURL=repo.appendingPathComponent(".venv/bin/python");process.currentDirectoryURL=repo
        process.arguments=["-B","-c",Self.script,root.path,classified ? "yes":"no",device.rawValue,activation ? "yes":"no"]
        process.standardOutput=log;process.standardError=log
        try process.run();let deadline=Date().addingTimeInterval(40)
        while process.isRunning && Date()<deadline {Thread.sleep(forTimeInterval:0.025)}
        if process.isRunning {_ = Darwin.kill(process.processIdentifier,SIGKILL)};process.waitUntilExit()
        let text=String(decoding:log.fileHandleForReading.readDataToEndOfFile(),as:UTF8.self)
        XCTAssertEqual(process.terminationStatus,0,text)
        let decoder=JSONDecoder()
        let summary=try decoder.decode([String:JSONValue].self,from:Data(contentsOf:root.appendingPathComponent("fixture.json")))
        func decode<T:Decodable>(_ type:T.Type,_ value:JSONValue)throws->T {try decoder.decode(type,from:WorkflowPhaseArtifacts.encode(value))}
        let reference=try decode(PreparedDataReference.self,XCTUnwrap(summary["reference"]))
        let configuration=try LossConfiguration.decodeRecipe(XCTUnwrap(summary["recipe"]))
        let target=try decode(LossTargetMapping.self,XCTUnwrap(summary["targetMapping"]))
        let selection=PreparedObjectiveSelection(reference:reference,objectiveRecipe:configuration.recipeValue,targetMapping:target,scalarization:classified ? .mean:.sum)
        let recipe=try LossRecipe(configuration:configuration,source:.init(domain:.control,localID:"objective"),targetMapping:target)
        let mapping=try BagInputMapping(value:XCTUnwrap(summary["mapping"]))
        guard case .object(let controls)=summary["sourceConfiguration"] else {throw WorkflowPhaseError("Actual source controls missing")}
        let requirement=BagWorkflowInputRequirement(sourceNodeID:"source",configuration:try .init(values:controls),numEmbeddings:5,embeddingDim:2)
        let preparation=try WorkflowBagPreparedEvidence.preparedEvidence(root:root,selection:.objective(selection),mapping:mapping,requirement:requirement)
        let manifest=try decoder.decode([String:JSONValue].self,from:Data(contentsOf:root.appendingPathComponent("training/artifacts/manifest.json")))
        let bag=WorkflowBagArtifactContext(requirement:requirement,mapping:mapping,preparation:preparation,runID:try XCTUnwrap(manifest["runID"]?.stringValue),device:device,outputContract:try .init(sample:[2],dtype:.float32),target:.objective(selection,recipe),policy:.init(activationValues:activation,predictionRows:true),explicitSampleCapture:true)
        return .init(root:root,bag:bag,manifest:manifest,metrics:try Data(contentsOf:root.appendingPathComponent("training/artifacts/metrics.json")))
    }
    private func objective(_ fixture: Fixture) throws -> JSONValue {
        guard case .object(let evaluation)=fixture.manifest["evaluation"] else {throw WorkflowPhaseError("Actual evaluation missing")}
        return try XCTUnwrap(evaluation["objective"])
    }
    func testActualMSEBagEvaluationAndEpochsUseCapturedSumScalarizationOnCPUAndMPS() throws {
        for device in [ContractDevice.cpu,.mps] {
            let f=try fixture(device:device);defer {try? FileManager.default.removeItem(at:f.root)}
            let c=try WorkflowBagArtifacts.objectiveEvaluationContext(context:f.bag,expectedEpochs:2)
            XCTAssertEqual(c.scalarization,.sum)
            let evaluation=try WorkflowObjectiveArtifacts.evaluation(objective(f),context:c)
            XCTAssertNil(evaluation.accuracy);XCTAssertEqual(evaluation.rawShape,[8,2]);XCTAssertEqual(evaluation.aggregationDenominator,1)
            let metrics=try WorkflowObjectiveArtifacts.metrics(f.metrics,context:c)
            XCTAssertEqual(metrics.count,2);XCTAssertTrue(metrics.allSatisfy {$0.accuracy == nil})
            let rows=try JSONDecoder().decode([JSONValue].self,from:f.metrics)
            XCTAssertEqual(try WorkflowObjectiveArtifacts.epoch(rows[1],expectedEpoch:2,context:c),metrics[1])
        }
    }
    func testActualIndexCEBagRecordsOnlyMeasuredClassification() throws {
        let f=try fixture(classified:true);defer {try? FileManager.default.removeItem(at:f.root)}
        let c=try WorkflowBagArtifacts.objectiveEvaluationContext(context:f.bag,expectedEpochs:2)
        let evaluation=try WorkflowObjectiveArtifacts.evaluation(objective(f),context:c)
        XCTAssertNotNil(evaluation.accuracy);XCTAssertEqual(evaluation.rawShape,[])
        XCTAssertEqual(try WorkflowObjectiveArtifacts.metrics(f.metrics,context:c).count,2)
    }
    func testEvaluationRejectsForeignIdentityArithmeticAndFabricatedAccuracy() throws {
        let f=try fixture();defer {try? FileManager.default.removeItem(at:f.root)}
        let c=try WorkflowBagArtifacts.objectiveEvaluationContext(context:f.bag,expectedEpochs:2)
        guard case .object(let original)=try objective(f) else {return XCTFail("Missing actual evaluation")}
        for key in ["componentID","device","observations","rawShape","scalarization","aggregationDenominator","classification"] {
            var altered=original;altered[key] = .null
            XCTAssertThrowsError(try WorkflowObjectiveArtifacts.evaluation(.object(altered),context:c),key)
        }
        var arithmetic=original;arithmetic["effectiveScalar"] = -999
        XCTAssertThrowsError(try WorkflowObjectiveArtifacts.evaluation(.object(arithmetic),context:c))
        var epoch=try XCTUnwrap(JSONDecoder().decode([JSONValue].self,from:f.metrics).first)
        if case .object(var record)=epoch {record["accuracy"] = 0.5;epoch = .object(record)}
        XCTAssertThrowsError(try WorkflowObjectiveArtifacts.epoch(epoch,expectedEpoch:1,context:c))
    }
    func testRawPreviewPermissionRemainsIndependentAndContextBoundsRefuse() throws {
        let f=try fixture(activation:false);defer {try? FileManager.default.removeItem(at:f.root)}
        let c=try WorkflowBagArtifacts.objectiveEvaluationContext(context:f.bag,expectedEpochs:2)
        let evaluation=try WorkflowObjectiveArtifacts.evaluation(objective(f),context:c)
        XCTAssertEqual(evaluation.rawPreview?.captureStatus,.omittedByPolicy);XCTAssertEqual(evaluation.rawPreview?.values,[])
        XCTAssertThrowsError(try WorkflowBagArtifacts.objectiveEvaluationContext(context:f.bag,expectedEpochs:0))
        XCTAssertThrowsError(try WorkflowObjectiveEvaluationContext(recipe:c.recipe,runID:c.runID,outputSampleShape:[0],trainingCount:24,evaluationCount:8,expectedEpochs:2,scalarization:.sum,device:.cpu,policy:.factory))
    }
    private static let script = #"""
import hashlib,json,shutil,sys
from pathlib import Path
from runner import engine
from runner.tests.test_prepared_bag_loader import fixture,options,load
from runner.tests.test_bag_source_graph import graph,registry
from runner.loss_configuration import canonical_configuration
root=Path(sys.argv[1]);classified=sys.argv[2]=='yes';device=sys.argv[3];activation=sys.argv[4]=='yes'
source=fixture(root/'source');args=options(source)
op='loss_cross_entropy' if classified else 'loss_mse'
config=canonical_configuration(op,{} if classified else {'reduction':'none','target_broadcast':True})
recipe={'schemaVersion':1,'componentID':op,'configurationVersion':1,'config':config}
mapping={'schemaVersion':1,'componentID':op,'inputColumns':['i1','i2'],'roles':[{'role':'target','columns':['label' if classified else 'desired'],'dtype':'int64' if classified else 'float32','sampleShape':[]}],'tripletSourceGroups':[]}
admitted=load(source,target_column=None,objective_mapping=mapping,objective_recipe=recipe,prediction_shape=[2])
reg=registry.__wrapped__(root);value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2
engine.run(value,root/'training/artifacts',registry=reg,prepared=admitted,device=device,epochs=2,batch_size=3,objective_recipe=recipe,target_mapping=mapping,objective_scalarization='mean' if classified else 'sum',sample_capture={'schemaVersion':1,'predictionRows':True,'activationValues':activation})
shutil.copytree(source,root/'preparation/artifacts');raw=(source/'manifest.json').read_bytes();manifest=json.loads(raw)
summary={'reference':{'manifestPath':str(root/'preparation/artifacts/manifest.json'),'manifestSHA256':hashlib.sha256(raw).hexdigest(),'graphDigest':manifest['graphDigest'],'trainingNode':'p','trainingPort':'training','validationNode':'p','validationPort':'validation'},'mapping':args['mapping'],'sourceConfiguration':args['source_config'],'recipe':recipe,'targetMapping':mapping}
(root/'fixture.json').write_text(json.dumps(summary))
"""#
}
