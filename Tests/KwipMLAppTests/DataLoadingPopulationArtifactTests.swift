import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataLoadingPopulationArtifactTests:XCTestCase {
    private static var generated:URL?
    override class func tearDown() {if let url=generated {try? FileManager.default.removeItem(at:url)};generated=nil;super.tearDown()}
    private func object(_ raw:JSONValue?)throws->[String:JSONValue] {guard case .object(let value)=raw else {throw WorkflowPhaseError("Invalid actual fixture object.")};return value}
    func fixture(_ name:String="subset")throws->([String:JSONValue],WorkflowFailedGradientEvidence.PreparedRowCounts) {
        let root=URL(fileURLWithPath:try XCTUnwrap(ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"]))
        if Self.generated == nil {
            let folder=FileManager.default.temporaryDirectory.appendingPathComponent("data-loading-verifier-"+UUID().uuidString)
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
            let process=Process(),pipe=Pipe();process.executableURL=root.appendingPathComponent(".venv/bin/python");process.currentDirectoryURL=root
            process.arguments=["-c",Self.script,folder.path];process.standardOutput=pipe;process.standardError=pipe
            try process.run();let output=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
            guard process.terminationStatus==0 else {throw WorkflowPhaseError(String(decoding:output,as:UTF8.self))};Self.generated=folder
        }
        let folder=try XCTUnwrap(Self.generated),manifest=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:folder.appendingPathComponent(name+"/manifest.json")))
        let rows=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:folder.appendingPathComponent("prepared.json")))
        func strings(_ key:String)throws->[String] {guard case .array(let values)=rows[key] else {throw WorkflowPhaseError("Missing actual prepared rows.")};return try values.map {try XCTUnwrap($0.stringValue)}}
        let training=try strings("trainingRowIDs"),validation=try strings("validationRowIDs")
        return(manifest,.init(trainingRows:training.count,validationRows:validation.count,trainingTableSHA256:try XCTUnwrap(rows["trainingTableSHA256"]?.stringValue),validationTableSHA256:try XCTUnwrap(rows["validationTableSHA256"]?.stringValue),trainingRowIDs:training,validationRowIDs:validation))
    }
    private static let script = #"""
import sys,json
from pathlib import Path
from runner import engine
from runner.tests.test_dataset_loader_engine import setup
from runner.tests.test_dataset_loader_configuration import recipe
base=Path(sys.argv[1]);graph,kw=setup(base)
graph['name']='Data loading population fixture'
for node in graph['nodes']:node.update(name='',x=0.0,y=0.0)
# Existing preparation helper performs actual CSV cast/split/pinning.
from runner.tests.test_dataset_loader_engine import policy
for name,dataset,loader,batch in [('subset',recipe(kind='subset',indices=[4,1,0,7,3]),recipe('data_loader',batch_size=2),2),('drop',recipe(kind='subset',indices=[4,1,0,7,3]),recipe('data_loader',batch_size=2,drop_last=True),2),('stream',recipe(kind='boundedIterable',max_rows=5),None,3000),('legacy',None,None,5)]:
 engine.run(graph,base/name,epochs=2,batch_size=batch,dataset_recipe=dataset,loader_recipe=loader,gradient_policy=policy() if name!='legacy' else None,**kw)
p=kw['prepared'][1]
# Retain actual source IDs and actual sealed table pins for native validation.
actual=json.loads((base/'source/preparation/manifest.json').read_text())
pins={entry['port']:entry['sha256'] for entry in actual['artifacts'] if entry.get('nodeID')=='p' and entry.get('kind')=='table'}
(base/'prepared.json').write_text(json.dumps(dict(trainingRowIDs=p['trainingRowIDs'],validationRowIDs=p['validationRowIDs'],trainingTableSHA256=pins['training'],validationTableSHA256=pins['validation'])))
"""#
    private struct Capture {
        let manifest:[String:JSONValue]
        let rows:WorkflowFailedGradientEvidence.PreparedRowCounts
        let folder:URL
        let graph:GraphDocument
        let registry:ComponentRegistry
        let selection:PreparedObjectiveSelection
        let objective:LossRecipe
        let dataset:DatasetRecipe?
        let loader:DataLoaderRecipe?
        let loading:WorkflowDataLoadingIdentity?
        let gradient:GradientPolicyConfiguration?
    }
    private func capture(_ name:String="drop")throws->Capture {
        let (manifest,rows)=try fixture(name),settings=try object(manifest["settings"]),provenance=try object(manifest["provenance"]),preparation=try object(provenance["preparation"])
        let base=try XCTUnwrap(Self.generated),folder=base.appendingPathComponent(name)
        let graph=try JSONDecoder().decode(GraphDocument.self,from:Data(contentsOf:folder.appendingPathComponent("graph.json")))
        let root=URL(fileURLWithPath:try XCTUnwrap(ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"]))
        let registry=try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json")))
        let configuration=try LossConfiguration.decodeRecipe(XCTUnwrap(settings["objectiveRecipe"]))
        let mapping=try JSONDecoder().decode(LossTargetMapping.self,from:WorkflowPhaseArtifacts.encode(XCTUnwrap(settings["targetMapping"])))
        let objective=try LossRecipe(configuration:configuration,source:.init(domain:.control,localID:"objective"),targetMapping:mapping)
        func string(_ key:String)throws->String {try XCTUnwrap(preparation[key]?.stringValue)}
        let reference=try PreparedDataReference(manifestPath:base.appendingPathComponent("source/preparation/manifest.json").path,manifestSHA256:string("manifestSHA256"),graphDigest:string("graphDigest"),trainingNode:string("trainingNode"),trainingPort:string("trainingPort"),validationNode:string("validationNode"),validationPort:string("validationPort"))
        let selection=PreparedObjectiveSelection(reference:reference,objectiveRecipe:configuration.recipeValue,targetMapping:mapping)
        let dataset=try settings["datasetRecipe"].map {DatasetRecipe(configuration:try DatasetConfiguration.decodeRecipe($0),source:.init(domain:.control,localID:"dataset"))}
        let loader=try settings["loaderRecipe"].map {DataLoaderRecipe(configuration:try DataLoaderConfiguration.decodeRecipe($0),source:.init(domain:.control,localID:"loader"))}
        let loading=try WorkflowDataLoadingVerifier.validate(manifest:manifest,datasetRecipe:dataset,loaderRecipe:loader,prepared:rows,capturedBatchSize:XCTUnwrap(settings["requestedBatchSize"]?.integerValue),expectedEpochs:2)
        let gradient=try settings["gradientPolicyRecipe"].map {try GradientPolicyConfiguration.decodeRecipe($0)}
        return .init(manifest:manifest,rows:rows,folder:folder,graph:graph,registry:registry,selection:selection,objective:objective,dataset:dataset,loader:loader,loading:loading,gradient:gradient)
    }
    func testActualObjectiveEpochsUseRetainedRowsWithoutChangingOriginalPreparation()throws {
        let c=try capture(),output=try WorkflowMappingRequirements.outputContract(c.graph,registry:c.registry)
        let context=try WorkflowObjectiveIdentity.validateEvaluationIdentity(c.manifest,selection:c.selection,recipe:c.objective,outputContract:output,expectedEpochs:2,policy:.factory,dataLoading:c.loading)
        XCTAssertEqual(context.trainingCount,4);XCTAssertEqual(context.evaluationCount,4);XCTAssertEqual(c.rows.trainingRows,12)
        let epochs=try WorkflowObjectiveArtifacts.metrics(Data(contentsOf:c.folder.appendingPathComponent("metrics.json")),context:context)
        XCTAssertEqual(epochs.count,2)
        let dense=try WorkflowObjectiveIdentity.validate(c.manifest,selection:c.selection,recipe:c.objective,inputContract:WorkflowMappingRequirements.inputContract(c.graph,registry:c.registry,context:.explicitObjective),outputContract:output,expectedEpochs:2,policy:.factory,dataLoading:c.loading)
        XCTAssertEqual(dense.trainingCount,4)
    }
    func testActualGradientCadenceUsesSelectedMicrobatchesAndRetainedTail()throws {
        for (name,m,u) in [("subset",3,2),("drop",2,1),("stream",1,1)] {
            let c=try capture(name),settings=try object(c.manifest["settings"])
            let identity=try WorkflowGradientPolicyVerifier.validate(manifest:c.manifest,gradientPolicy:c.gradient,trainability:nil,derivativeCapture:.factory,graph:c.graph,trainingRows:c.rows.trainingRows,batchSize:XCTUnwrap(settings["requestedBatchSize"]?.integerValue),context:.explicitObjective,capturedRegistry:c.registry,dataLoading:c.loading)
            XCTAssertEqual(identity.microbatchesPerEpoch,m);XCTAssertEqual(identity.optimizerUpdatesPerEpoch,u)
        }
    }
    func testActualPhaseRetainsSelectedPortableHelperAndPopulationIdentity()throws {
        let c=try capture(),settings=try object(c.manifest["settings"])
        let optimizer=OptimizerRecipe(configuration:try OptimizerConfiguration.decodeRecipe(XCTUnwrap(settings["optimizerRecipe"])),source:.init(domain:.control,localID:"optimizer"))
        let manifest=try Data(contentsOf:try XCTUnwrap(Self.generated).appendingPathComponent("source/preparation/manifest.json"))
        let trained=try WorkflowPhaseArtifacts.trainingArtifacts(graph:c.graph,settings:.init(device:"cpu",epochs:2,batchSize:2,seed:41,learningRate:XCTUnwrap(settings["learningRate"]?.doubleValue)),preparation:nil,compositeSource:nil,trainingMode:.fresh,requiresLifecycleEvidence:false,lifecycleParent:nil,readRoot:c.folder,parts:[],artifactPrefix:"",optimizerRecipe:optimizer,objectiveSelection:c.selection,objectiveRecipe:c.objective,capturedModelRegistry:c.registry,capturedModelRegistrySHA256:WorkflowArchiveCodec.registryHash(c.registry),gradientPolicy:c.gradient,derivativeCapture:.factory,preparedGradientEvidence:.init(rows:c.rows,manifest:manifest,graphDigest:c.selection.reference.graphDigest),datasetRecipe:c.dataset,loaderRecipe:c.loader)
        XCTAssertEqual(trained.dataLoadingIdentity,c.loading);XCTAssertNotNil(trained.dataLoadingModelCode)
        let helper=try Data(contentsOf:c.folder.appendingPathComponent("data_loading_recipe.py"))
        XCTAssertEqual(trained.hashes["data_loading_recipe.py"],WorkflowArchiveCodec.hash(helper))
        XCTAssertEqual(trained.artifactFiles.first(where:{$0.path=="data_loading_recipe.py"})?.bytes,helper.count)
        XCTAssertEqual(trained.objectiveEvaluationContext?.trainingCount,4)
    }
    func testLegacyObjectiveCountRemainsOriginalAndSelectedWithoutProofRefuses()throws {
        let legacy=try capture("legacy"),output=try WorkflowMappingRequirements.outputContract(legacy.graph,registry:legacy.registry)
        let common=try WorkflowObjectiveIdentity.validateEvaluationIdentity(legacy.manifest,selection:legacy.selection,recipe:legacy.objective,outputContract:output,expectedEpochs:2,policy:.factory)
        XCTAssertEqual(common.trainingCount,12)
        let selected=try capture()
        XCTAssertThrowsError(try WorkflowObjectiveIdentity.validateEvaluationIdentity(selected.manifest,selection:selected.selection,recipe:selected.objective,outputContract:output,expectedEpochs:2,policy:.factory))
    }
    func testLivePopulationCalculationUsesSameSelectionDropAndBatchAdmission()throws {
        for (name,expected) in [("subset",5),("drop",4),("stream",5),("legacy",12)] {
            let c=try capture(name),settings=try object(c.manifest["settings"])
            XCTAssertEqual(try WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:c.rows.trainingRows,datasetRecipe:c.dataset,loaderRecipe:c.loader,capturedBatchSize:XCTUnwrap(settings["requestedBatchSize"]?.integerValue)),expected)
        }
        let c=try capture()
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:2,datasetRecipe:c.dataset,loaderRecipe:c.loader,capturedBatchSize:2))
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:12,datasetRecipe:c.dataset,loaderRecipe:c.loader,capturedBatchSize:3))
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:0,datasetRecipe:nil,loaderRecipe:nil,capturedBatchSize:2))
    }

}
