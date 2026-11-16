import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowDataLoadingVerifierTests:XCTestCase {
    private static var generated:URL?
    override class func tearDown() {if let url=generated {try? FileManager.default.removeItem(at:url)};generated=nil;super.tearDown()}
    private func object(_ raw:JSONValue?)throws->[String:JSONValue] {guard case .object(let value)=raw else {throw WorkflowPhaseError("Invalid actual fixture object.")};return value}
    private func fixture(_ name:String="subset")throws->([String:JSONValue],WorkflowFailedGradientEvidence.PreparedRowCounts) {
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
    private func verify(_ manifest:[String:JSONValue],_ prepared:WorkflowFailedGradientEvidence.PreparedRowCounts,selected:Bool=true)throws->WorkflowDataLoadingIdentity? {
        let settings=try object(manifest["settings"])
        let dataset=try settings["datasetRecipe"].map {DatasetRecipe(configuration:try DatasetConfiguration.decodeRecipe($0),source:.init(domain:.control,localID:"dataset"))}
        let loader=try settings["loaderRecipe"].map {DataLoaderRecipe(configuration:try DataLoaderConfiguration.decodeRecipe($0),source:.init(domain:.control,localID:"loader"))}
        return try WorkflowDataLoadingVerifier.validate(manifest:manifest,datasetRecipe:selected ? dataset:nil,loaderRecipe:selected ? loader:nil,prepared:prepared,capturedBatchSize:try XCTUnwrap(settings["requestedBatchSize"]?.integerValue),expectedEpochs:2)
    }
    func testActualSelectedPopulationDropTailAndLegacyLargeBatch()throws {
        for (name,retained,dropped,m,last) in [("subset",5,0,3,1),("drop",4,1,2,2),("stream",5,0,1,5)] {
            let (manifest,prepared)=try fixture(name),identity=try XCTUnwrap(verify(manifest,prepared))
            XCTAssertEqual(identity.originalRows,12);XCTAssertEqual(identity.selectedRows,5);XCTAssertEqual(identity.retainedRows,retained);XCTAssertEqual(identity.droppedRows,dropped);XCTAssertEqual(identity.batchesPerEpoch,m);XCTAssertEqual(identity.lastBatchSize,last)
            XCTAssertEqual(try identity.optimizerUpdatesPerEpoch(accumulationSteps:2),(m+1)/2)
            XCTAssertEqual(identity.requestedBatchSize,name == "stream" ? 3000:2)
        }
    }
    func testActualLegacyAbsenceAndReservedMetadataRefuse()throws {
        let (manifest,prepared)=try fixture("legacy");XCTAssertNil(try verify(manifest,prepared))
        var forged=manifest;forged["dataLoading"] = .object([:]);XCTAssertThrowsError(try verify(forged,prepared))
        let (selected,rows)=try fixture();XCTAssertThrowsError(try verify(selected,rows,selected:false))
    }
    func testCapturedRecipesPopulationPinsAndAllCopiesAreBound()throws {
        let (manifest,prepared)=try fixture()
        for field in ["retainedRows","droppedRows","selectedRows","originalTrainingRows","batchesPerEpoch","selectionSHA256","schemaVersion"] {
            var changed=manifest,identity=try object(changed["dataLoading"]);identity[field] = field == "selectionSHA256" ? .string(String(repeating:"0",count:64)):.integer(99);changed["dataLoading"] = .object(identity)
            // Matching forged copies still cannot replace captured row authority.
            var provenance=try object(changed["provenance"]),checkpoint=try object(changed["checkpoint"]),contract=try object(checkpoint["contract"])
            provenance["dataLoading"] = .object(identity);contract["dataLoading"] = .object(identity);checkpoint["contract"] = .object(contract)
            changed["provenance"] = .object(provenance);changed["checkpoint"] = .object(checkpoint)
            XCTAssertThrowsError(try verify(changed,prepared),field)
        }
        let changed=WorkflowFailedGradientEvidence.PreparedRowCounts(trainingRows:prepared.trainingRows,validationRows:prepared.validationRows,trainingTableSHA256:prepared.trainingTableSHA256,validationTableSHA256:prepared.validationTableSHA256,trainingRowIDs:Array(prepared.trainingRowIDs.reversed()),validationRowIDs:prepared.validationRowIDs)
        XCTAssertThrowsError(try verify(manifest,changed))
        for key in ["requestedBatchSize","batchSize","lastBatchSize","batchesPerEpoch","epochs"] {
            var forged=manifest,settings=try object(manifest["settings"]);settings[key] = .integer(99);forged["settings"] = .object(settings)
            XCTAssertThrowsError(try verify(forged,prepared),key)
        }
        let settings=try object(manifest["settings"])
        let captured=DatasetRecipe(configuration:try DatasetConfiguration.decodeRecipe(settings["datasetRecipe"]!),source:.init(domain:.control,localID:"dataset"))
        var values=try DataLoaderConfiguration.decodeRecipe(settings["loaderRecipe"]!).values;values["shuffle"] = .bool(true)
        let wrongLoader=DataLoaderRecipe(configuration:try .init(values:values),source:.init(domain:.control,localID:"loader"))
        XCTAssertThrowsError(try WorkflowDataLoadingVerifier.validate(manifest:manifest,datasetRecipe:captured,loaderRecipe:wrongLoader,prepared:prepared,capturedBatchSize:2,expectedEpochs:2))
    }
    func testSelectionDigestUsesExactUnescapedUnicodePythonBytes()throws {
        XCTAssertEqual(try WorkflowDataLoadingVerifier.selectionSHA256(positions:[4,1,0],rowIDs:["row-4","row-1","row-0"]),"44347bbf5ce603ec834895f0d5d7827e8fcd19a9ac47c3ecc1af519b2d408457")
        // Independent Python golden vector includes slash, newline, quote, emoji,
        // non-ASCII and U+2028 rather than relying on JSONEncoder spelling.
        XCTAssertEqual(try WorkflowDataLoadingVerifier.selectionSHA256(positions:[2,0,1],rowIDs:["é/😀","line\nquote\"","\u{2028}"]),"d9378e6e5c0aca939446a22a919f6d795b599cadc4e874659a5056c78addcb6e")
    }
}
