import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationArtifactsTests:XCTestCase {
    private struct Fixture {
        let root:URL,snapshot:WorkflowFrozenModelSnapshot,request:WorkflowEvaluationWorkerRequest
        var manifest:[String:Any]
        var summary:[String:Any]
        var predictions:[[String:Any]]
    }
    private func json(_ value:Any)throws->Data {try JSONSerialization.data(withJSONObject:value,options:[.sortedKeys,.withoutEscapingSlashes])}
    private func object<T:Encodable>(_ value:T)throws->Any {try JSONSerialization.jsonObject(with:JSONEncoder().encode(value))}
    private func membership(_ ids:[String])->String {
        var bytes=Data("kwip.membership.v1\n".utf8)
        for id in ids.sorted() {bytes.append(Data("\(id.utf8.count):\(id)".utf8))}
        return WorkflowArchiveCodec.hash(bytes)
    }
    /// Synthetic pinned bytes exercise native artifact acceptance, not selection of
    /// a real trained parent. Actual worker/archive integration is separate evidence.
    private func fixture(device:String="cpu",role:String="test",parentValidationIDs:[String]?=nil,parentValidationPort:String="validation",parentValidationVersion:String?=nil)throws->Fixture {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let input=root.appendingPathComponent("input"),output=root.appendingPathComponent("artifacts")
        try FileManager.default.createDirectory(at:input,withIntermediateDirectories:false)
        try FileManager.default.createDirectory(at:output,withIntermediateDirectories:false)
        let project=UUID().uuidString,run=UUID().uuidString,ids=["test-2","test-1","test-3"],hash=String(repeating:"a",count:64),version=String(repeating:"b",count:64)
        let endpoint=DataFlowEndpoint(node:"split",port:role)
        let settings=WorkflowEvaluationSettings(device:device,batchSize:2,previewLimit:1,partitionRole:role)
        let table:[String:Any]=["schemaVersion":1,"kind":"table","columns":[["name":"a","dtype":"float64","nullable":false],["name":"b","dtype":"float64","nullable":false],["name":"label","dtype":"int64","nullable":false]],"rows":[[1,2,0],[2,3,1],[3,4,0]],"lineage":["role":role,"sourceDigest":hash,"partitionDigest":hash,"membershipDigest":membership(ids),"observationIDs":ids,"dataVersion":version,"fittedStateDigests":[]]]
        let graph:[String:Any]=["schemaVersion":1,"nodes":[["id":"source","operator":"kwip.data.sourceCSV","config":["separator":","]],["id":"split","operator":"kwip.data.groupSplit","config":["groupBy":["a"],"nullKeys":"error","seed":42,"allocationUnit":"groups","validationPercent":20,"testPercent":20]]],"edges":[["from":["node":"source","port":"table"],"to":["node":"split","port":"table"]]]]
        let tableBytes=try json(table),graphBytes=try json(graph),registry=Data("{}".utf8),checkpoint=Data("synthetic-checkpoint".utf8)
        let preparation:[String:Any]=["schemaVersion":1,"kind":"dataflowExecution","status":"completed","graphDigest":hash,"registryDigest":hash,"artifacts":[["path":"test.json","kind":"table","nodeID":"split","port":role,"bytes":tableBytes.count,"sha256":WorkflowArchiveCodec.hash(tableBytes)],["path":"dataflow.json","kind":"graph","bytes":graphBytes.count,"sha256":WorkflowArchiveCodec.hash(graphBytes)]]]
        let prepBytes=try json(preparation),prepHash=WorkflowArchiveCodec.hash(prepBytes)
        var provenance:[String:Any]=["sourceHash":hash,"featureColumns":["a","b"],"targetColumn":"label","trainingRowIDs":["train-1","train-2"],"validationRowIDs":["validation-1","validation-2"],"preparation":["manifestSHA256":prepHash,"graphDigest":hash,"registryDigest":hash,"fittedStateDigests":[]]]
        if role == "validation" {
            provenance["validationRowIDs"]=parentValidationIDs ?? ids
            provenance["validationSamples"]=3
            var prep=provenance["preparation"] as! [String:Any]
            prep["validationNode"]="split";prep["validationPort"]=parentValidationPort
            prep["validationDataVersion"]=parentValidationVersion ?? version
            provenance["preparation"]=prep
        }
        let trainingRuntime:[String:Any]=["python":"3.11","torch":"2.0","platform":"test","machine":"arm64","device":"cpu"]
        let trained:[String:Any]=["runID":run,"status":"completed","provenance":provenance,"checkpoint":["completedEpoch":3,"contract":["runtime":trainingRuntime]]]
        let trainingBytes=try json(trained),trainingHash=WorkflowArchiveCodec.hash(trainingBytes)
        let reference=WorkflowFrozenModelReference(projectID:project,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:hash,outcomeSHA256:hash,modelManifestSHA256:trainingHash,preparationManifestSHA256:prepHash,checkpointSHA256:WorkflowArchiveCodec.hash(checkpoint))
        let contents:[String:Data]=["model-registry.json":registry,"training/artifacts/checkpoint.pt":checkpoint,"training/artifacts/manifest.json":trainingBytes,"preparation/artifacts/manifest.json":prepBytes,"preparation/artifacts/dataflow.json":graphBytes,"preparation/artifacts/test.json":tableBytes]
        var files:[WorkflowArchiveFile]=[]
        for (path,bytes) in contents {
            let url=input.appendingPathComponent(path)
            try FileManager.default.createDirectory(at:url.deletingLastPathComponent(),withIntermediateDirectories:true)
            try bytes.write(to:url)
            files.append(.init(path:path,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes)))
        }
        files.sort {$0.path<$1.path}
        var receiptObject:[String:Any]=["schemaVersion":role == "test" ? 1:2,"reference":try object(reference),"testEndpoint":try object(endpoint),"files":try object(files)]
        if role == "validation" {receiptObject["partitionRole"]="validation"}
        let receipt=try json(receiptObject)
        try receipt.write(to:input.appendingPathComponent("snapshot.json"))
        let snapshot=WorkflowFrozenModelSnapshot(reference:reference,directory:input,files:files,receiptSHA256:WorkflowArchiveCodec.hash(receipt))
        let request=WorkflowEvaluationWorkerRequest(evaluationID:UUID().uuidString,projectID:project,requestIdentity:hash,snapshotReceiptSHA256:snapshot.receiptSHA256,settings:settings)
        let predictions:[[String:Any]]=[
            ["observationID":ids[0],"logits":[0.0,0.0],"predicted":0,"target":0],
            ["observationID":ids[1],"logits":[0.0,log(3.0)],"predicted":1,"target":1],
            ["observationID":ids[2],"logits":[0.0,log(3.0)],"predicted":1,"target":0]]
        let loss=log(2.0)+log(4.0/3.0)+log(4.0)
        let summary:[String:Any]=["schemaVersion":1,"evaluatedCount":3,"device":"cpu","batchSize":2,"lossSum":loss,"lossMean":loss/3,"accuracy":2.0/3,"confusionMatrix":[[1,1],[0,1]],"perClass":[["classIndex":0,"precision":1.0,"precisionUnavailableReason":NSNull(),"recall":0.5,"recallUnavailableReason":NSNull(),"support":2],["classIndex":1,"precision":0.5,"precisionUnavailableReason":NSNull(),"recall":1.0,"recallUnavailableReason":NSNull(),"support":1]],"preview":[predictions[0]]]
        let test:[String:Any]=["dataset":"prepared-"+role+"-v1","sourceHash":hash,role+"Samples":3,"featureColumns":["a","b"],"targetColumn":"label","preparation":["manifestSHA256":prepHash,"graphDigest":hash,"registryDigest":hash,role+"Node":"split",role+"Port":role,"selectedDataVersion":version,"evaluatedDataVersion":version,"partitionDigest":hash,"membershipDigest":membership(ids),"fittedStateDigests":[],"appliedStateDigests":[]]]
        let manifest:[String:Any]=["schemaVersion":1,"kind":"frozenEvaluation","status":"completed","evaluationID":request.evaluationID,"projectID":project,"requestIdentity":hash,"requestSHA256":try request.requestSHA256(),"snapshotReceiptSHA256":snapshot.receiptSHA256,"parent":try object(reference),"testEndpoint":try object(endpoint),"settings":try object(settings),"testProvenance":test,"frozenLineage":["parentRunID":run,"manifestSHA256":trainingHash,"checkpointSHA256":reference.checkpointSHA256,"registrySHA256":WorkflowArchiveCodec.hash(registry),"completedEpoch":3,"trainingRuntime":trainingRuntime,"stateDigest":hash,"mode":"frozenInference"],"runtime":["python":"3.11","torch":"2.0","device":"cpu"],"artifacts":[]]
        var f=Fixture(root:root,snapshot:snapshot,request:request,manifest:manifest,summary:summary,predictions:predictions)
        f.summary["device"]=device
        f.manifest["runtime"]=["python":"3.11","torch":"2.0","device":device]
        try write(&f);return f
    }
    func testValidationVerifiesExactFrozenPopulationWithoutSubstitutingTest()throws {
        let f=try fixture(role:"validation")
        let result=try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root)
        XCTAssertEqual(result.manifest.settings.partitionRole,"validation")
        XCTAssertEqual(result.manifest.testProvenance["dataset"],.string("prepared-validation-v1"))
        XCTAssertEqual(result.summary.evaluatedCount,3)
        for invalid in [try fixture(role:"validation",parentValidationIDs:["test-1","test-2","test-3"]),try fixture(role:"validation",parentValidationPort:"test"),try fixture(role:"validation",parentValidationVersion:String(repeating:"c",count:64))] {
            XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:invalid.request,snapshot:invalid.snapshot,root:invalid.root))
        }
        let test=try fixture()
        let crosswired=WorkflowEvaluationWorkerRequest(evaluationID:test.request.evaluationID,projectID:test.request.projectID,requestIdentity:test.request.requestIdentity,snapshotReceiptSHA256:test.snapshot.receiptSHA256,settings:.init(device:"cpu",batchSize:2,previewLimit:1,partitionRole:"validation"))
        XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:crosswired,snapshot:test.snapshot,root:test.root))
    }
    func testReceiptRoleCannotBeInferredOrAddedToLegacyReceipt()throws {
        let base:[String:Any]=["schemaVersion":1,"reference":[:],"testEndpoint":["node":"x","port":"validation"],"files":[]]
        XCTAssertEqual(try WorkflowFrozenModelSnapshot.receiptRole(json(base)),"test")
        var altered=base;altered["partitionRole"]="validation"
        XCTAssertThrowsError(try WorkflowFrozenModelSnapshot.receiptRole(json(altered)))
        altered["schemaVersion"]=2
        XCTAssertEqual(try WorkflowFrozenModelSnapshot.receiptRole(json(altered)),"validation")
        altered["partitionRole"]="test"
        XCTAssertThrowsError(try WorkflowFrozenModelSnapshot.receiptRole(json(altered)))
    }
    func testCorrectedSchemaAcceptsCPUAndMPSButLegacyMPSIsUntrustedAndPreserved()throws {
        for device in ["cpu","mps"] {
            var f=try fixture(device:device)
            f.manifest["schemaVersion"]=2
            try write(&f)
            XCTAssertNoThrow(try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root))
        }
        let legacy=try fixture(device:"mps")
        let manifestURL=legacy.root.appendingPathComponent("artifacts/manifest.json")
        let before=try Data(contentsOf:manifestURL)
        XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:legacy.request,snapshot:legacy.snapshot,root:legacy.root)) { error in
            XCTAssertTrue(error.localizedDescription.contains("MPS"))
            XCTAssertTrue(error.localizedDescription.contains("may be incorrect"))
            XCTAssertTrue(error.localizedDescription.lowercased().contains("rerun"))
        }
        XCTAssertEqual(try Data(contentsOf:manifestURL),before)
        let cpu=try fixture()
        XCTAssertNoThrow(try WorkflowEvaluationArtifacts.verify(request:cpu.request,snapshot:cpu.snapshot,root:cpu.root))
    }
    private func write(_ f:inout Fixture)throws {
        let summary=try json(f.summary)
        var predictions=Data()
        for row in f.predictions {predictions.append(try json(row));predictions.append(10)}
        try summary.write(to:f.root.appendingPathComponent("artifacts/summary.json"))
        try predictions.write(to:f.root.appendingPathComponent("artifacts/predictions.jsonl"))
        f.manifest["artifacts"]=[("summary.json",summary),("predictions.jsonl",predictions)].map {["path":$0.0,"bytes":$0.1.count,"sha256":WorkflowArchiveCodec.hash($0.1)] as [String:Any]}
        try json(f.manifest).write(to:f.root.appendingPathComponent("artifacts/manifest.json"))
    }
    func testExactFullPredictionsAndIndependentlyComputedMetricsVerify()throws {
        let f=try fixture()
        let result=try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root)
        XCTAssertEqual(result.summary.evaluatedCount,3)
        XCTAssertEqual(result.predictions.map(\.observationID),["test-2","test-1","test-3"])
        XCTAssertEqual(result.summary.confusionMatrix,[[1,1],[0,1]])
        XCTAssertEqual(try XCTUnwrap(result.summary.lossSum),log(2)+log(4.0/3)+log(4),accuracy:1e-10)
    }
    func testResealedFalseMetricsArgmaxPreviewAndPopulationReject()throws {
        for mutation in ["loss","accuracy","confusion","argmax","preview","subset","order","duplicate","classReason","mapping","source","identity","extra"] {
            var f=try fixture()
            switch mutation {
            case "loss":f.summary["lossMean"]=0.0
            case "accuracy":f.summary["accuracy"]=1.0
            case "confusion":f.summary["confusionMatrix"]=[[2,0],[0,1]]
            case "argmax":f.predictions[0]["predicted"]=1
            case "preview":f.summary["preview"]=[]
            case "subset":f.predictions.removeLast()
            case "order":f.predictions.swapAt(0,1)
            case "duplicate":f.predictions[1]["observationID"]="test-2"
            case "classReason":var rows=f.summary["perClass"] as! [[String:Any]];rows[0]["precisionUnavailableReason"]="noPredictedObservations";f.summary["perClass"]=rows
            case "mapping":var value=f.manifest["testProvenance"] as! [String:Any];value["featureColumns"]=["b","a"];f.manifest["testProvenance"]=value
            case "source":var value=f.manifest["testProvenance"] as! [String:Any];value["sourceHash"]=String(repeating:"f",count:64);f.manifest["testProvenance"]=value
            case "identity":f.manifest["evaluationID"]=UUID().uuidString
            default:f.summary["ignoredExtra"]=1
            }
            try write(&f)
            XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root),mutation)
        }
    }
    func testArtifactBytesAndSymlinkCannotBeResealedByPathAlone()throws {
        let f=try fixture(),path=f.root.appendingPathComponent("artifacts/summary.json")
        let bytes=try Data(contentsOf:path)
        try Data("{}".utf8).write(to:path)
        XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root))
        try bytes.write(to:path)
        let saved=f.root.appendingPathComponent("saved-summary.json")
        try FileManager.default.moveItem(at:path,to:saved)
        try FileManager.default.createSymbolicLink(at:path,withDestinationURL:saved)
        XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root))
    }
    func testResealedNewlineHeavyPredictionsRejectWithinDeclaredRowBudget()throws {
        var f=try fixture()
        let bytes=Data(repeating:10,count:1_000_000)
        try bytes.write(to:f.root.appendingPathComponent("artifacts/predictions.jsonl"))
        var entries=try XCTUnwrap(f.manifest["artifacts"] as? [[String:Any]])
        let index=try XCTUnwrap(entries.firstIndex {$0["path"] as? String == "predictions.jsonl"})
        entries[index]["bytes"]=bytes.count
        entries[index]["sha256"]=WorkflowArchiveCodec.hash(bytes)
        f.manifest["artifacts"]=entries
        try json(f.manifest).write(to:f.root.appendingPathComponent("artifacts/manifest.json"))
        XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root))
    }
    func testSingleClassUndefinedRecallAndLargeFiniteLogitsRemainValid()throws {
        var f=try fixture()
        f.predictions[0]["logits"]=[1000.0,1000.0]
        for index in f.predictions.indices {f.predictions[index]["target"]=0}
        let loss=log(2.0)+2*log(4.0)
        f.summary["lossSum"]=loss;f.summary["lossMean"]=loss/3;f.summary["accuracy"]=1.0/3
        f.summary["confusionMatrix"]=[[1,2],[0,0]]
        f.summary["perClass"]=[
            ["classIndex":0,"precision":1.0,"precisionUnavailableReason":NSNull(),"recall":1.0/3,"recallUnavailableReason":NSNull(),"support":3],
            ["classIndex":1,"precision":0.0,"precisionUnavailableReason":NSNull(),"recall":NSNull(),"recallUnavailableReason":"noActualObservations","support":0]]
        f.summary["preview"]=[f.predictions[0]]
        try write(&f)
        XCTAssertNoThrow(try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root))
        var classes=f.summary["perClass"] as! [[String:Any]]
        classes[1]["recall"]=0.0;classes[1]["recallUnavailableReason"]=NSNull()
        f.summary["perClass"]=classes;try write(&f)
        XCTAssertThrowsError(try WorkflowEvaluationArtifacts.verify(request:f.request,snapshot:f.snapshot,root:f.root))
    }

}
