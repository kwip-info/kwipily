import Foundation
import CryptoKit
import KwipMLCore

/// Version 2 separates the MPS-to-CPU transfer from float64 promotion. Prior
/// MPS results may be self-consistent zero logits and cannot be trusted by hashes.
enum WorkflowEvaluationComputationCompatibility {
    static func validate(version:Int,actualDevice:String?)throws {
        guard [1,2,3,4].contains(version),let actualDevice,["cpu","mps"].contains(actualDevice) else {
            throw WorkflowArchiveError("Unsupported evaluation computation version or actual device.")
        }
        if version == 1 && actualDevice == "mps" {
            throw WorkflowArchiveError("This saved MPS evaluation may be incorrect due to a prior MPS conversion issue. Rerun the evaluation with the current app before inspecting or exporting its results. The saved record is preserved.")
        }
    }
}

struct WorkflowEvaluationPrediction:Codable,Equatable {
    let observationID:String
    let logits:[Double]
    let predicted:Int
    let target:Int
}
struct WorkflowEvaluationClassMetric:Codable,Equatable {
    let classIndex:Int
    let precision:Double?
    let precisionUnavailableReason:String?
    let recall:Double?
    let recallUnavailableReason:String?
    let support:Int
}
struct WorkflowEvaluationObjectivePrediction:Codable,Equatable {
    let observationID:String
    let output:JSONValue
    let targets:[String:JSONValue]
}
struct WorkflowEvaluationSummary:Decodable,Equatable {
    let schemaVersion:Int
    let evaluatedCount:Int
    let device:String
    let batchSize:Int
    let lossSum:Double?
    let lossMean:Double?
    let accuracy:Double?
    let confusionMatrix:[[Int]]
    let perClass:[WorkflowEvaluationClassMetric]
    let preview:[WorkflowEvaluationPrediction]
    let objective:JSONValue?
    let objectivePreview:[WorkflowEvaluationObjectivePrediction]
    init(schemaVersion:Int,evaluatedCount:Int,device:String,batchSize:Int,lossSum:Double,lossMean:Double,accuracy:Double,confusionMatrix:[[Int]],perClass:[WorkflowEvaluationClassMetric],preview:[WorkflowEvaluationPrediction]) {
        self.schemaVersion=schemaVersion;self.evaluatedCount=evaluatedCount;self.device=device;self.batchSize=batchSize
        self.lossSum=lossSum;self.lossMean=lossMean;self.accuracy=accuracy;self.confusionMatrix=confusionMatrix;self.perClass=perClass;self.preview=preview
        objective=nil;objectivePreview=[]
    }
    private enum CodingKeys:String,CodingKey {case schemaVersion,evaluatedCount,device,batchSize,lossSum,lossMean,accuracy,confusionMatrix,perClass,preview,objective}
    init(from decoder:Decoder)throws {
        let c=try decoder.container(keyedBy:CodingKeys.self)
        schemaVersion=try c.decode(Int.self,forKey:.schemaVersion);evaluatedCount=try c.decode(Int.self,forKey:.evaluatedCount)
        device=try c.decode(String.self,forKey:.device);batchSize=try c.decode(Int.self,forKey:.batchSize)
        switch schemaVersion {
        case 1:
            lossSum=try c.decode(Double.self,forKey:.lossSum);lossMean=try c.decode(Double.self,forKey:.lossMean);accuracy=try c.decode(Double.self,forKey:.accuracy)
            confusionMatrix=try c.decode([[Int]].self,forKey:.confusionMatrix);perClass=try c.decode([WorkflowEvaluationClassMetric].self,forKey:.perClass)
            preview=try c.decode([WorkflowEvaluationPrediction].self,forKey:.preview);objective=nil;objectivePreview=[]
        case 2:
            objective=try c.decode(JSONValue.self,forKey:.objective);objectivePreview=try c.decode([WorkflowEvaluationObjectivePrediction].self,forKey:.preview)
            lossSum=nil;lossMean=nil;accuracy=nil;confusionMatrix=[];perClass=[];preview=[]
        default:throw WorkflowArchiveError("Unsupported evaluation summary version.")
        }
    }
}
struct WorkflowEvaluationManifest:Decodable {
    let schemaVersion:Int
    let kind:String
    let status:String
    let evaluationID:String
    let projectID:String
    let requestIdentity:String
    let requestSHA256:String
    let snapshotReceiptSHA256:String
    let parent:WorkflowFrozenModelReference
    let testEndpoint:DataFlowEndpoint
    let settings:WorkflowEvaluationSettings
    let testProvenance:[String:JSONValue]
    let frozenLineage:[String:JSONValue]
    let runtime:[String:String]
    let artifacts:[WorkflowArchiveFile]
}
enum WorkflowEvaluationArtifacts {
    struct Verified {
        let manifest:WorkflowEvaluationManifest
        let manifestSHA256:String
        let summary:WorkflowEvaluationSummary
        let predictions:[WorkflowEvaluationPrediction]
        let files:[WorkflowArchiveFile]
        var predictionTable:WorkflowPredictionTableEvidence? = nil
        var objectivePredictions:[WorkflowEvaluationObjectivePrediction] = []
        var objectiveEvaluation:ObjectiveEvaluation? = nil
        var observationIDs:[String] {manifest.schemaVersion == 4 ? objectivePredictions.map(\.observationID):predictions.map(\.observationID)}
        var score:Double? {objectiveEvaluation?.effectiveScalar ?? summary.lossMean}
        var accuracy:Double? {objectiveEvaluation?.accuracy ?? summary.accuracy}
    }
    static func verify(request:WorkflowEvaluationWorkerRequest,snapshot:WorkflowFrozenModelSnapshot,root:URL)throws->Verified {
        try Task.checkCancellation();try request.validate();try snapshot.verify()
        guard request.projectID==snapshot.reference.projectID,request.snapshotReceiptSHA256==snapshot.receiptSHA256 else {throw failure("Evaluation request differs from its frozen snapshot.")}
        let manifestBytes=try DiagnosticFileIO.read(root:root,parts:["artifacts","manifest.json"],limit:2_000_000)
        let object=try dictionary(manifestBytes)
        try keys(object,["schemaVersion","kind","status","evaluationID","projectID","requestIdentity","requestSHA256","snapshotReceiptSHA256","parent","testEndpoint","settings","testProvenance","frozenLineage","runtime","artifacts"])
        try keys(try child(object,"parent"),WorkflowFrozenModelReference.fieldNames)
        try keys(try child(object,"testEndpoint"),["node","port"])
        _ = try WorkflowEvaluationSettings.decode(JSONDecoder().decode([String:JSONValue].self,from:JSONSerialization.data(withJSONObject:child(object,"settings"))))
        try keys(try child(object,"runtime"),["python","torch","device"])
        let manifest=try JSONDecoder().decode(WorkflowEvaluationManifest.self,from:manifestBytes)
        try WorkflowEvaluationComputationCompatibility.validate(version:manifest.schemaVersion,actualDevice:manifest.runtime["device"])
        guard manifest.kind=="frozenEvaluation",manifest.status=="completed",
              manifest.evaluationID==request.evaluationID,manifest.projectID==request.projectID,
              manifest.requestIdentity==request.requestIdentity,try manifest.requestSHA256==request.requestSHA256(),
              manifest.snapshotReceiptSHA256==snapshot.receiptSHA256,manifest.parent==snapshot.reference,
              manifest.settings==request.settings,manifest.artifacts.count==(manifest.schemaVersion>=3 ? 3:2),
              Set(manifest.artifacts.map(\.path))==Set(["summary.json","predictions.jsonl"]+(manifest.schemaVersion>=3 ? ["prediction-table.json"]:[])),
              manifest.runtime.values.allSatisfy({!$0.isEmpty && $0.utf8.count<=256}) else {throw failure("Evaluation manifest identity or inventory differs from the request.")}
        guard let rawEntries=object["artifacts"] as? [[String:Any]] else {throw failure("Invalid evaluation inventory.")}
        for entry in rawEntries {try keys(entry,["path","bytes","sha256"])}
        var contents:[String:Data]=[:]
        for file in manifest.artifacts {
            let limit=file.path=="summary.json" ? 1_000_000:100_000_000
            guard file.bytes>0,file.bytes<=limit,WorkflowArchiveCodec.isHash(file.sha256) else {throw failure("Evaluation artifact exceeds its bounds.")}
            let bytes=try DiagnosticFileIO.read(root:root,parts:["artifacts",file.path],limit:limit)
            guard bytes.count==file.bytes,WorkflowArchiveCodec.hash(bytes)==file.sha256 else {throw failure("Evaluation artifact changed.")}
            contents[file.path]=bytes
        }
        if manifest.schemaVersion == 4 {
            let typed=try WorkflowObjectiveEvaluationArtifacts.verify(manifest:manifest,contents:contents,request:request,snapshot:snapshot)
            try verifyPopulation(manifest,snapshot:snapshot,observationIDs:typed.rows.map(\.observationID))
            guard try DiagnosticFileIO.read(root:root,parts:["artifacts","manifest.json"],limit:2_000_000)==manifestBytes else {throw failure("Evaluation changed during verification.")}
            return Verified(manifest:manifest,manifestSHA256:WorkflowArchiveCodec.hash(manifestBytes),summary:typed.summary,predictions:[],files:manifest.artifacts.map {.init(path:"artifacts/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)} + [.init(path:"artifacts/manifest.json",bytes:manifestBytes.count,sha256:WorkflowArchiveCodec.hash(manifestBytes))],predictionTable:typed.table,objectivePredictions:typed.rows,objectiveEvaluation:typed.evaluation)
        }
        let summaryBytes=contents["summary.json"]!,summaryObject=try dictionary(summaryBytes)
        try keys(summaryObject,["schemaVersion","evaluatedCount","device","batchSize","lossSum","lossMean","accuracy","confusionMatrix","perClass","preview"])
        guard let classes=summaryObject["perClass"] as? [[String:Any]],classes.count==2,
              let preview=summaryObject["preview"] as? [[String:Any]],preview.count<=100 else {throw failure("Invalid evaluation summary arrays.")}
        for value in classes {try keys(value,["classIndex","precision","precisionUnavailableReason","recall","recallUnavailableReason","support"])}
        for value in preview {try predictionKeys(value)}
        let summary=try JSONDecoder().decode(WorkflowEvaluationSummary.self,from:summaryBytes)
        guard summary.schemaVersion==1,(1...100000).contains(summary.evaluatedCount),summary.batchSize==request.settings.batchSize,
              ["cpu","mps"].contains(summary.device),manifest.runtime["device"]==summary.device,
              request.settings.device=="auto" || request.settings.device==summary.device else {throw failure("Evaluation count, device or batch configuration differs.")}
        let raw=contents["predictions.jsonl"]!
        guard raw.last==10 else {throw failure("Prediction JSONL must end in a newline.")}
        // Bound the slice array before checking row count: a newline-heavy file
        // can be byte-bounded while otherwise allocating millions of slices.
        let lines=raw.split(separator:10,maxSplits:summary.evaluatedCount+1,omittingEmptySubsequences:false)
        guard lines.count==summary.evaluatedCount+1,lines.last?.isEmpty==true else {throw failure("Prediction count differs from the full evaluated population.")}
        var predictions:[WorkflowEvaluationPrediction]=[]
        for line in lines.dropLast() {
            try Task.checkCancellation()
            guard !line.isEmpty,line.count<=4096 else {throw failure("Prediction line exceeds its bounds.")}
            let bytes=Data(line);try predictionKeys(dictionary(bytes))
            let row=try JSONDecoder().decode(WorkflowEvaluationPrediction.self,from:bytes)
            guard !row.observationID.isEmpty,row.observationID.unicodeScalars.count<=200,row.logits.count==2,
                  row.logits.allSatisfy(\.isFinite),[0,1].contains(row.target),[0,1].contains(row.predicted),
                  row.predicted==(row.logits[1]>row.logits[0] ? 1:0) else {throw failure("Prediction logits, labels or argmax are invalid.")}
            predictions.append(row)
        }
        guard Set(predictions.map(\.observationID)).count==predictions.count,
              summary.preview==Array(predictions.prefix(request.retainComponentOutputs == false ? 0:request.settings.previewLimit)) else {throw failure("Evaluation rows or preview differ from full predictions.")}
        try verifyPopulation(manifest,snapshot:snapshot,observationIDs:predictions.map(\.observationID))
        try verifyMetrics(summary,rows:predictions)
        var tableEvidence:WorkflowPredictionTableEvidence?
        if let bytes=contents["prediction-table.json"],let file=manifest.artifacts.first(where:{$0.path=="prediction-table.json"}) {
            try WorkflowPredictionTableEvidence.verifyClassification(bytes,rows:predictions)
            tableEvidence=try WorkflowPredictionTableEvidence.verify(bytes,file:.init(path:"artifacts/"+file.path,bytes:file.bytes,sha256:file.sha256),partition:request.settings.partitionRole,observationIDs:predictions.map(\.observationID),sourceDigest:try request.requestSHA256())
        }
        // Observe the same completed manifest at the end of verification.
        guard try DiagnosticFileIO.read(root:root,parts:["artifacts","manifest.json"],limit:2_000_000)==manifestBytes else {throw failure("Evaluation changed during verification.")}
        try Task.checkCancellation()
        var verified=Verified(manifest:manifest,manifestSHA256:WorkflowArchiveCodec.hash(manifestBytes),summary:summary,predictions:predictions,
            files:manifest.artifacts.map {.init(path:"artifacts/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)} + [.init(path:"artifacts/manifest.json",bytes:manifestBytes.count,sha256:WorkflowArchiveCodec.hash(manifestBytes))])
        verified.predictionTable=tableEvidence
        return verified
    }
    private static func verifyMetrics(_ summary:WorkflowEvaluationSummary,rows:[WorkflowEvaluationPrediction])throws {
        var matrix=[[0,0],[0,0]],sum=0.0,correction=0.0
        for row in rows {
            matrix[row.target][row.predicted]+=1
            let maximum=max(row.logits[0],row.logits[1])
            let loss=(maximum-row.logits[row.target])+log(exp(row.logits[0]-maximum)+exp(row.logits[1]-maximum))
            guard loss.isFinite else {throw failure("Prediction loss is not finite.")}
            let adjusted=loss-correction,next=sum+adjusted
            correction=(next-sum)-adjusted;sum=next
        }
        guard let lossSum=summary.lossSum,let lossMean=summary.lossMean,let accuracy=summary.accuracy, matrix==summary.confusionMatrix,near(sum,lossSum),near(sum/Double(rows.count),lossMean),
              near(Double(matrix[0][0]+matrix[1][1])/Double(rows.count),accuracy),summary.perClass.count==2 else {throw failure("Summary metrics do not reproduce from full prediction rows.")}
        for index in 0..<2 {
            let metric=summary.perClass[index],support=matrix[index].reduce(0,+),predicted=matrix[0][index]+matrix[1][index],correct=matrix[index][index]
            guard metric.classIndex==index,metric.support==support else {throw failure("Per-class support differs from predictions.")}
            if predicted==0 {
                guard metric.precision==nil,metric.precisionUnavailableReason=="noPredictedObservations" else {throw failure("Undefined precision requires its reason.")}
            } else {guard let precision=metric.precision,near(precision,Double(correct)/Double(predicted)),metric.precisionUnavailableReason==nil else {throw failure("Precision differs from predictions.")}}
            if support==0 {
                guard metric.recall==nil,metric.recallUnavailableReason=="noActualObservations" else {throw failure("Undefined recall requires its reason.")}
            } else {guard let recall=metric.recall,near(recall,Double(correct)/Double(support)),metric.recallUnavailableReason==nil else {throw failure("Recall differs from predictions.")}}
        }
    }
    private static func verifyPopulation(_ manifest:WorkflowEvaluationManifest,snapshot:WorkflowFrozenModelSnapshot,observationIDs:[String])throws {
        func read(_ path:String)throws->Data {
            guard let pin=snapshot.files.first(where:{$0.path==path}) else {throw failure("Frozen snapshot inventory is missing required evaluation evidence.")}
            let bytes=try DiagnosticFileIO.read(root:snapshot.directory,parts:WorkflowArchiveCodec.pathParts(path),limit:WorkflowFrozenModelSnapshot.fileLimit(path))
            guard bytes.count==pin.bytes,WorkflowArchiveCodec.hash(bytes)==pin.sha256 else {throw failure("Frozen evaluation evidence changed.")}
            return bytes
        }
        let role=manifest.settings.partitionRole
        let receiptBytes=try DiagnosticFileIO.read(root:snapshot.directory,parts:["snapshot.json"],limit:256_000)
        guard try WorkflowFrozenModelSnapshot.receiptRole(receiptBytes)==role else {throw failure("Evaluation role differs from captured snapshot.")}
        let receipt=try dictionary(receiptBytes)
        guard (receipt["schemaVersion"] as? Int == 1 ? "test":receipt["partitionRole"] as? String) == role,let endpoint=receipt["testEndpoint"] as? [String:String],endpoint["node"]==manifest.testEndpoint.node,endpoint["port"]==manifest.testEndpoint.port else {throw failure("Evaluated test endpoint differs from frozen selection.")}
        let preparationBytes=try read("preparation/artifacts/manifest.json"),preparation=try dictionary(preparationBytes)
        guard WorkflowArchiveCodec.hash(preparationBytes)==snapshot.reference.preparationManifestSHA256,
              let entries=preparation["artifacts"] as? [[String:Any]] else {throw failure("Frozen preparation inventory differs.")}
        let population=try WorkflowRetainedTablePopulation(root:snapshot.directory,prefix:["preparation","artifacts"],inventory:snapshot.files,
            maximumBytes:PreparationArtifactPolicy(manifest:preparationBytes).totalBytes)
        func table(_ endpoint:DataFlowEndpoint)throws->[String:Any] {
            let matches=entries.filter {$0["kind"] as? String=="table" && $0["nodeID"] as? String==endpoint.node && $0["port"] as? String==endpoint.port}
            guard matches.count==1,let path=matches[0]["path"] as? String,!path.contains("/"),let hash=matches[0]["sha256"] as? String,
                  let count=matches[0]["bytes"] as? Int else {throw failure("Selected table is not uniquely inventoried.")}
            let bytes=try read("preparation/artifacts/"+path)
            guard bytes.count==count,WorkflowArchiveCodec.hash(bytes)==hash else {throw failure("Selected table inventory differs.")}
            return try population.lineage(bytes)
        }
        let lineage=try table(manifest.testEndpoint)
        let graph=try JSONDecoder().decode(DataFlowDocument.self,from:read("preparation/artifacts/dataflow.json"))
        try DataFlowCodec.validate(graph)
        let nodes=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,$0)})
        var current=manifest.testEndpoint,visited=Set<String>()
        while role == "test" {
            guard visited.insert(current.node).inserted,visited.count<=128,let node=nodes[current.node] else {throw failure("Protected test origin is missing or cyclic.")}
            if ["kwip.data.groupSplit","kwip.data.temporalSplit","kwip.data.partition"].contains(node.operatorID) {
                guard current.port=="test" else {throw failure("Evaluation does not originate at a protected test output.")};break
            }
            let inputPort=(node.operatorID == "kwip.data.preprocess" && node.config["mode"] != .string("table") || node.operatorID == TextEncodingConfiguration.operatorID && node.config["vocabularySource"] == .string("training")) ? current.port:"table"
            let parents=graph.edges.filter {$0.to.node==current.node && $0.to.port==inputPort}
            guard parents.count==1 else {throw failure("Protected test ancestry is ambiguous.")}
            current=parents[0].from
        }
        let canonical = role == "test" ? try table(current) : lineage
        guard let ids=lineage["observationIDs"] as? [String],let full=canonical["observationIDs"] as? [String],
              ids==observationIDs,Set(ids)==Set(full),ids.count==full.count,
              lineage["role"] as? String==role,canonical["role"] as? String==role,
              (lineage["sourceDigest"] as? String)==(canonical["sourceDigest"] as? String),
              (lineage["partitionDigest"] as? String)==(canonical["partitionDigest"] as? String) else {throw failure("Predictions differ from the complete archived protected population or order.")}
        let member=membership(ids)
        guard lineage["membershipDigest"] as? String==member,canonical["membershipDigest"] as? String==member else {throw failure("Archived protected membership digest differs.")}
        let trainedBytes=try read("training/artifacts/manifest.json")
        guard WorkflowArchiveCodec.hash(trainedBytes)==snapshot.reference.modelManifestSHA256,
              WorkflowArchiveCodec.hash(try read("training/artifacts/checkpoint.pt"))==snapshot.reference.checkpointSHA256 else {throw failure("Frozen parent artifact pins differ.")}
        let trained=try dictionary(trainedBytes),training=try child(trained,"provenance"),trainingPrep=try child(training,"preparation")
        guard let trainIDs=training["trainingRowIDs"] as? [String],let validationIDs=training["validationRowIDs"] as? [String],
              Set(ids).isDisjoint(with:trainIDs) else {throw failure("Evaluation membership overlaps training.")}
        if role == "test" {
            guard Set(ids).isDisjoint(with:validationIDs) else {throw failure("Test membership overlaps validation.")}
        } else {
            guard ids == validationIDs,training["validationSamples"] as? Int == ids.count,
                  trainingPrep["validationNode"] as? String == manifest.testEndpoint.node,
                  trainingPrep["validationPort"] as? String == manifest.testEndpoint.port,
                  trainingPrep["validationDataVersion"] as? String == lineage["dataVersion"] as? String else {throw failure("Validation must match the exact frozen parent endpoint, order and population.")}
        }
        let test=manifest.testProvenance
        let required = Set(["dataset","sourceHash",role+"Samples","featureColumns","preparation"]).union(manifest.schemaVersion == 4 ? ["objectiveRecipe","objectiveMapping"]:["targetColumn"])
        guard Set(test.keys)==required || Set(test.keys)==required.union(["modelInput"]),
              test["dataset"] == .string("prepared-"+role+"-v1"),test[role+"Samples"] == .integer(Int64(ids.count)),
              test["sourceHash"] == tryValue(training["sourceHash"]),test["sourceHash"] == tryValue(lineage["sourceDigest"]),
              test["featureColumns"] == tryValue(training["featureColumns"]),test["targetColumn"] == optionalValue(training["targetColumn"]),
              test["objectiveRecipe"] == optionalValue(training["objectiveRecipe"]),test["objectiveMapping"] == optionalValue(training["objectiveMapping"]),
              test["modelInput"] == optionalValue(training["modelInput"]),case .object(let prep)=test["preparation"] else {throw failure("Test mapping or source provenance differs from frozen training.")}
        guard Set(prep.keys)==["manifestSHA256","graphDigest","registryDigest",role+"Node",role+"Port","selectedDataVersion","evaluatedDataVersion","partitionDigest","membershipDigest","fittedStateDigests","appliedStateDigests"],
              prep["manifestSHA256"] == .string(snapshot.reference.preparationManifestSHA256),
              prep["graphDigest"] == tryValue(preparation["graphDigest"]),prep["graphDigest"] == tryValue(trainingPrep["graphDigest"]),
              prep["registryDigest"] == tryValue(preparation["registryDigest"]),
              prep[role+"Node"] == .string(manifest.testEndpoint.node),prep[role+"Port"] == .string(manifest.testEndpoint.port),
              prep["selectedDataVersion"] == tryValue(lineage["dataVersion"]),prep["partitionDigest"] == tryValue(lineage["partitionDigest"]),
              prep["membershipDigest"] == .string(member),prep["fittedStateDigests"] == tryValue(trainingPrep["fittedStateDigests"]),
              case .string(let evaluated)=prep["evaluatedDataVersion"],WorkflowArchiveCodec.isHash(evaluated),
              let fitted=trainingPrep["fittedStateDigests"] as? [String],let prefix=lineage["fittedStateDigests"] as? [String],
              Array(fitted.prefix(prefix.count))==prefix,
              prep["appliedStateDigests"] == .array(fitted.dropFirst(prefix.count).map(JSONValue.string)) else {throw failure("Test preparation or transform identity differs from frozen training.")}
        if prefix==fitted,prep["evaluatedDataVersion"] != prep["selectedDataVersion"] {throw failure("Untransformed selected test data version changed.")}
        let frozen=manifest.frozenLineage
        guard Set(frozen.keys)==["parentRunID","manifestSHA256","checkpointSHA256","registrySHA256","completedEpoch","trainingRuntime","stateDigest","mode"],
              frozen["parentRunID"] == tryValue(trained["runID"]),frozen["manifestSHA256"] == .string(snapshot.reference.modelManifestSHA256),
              frozen["checkpointSHA256"] == .string(snapshot.reference.checkpointSHA256),
              frozen["registrySHA256"] == .string(WorkflowArchiveCodec.hash(try read("model-registry.json"))),
              frozen["mode"] == .string("frozenInference"),case .string(let state)=frozen["stateDigest"],WorkflowArchiveCodec.isHash(state) else {throw failure("Frozen model lineage differs from the pinned parent.")}
        let checkpoint=try child(trained,"checkpoint"),contract=try child(checkpoint,"contract")
        guard frozen["completedEpoch"] == tryValue(checkpoint["completedEpoch"]),frozen["trainingRuntime"] == tryValue(contract["runtime"]) else {throw failure("Frozen checkpoint epoch or runtime differs.")}
        try population.finish()
    }
    private static func membership(_ ids:[String])->String {
        var bytes=Data("kwip.membership.v1\n".utf8)
        for value in ids.map({Array($0.utf8)}).sorted(by:{$0.lexicographicallyPrecedes($1)}) {bytes.append(Data("\(value.count):".utf8));bytes.append(contentsOf:value)}
        return WorkflowArchiveCodec.hash(bytes)
    }
    private static func near(_ a:Double,_ b:Double)->Bool {a.isFinite && b.isFinite && abs(a-b)<=max(1e-10,1e-9*max(abs(a),abs(b)))}
    private static func predictionKeys(_ object:[String:Any])throws {try keys(object,["observationID","logits","predicted","target"])}
    private static func keys(_ object:[String:Any],_ expected:Set<String>)throws {guard Set(object.keys)==expected else {throw failure("Evaluation record has unknown or missing fields.")}}
    private static func dictionary(_ bytes:Data)throws->[String:Any] {guard let value=try JSONSerialization.jsonObject(with:bytes) as? [String:Any] else {throw failure("Expected evaluation object.")};return value}
    private static func child(_ object:[String:Any],_ key:String)throws->[String:Any] {guard let value=object[key] as? [String:Any] else {throw failure("Missing evaluation evidence object.")};return value}
    private static func optionalValue(_ value:Any?)->JSONValue? {guard let value else {return nil};return try? JSONDecoder().decode(JSONValue.self,from:JSONSerialization.data(withJSONObject:value,options:[.fragmentsAllowed]))}
    private static func tryValue(_ value:Any?)->JSONValue {optionalValue(value) ?? .null}
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
