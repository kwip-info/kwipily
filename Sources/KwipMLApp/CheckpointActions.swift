import Foundation
import CryptoKit
import KwipMLCore

struct CheckpointParent:Equatable {
    let directory:URL
    let runID:String
    let completedEpoch:Int
    let manifestSHA256:String
    let checkpointSHA256:String
    let metricsSHA256:String
    let graph:GraphDocument
    let device:String
    let lineageMode:String
    let parentRunID:String?
}
struct CheckpointContinuation {
    enum Mode {case exactResume,fineTune}
    let parent:CheckpointParent
    let mode:Mode
}
extension WorkspaceStore {
    func checkpointParent(at directory:URL)throws->CheckpointParent {
        let manifestBytes=try checkpointRead(directory.appendingPathComponent("manifest.json"),maximum:20_000_000)
        guard let manifest=try JSONSerialization.jsonObject(with:manifestBytes) as? [String:Any],manifest["status"] as? String=="completed",let runID=manifest["runID"] as? String,
              let metadata=manifest["checkpoint"] as? [String:Any],metadata["schemaVersion"] as? Int==1,
              let epoch=metadata["completedEpoch"] as? Int,(1...10000).contains(epoch),let sha=metadata["sha256"] as? String,let size=metadata["bytes"] as? Int,let metricsSHA=metadata["metricsSHA256"] as? String,
              let graphObject=manifest["graph"],let contract=metadata["contract"] as? [String:Any],let runtime=contract["runtime"] as? [String:Any],let device=runtime["device"] as? String else {throw checkpointError("Select a completed run with a versioned lifecycle checkpoint. Legacy checkpoints cannot resume or fine-tune.")}
        let checkpoint=try checkpointRead(directory.appendingPathComponent("checkpoint.pt"),maximum:100_000_000)
        let metrics=try checkpointRead(directory.appendingPathComponent("metrics.json"),maximum:4_000_000)
        guard checkpoint.count==size,checkpointHash(checkpoint)==sha,checkpointHash(metrics)==metricsSHA else {throw checkpointError("Parent checkpoint or metrics changed. Select a verified completed run again.")}
        let parentGraph=try JSONDecoder().decode(GraphDocument.self,from:JSONSerialization.data(withJSONObject:graphObject))
        return CheckpointParent(directory:directory,runID:runID,completedEpoch:epoch,manifestSHA256:checkpointHash(manifestBytes),checkpointSHA256:sha,metricsSHA256:metricsSHA,graph:parentGraph,device:device,lineageMode:(manifest["lineage"] as? [String:Any])?["mode"] as? String ?? "newTraining",parentRunID:(manifest["lineage"] as? [String:Any])?["parentRunID"] as? String)
    }
    func checkpointCompatibility(_ continuation:CheckpointContinuation)->String? {
        CheckpointCompatibility.message(continuation, graph: graph, execution: execution, registry: registry)
    }
    func checkpointArguments(_ continuation:CheckpointContinuation,requestDirectory:URL)throws->[String] {
        if let message=checkpointCompatibility(continuation) {throw checkpointError(message)}
        let parent=continuation.parent
        let current=try checkpointParent(at:parent.directory)
        guard current==parent else {throw checkpointError("The selected parent changed after selection. Select it again before continuing.")}
        let destination=requestDirectory.appendingPathComponent("continuation-parent")
        guard !FileManager.default.fileExists(atPath:destination.path) else {throw checkpointError("Continuation snapshot already exists; create a fresh run.")}
        // Re-check every byte against the captured pins while copying, closing the read-to-worker gap.
        let files=[("manifest.json",20_000_000,parent.manifestSHA256),("checkpoint.pt",100_000_000,parent.checkpointSHA256),("metrics.json",4_000_000,parent.metricsSHA256)]
        var snapshot:[String:Data]=[:]
        for (name,limit,sha) in files {let bytes=try checkpointRead(parent.directory.appendingPathComponent(name),maximum:limit);guard checkpointHash(bytes)==sha else {throw checkpointError("Parent artifacts changed during snapshot. No continuation started.")};snapshot[name]=bytes}
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:true)
        for (name,bytes) in snapshot {try bytes.write(to:destination.appendingPathComponent(name),options:.atomic)}
        return [continuation.mode == .exactResume ? "--resume-checkpoint":"--finetune-checkpoint",destination.appendingPathComponent("checkpoint.pt").path]
    }
    func restoreContinuationMetrics(from directory:URL) {
        struct Row:Decodable {var epoch:Int;var trainLoss:Double;var validationLoss:Double;var accuracy:Double}
        guard let data=try? checkpointRead(directory.appendingPathComponent("metrics.json"),maximum:4_000_000),let rows=try? JSONDecoder().decode([Row].self,from:data),rows.count<=10000,rows.allSatisfy({$0.epoch>0 && $0.trainLoss.isFinite && $0.validationLoss.isFinite && $0.accuracy.isFinite}) else{return}
        metrics=rows.map{EpochMetric(epoch:$0.epoch,trainLoss:$0.trainLoss,validationLoss:$0.validationLoss,accuracy:$0.accuracy)}
    }
    private func checkpointRead(_ url:URL,maximum:Int)throws->Data {
        let info=try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey,.fileSizeKey])
        guard info.isRegularFile==true,info.isSymbolicLink != true,(info.fileSize ?? Int.max)<=maximum else {throw checkpointError("Checkpoint artifacts must be bounded regular files; symbolic links are not accepted.")}
        let handle=try FileHandle(forReadingFrom:url);defer{try? handle.close()}
        let bytes=try handle.read(upToCount:maximum+1) ?? Data();guard bytes.count<=maximum else {throw checkpointError("Checkpoint artifact exceeded its size limit.")};return bytes
    }
    private func checkpointHash(_ data:Data)->String {SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}
    private func checkpointError(_ message:String)->NSError {NSError(domain:"KwipML.Checkpoint",code:1,userInfo:[NSLocalizedDescriptionKey:message])}
}

/// Pure compatibility policy shared by UI availability and immutable launch capture.
enum CheckpointCompatibility {
    static func message(_ continuation: CheckpointContinuation, graph: GraphDocument,
                        execution: ExecutionSettings, registry: ComponentRegistry) -> String? {
        let parent=continuation.parent
        guard let currentArchitecture = try? GraphSemantics.canonicalData(graph, registry: registry),
              let parentArchitecture = try? GraphSemantics.canonicalData(parent.graph, registry: registry),
              currentArchitecture == parentArchitecture else {
            return "Use the same layer identities, operators, dimensions and parameter configuration as the parent. Cosmetic names and positions may change."
        }
        if continuation.mode == .exactResume {
            guard execution.device=="cpu",parent.device=="cpu" else{return "Exact resume requires an existing CPU checkpoint and explicit CPU selection in Settings. No device is changed automatically."}
            guard execution.epochs>parent.completedEpoch else{return "Set target total epochs above \(parent.completedEpoch) in Settings before exact resume."}
            let old=parent.graph.execution ?? ExecutionSettings()
            guard execution.seed==old.seed,execution.batchSize==old.batchSize,execution.learningRate==old.learningRate else{return "Exact resume requires the parent's model seed, batch size and learning rate."}
            guard execution.preparedTraining==old.preparedTraining,execution.datasetCSV==old.datasetCSV,execution.featureColumns==old.featureColumns,execution.targetColumn==old.targetColumn else{return "Exact resume requires the same selected data and feature/target mapping."}
            if execution.preparedTraining==nil && (execution.sampleLimit != old.sampleLimit || execution.splitSeed != old.splitSeed || execution.validationPercent != old.validationPercent) {return "Exact resume requires the parent's sample limit and split settings."}
        }
        return nil
    }
}
