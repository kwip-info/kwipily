import Foundation
import CryptoKit
import KwipMLCore

struct TensorFlowWorkflowResult {
    struct Evaluation:Codable,Equatable {let loss:Double;let accuracy:Double;let samples:Int}
    struct Predictions:Codable,Equatable {let rowIDs:[String];let logits:[[Double]];let classes:[Int]}
    let predictions:Predictions?
    let evaluation:Evaluation
    let provenance:[String:JSONValue]
    let artifacts:[WorkflowArchiveFile]
    let manifestSHA256:String
    private struct Manifest:Decodable {
        let schemaVersion:Int;let backendID:String;let runtimeVersion:String;let device:String;let task:String
        let configuration:TensorFlowPreparedWorkerRequest.Configuration
        let predictions:Predictions?
        let provenance:[String:JSONValue];let evaluation:Evaluation;let artifacts:[WorkflowArchiveFile]
    }
    static func verify(directory:URL,request:TensorFlowPreparedWorkerRequest)throws->Self {
        try request.validate()
        let raw=try DiagnosticFileIO.read(root:directory,parts:["manifest.json"],limit:2_000_000)
        let manifest=try JSONDecoder().decode(Manifest.self,from:raw)
        let canonical=try JSONDecoder().decode(TensorFlowPreparedWorkerRequest.self,from:JSONEncoder().encode(request))
        guard manifest.schemaVersion==1,manifest.backendID=="tensorflow",manifest.runtimeVersion=="2.21.0",manifest.device=="cpu",manifest.task=="binaryClassification",manifest.configuration==canonical.configuration,
              manifest.evaluation.loss.isFinite,manifest.evaluation.loss>=0,manifest.evaluation.accuracy.isFinite,(0...1).contains(manifest.evaluation.accuracy),(1...100000).contains(manifest.evaluation.samples),
              (8...512).contains(manifest.artifacts.count),Set(manifest.artifacts.map(\.path)).count==manifest.artifacts.count else {throw TensorFlowBridgeError("TensorFlow result does not match its captured request.")}
        let p=manifest.provenance
        guard case .object(let prep)=p["preparation"],prep["manifestSHA256"] == .string(request.prepared.manifestSHA256),prep["graphDigest"] == .string(request.prepared.graphDigest),
              prep["trainingNode"] == .string(request.prepared.trainingNode),prep["trainingPort"] == .string(request.prepared.trainingPort),prep["validationNode"] == .string(request.prepared.validationNode),prep["validationPort"] == .string(request.prepared.validationPort),
              p["featureColumns"] == .array(request.prepared.featureColumns.map(JSONValue.string)),p["targetColumn"] == .string(request.prepared.targetColumn),
              case .integer(let trainingCount)=p["trainingSamples"],(1...100000).contains(trainingCount),p["validationSamples"] == .integer(Int64(manifest.evaluation.samples)),
              case .array(let training)=p["trainingRowIDs"],case .array(let validation)=p["validationRowIDs"],training.count==Int(trainingCount),validation.count==manifest.evaluation.samples,
              training.allSatisfy(isRowID),validation.allSatisfy(isRowID),Set(training).count==training.count,Set(validation).count==validation.count,Set(training).isDisjoint(with:Set(validation)) else {throw TensorFlowBridgeError("TensorFlow prepared-data provenance differs from captured inputs.")}
        if request.capturePredictions == true {
            guard let captured=manifest.predictions,captured.rowIDs.count==min(8,manifest.evaluation.samples),captured.logits.count==captured.rowIDs.count,captured.classes.count==captured.rowIDs.count,
                  captured.rowIDs.map(JSONValue.string)==Array(validation.prefix(8)),captured.logits.allSatisfy({$0.count==2 && $0.allSatisfy(\.isFinite)}),
                  zip(captured.logits,captured.classes).allSatisfy({row,label in label == (row[1]>row[0] ? 1 : 0)}) else {throw TensorFlowBridgeError("TensorFlow prediction samples differ from the captured privacy policy or validation rows.")}
        } else if manifest.predictions != nil {throw TensorFlowBridgeError("TensorFlow output captured predictions despite the disabled policy.")}
        let required:Set<String>=["history.json","preparation/manifest.json","portable/manifest.json","portable/model.weights.h5","portable/graph.json","portable/inference.py","portable/tensorflow_foundation.py","portable/requirements.txt"]
        let actual=Set(manifest.artifacts.map(\.path))
        guard required.isSubset(of:actual),actual.allSatisfy({$0=="history.json" || $0.hasPrefix("preparation/") || $0.hasPrefix("portable/")}) else {throw TensorFlowBridgeError("Incomplete TensorFlow output inventory.")}
        var total=0
        for file in manifest.artifacts {
            guard file.bytes>=0,file.bytes<=50_000_000 else {throw TensorFlowBridgeError("TensorFlow artifact exceeds size bounds.")}
            total += file.bytes;guard total<=250_000_000 else {throw TensorFlowBridgeError("TensorFlow run exceeds size bounds.")}
            try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:50_000_000)
        }
        guard manifest.artifacts.first(where:{$0.path=="preparation/manifest.json"})?.sha256==request.prepared.manifestSHA256 else {throw TensorFlowBridgeError("Archived TensorFlow preparation differs from the selected manifest.")}
        let graph=try JSONDecoder().decode(GraphDocument.self,from:DiagnosticFileIO.read(root:directory,parts:["portable","graph.json"],limit:2_000_000))
        guard graph==request.configuration.graph else {throw TensorFlowBridgeError("Exported TensorFlow graph differs from the captured model.")}
        return .init(predictions:manifest.predictions,evaluation:manifest.evaluation,provenance:p,artifacts:manifest.artifacts,manifestSHA256:SHA256.hash(data:raw).map{String(format:"%02x",$0)}.joined())
    }
    private static func isRowID(_ value:JSONValue)->Bool {
        if case .string(let id)=value {return !id.isEmpty && id.count<=200};return false
    }
    static func validateCompletion(_ data:Data,directory:URL,request:TensorFlowPreparedWorkerRequest)throws->Self {
        struct Completion:Decodable {let event:String;let backendID:String;let manifest:String;let evaluation:Evaluation}
        guard data.count<=16384 else {throw TensorFlowBridgeError("Oversized TensorFlow completion event.")}
        let completion=try JSONDecoder().decode(Completion.self,from:data)
        let result=try verify(directory:directory,request:request)
        guard completion.event=="backendCompleted",completion.backendID=="tensorflow",completion.manifest==directory.appendingPathComponent("manifest.json").path,completion.evaluation==result.evaluation else {throw TensorFlowBridgeError("TensorFlow completion does not match verified output.")}
        return result
    }
}
