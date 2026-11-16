import Foundation
import KwipMLCore

/// Verify the existing retained backend operation without treating text as numeric columns.
enum DocumentClassificationWorkflowResult {
    static func verify(directory:URL,request:SklearnPreparedWorkerRequest)throws->SklearnWorkflowResult {
        if request.configuration["operation"] == .string("predict") || request.configuration["operation"] == .string("evaluate") {return try verifyPrediction(directory:directory,request:request)}
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["manifest.json"],limit:2_000_000)
        let m=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard m["schemaVersion"] == 1,m["backendID"] == .string("sklearn"),m["runtimeVersion"] == .string("1.7.2"),m["device"] == .string("cpu"),
              m["task"] == .string("documentClassification"),m["stateFormat"] == .string("kwipml.document-classifier.v1"),m["configuration"] == .object(request.configuration),
              case .object(let evaluation)=m["evaluation"],let count=evaluation["count"]?.integerValue,(1...1000000).contains(count),
              let accuracy=evaluation["accuracy"]?.doubleValue,accuracy.isFinite,(0...1).contains(accuracy),
              let logLoss=evaluation["logLoss"]?.doubleValue,logLoss.isFinite,logLoss>=0,
              case .object(let p)=m["provenance"],case .object(let preparation)=p["preparation"],
              preparation["manifestSHA256"] == .string(request.prepared.manifestSHA256),preparation["graphDigest"] == .string(request.prepared.graphDigest),
              preparation["trainingNode"] == .string(request.prepared.trainingNode),preparation["trainingPort"] == .string(request.prepared.trainingPort),
              preparation["validationNode"] == .string(request.prepared.validationNode),preparation["validationPort"] == .string(request.prepared.validationPort),
              p["featureColumns"] == .array(request.prepared.featureColumns.map(JSONValue.string)),p["targetColumn"] == .string(request.prepared.targetColumn),
              case .array(let train)=p["trainingRowIDs"],case .array(let validation)=p["validationRowIDs"],
              !train.isEmpty,train.count<=1000000,validation.count==Int(count),Set(train).count==train.count,Set(validation).count==validation.count,Set(train).isDisjoint(with:Set(validation)),
              p["trainingSamples"] == .integer(Int64(train.count)),p["validationSamples"] == .integer(Int64(count)),
              train.allSatisfy(validID),validation.allSatisfy(validID),case .array(let inventory)=m["artifacts"] else {throw WorkflowArchiveError("Document classification receipt differs from the captured task or partitions.")}
        let files=try inventory.map {try JSONDecoder().decode(WorkflowArchiveFile.self,from:WorkflowArchiveCodec.encode($0))}
        let paths=Set(files.map(\.path))
        guard paths.count==files.count,(7...512).contains(files.count),Set(["model-state.json","portable/state.json","portable/document_classification.py","portable/inference.py","portable/manifest.json","preparation/manifest.json"]).isSubset(of:paths),
              paths.allSatisfy({$0=="model-state.json" || $0=="predictions.json" || $0.hasPrefix("portable/") || $0.hasPrefix("preparation/")}) else {throw WorkflowFileLeaseError.unsafePath}
        var total=0
        for file in files {total+=file.bytes;guard file.bytes>=0,file.bytes<=50_000_000,total<=202_100_000 else {throw WorkflowFileLeaseError.unsafePath};try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:50_000_000)}
        guard files.first(where:{$0.path=="preparation/manifest.json"})?.sha256==request.prepared.manifestSHA256,
              files.first(where:{$0.path=="model-state.json"})?.sha256==files.first(where:{$0.path=="portable/state.json"})?.sha256 else {throw WorkflowArchiveError("Document classifier package differs from its accepted model.")}
        if m["predictionColumns"] != nil {
            guard paths.contains("predictions.json"),m["predictionPartition"] == .string("validation") else {throw WorkflowArchiveError("Fitted prediction table lacks its Validation population.")}
            let table=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["predictions.json"],limit:50_000_000))
            guard table["columns"] == m["predictionColumns"],case .array(let rows)=table["rows"],rows.count==count,case .object(let lineage)=table["lineage"],lineage["role"] == .string("validation"),lineage["observationIDs"] == .array(validation) else {throw WorkflowArchiveError("Fitted prediction table differs from captured Validation rows.")}
        }
        let stateBytes=try DiagnosticFileIO.read(root:directory,parts:["model-state.json"],limit:32_000_000)
        let state=try JSONDecoder().decode([String:JSONValue].self,from:stateBytes)
        guard state["stateFormat"] == .string("kwipml.document-classifier.v1"),case .array(let classes)=state["classes"],(2...128).contains(classes.count),Set(classes).count==classes.count,
              evaluation["classOrder"] == .array(classes),case .array(let matrix)=evaluation["confusionMatrix"],matrix.count==classes.count else {throw WorkflowArchiveError("Classification categories or matrix differ from saved state.")}
        var matrixTotal=0,correct=0
        for (i,row) in matrix.enumerated() {
            guard case .array(let cells)=row,cells.count==classes.count else {throw WorkflowArchiveError("Invalid confusion matrix shape.")}
            for (j,cell) in cells.enumerated() {guard let n=cell.integerValue,n>=0,n<=count else {throw WorkflowArchiveError("Invalid confusion matrix count.")};matrixTotal+=n;if i==j {correct+=n}}
        }
        guard matrixTotal==count,abs(accuracy-Double(correct)/Double(count))<=1e-12 else {throw WorkflowArchiveError("Evaluation denominator differs from confusion matrix.")}
        if request.capturePredictions == true {
            guard case .object(let prediction)=m["predictions"],case .array(let ids)=prediction["rowIDs"],ids==Array(validation.prefix(8)),
                  case .array(let predicted)=prediction["classes"],predicted.count==ids.count,predicted.allSatisfy({classes.contains($0)}),prediction["classOrder"] == .array(classes),
                  case .array(let probabilities)=prediction["probabilities"],probabilities.count==ids.count else {throw WorkflowArchiveError("Captured classification sample differs from the validation prefix.")}
            for row in probabilities {guard case .array(let values)=row,values.count==classes.count else {throw WorkflowArchiveError("Invalid probability width.")};let p=values.compactMap(\.doubleValue);guard p.count==values.count,p.allSatisfy({$0.isFinite && (0...1).contains($0)}),abs(p.reduce(0,+)-1)<1e-9 else {throw WorkflowArchiveError("Invalid class probability distribution.")}}
        } else if m["predictions"] != nil {throw WorkflowArchiveError("Predictions were captured despite disabled retention.")}
        return .init(evaluation:.init(count:Int(count),accuracy:accuracy,logLoss:logLoss),provenance:p,artifacts:files,manifestSHA256:WorkflowArchiveCodec.hash(bytes))
    }
    private static func verifyPrediction(directory:URL,request:SklearnPreparedWorkerRequest)throws->SklearnWorkflowResult {
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["manifest.json"],limit:2_000_000)
        let m=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard let model=request.model,m["schemaVersion"] == 1,m["backendID"] == .string("sklearn"),m["runtimeVersion"] == .string("1.7.2"),m["device"] == .string("cpu"),m["task"] == .string("documentClassification"),m["operation"] == request.configuration["operation"],m["configuration"] == .object(request.configuration),m["stateFormat"] == .string("kwipml.document-classifier.v1"),
              case .object(let p)=m["provenance"],p["modelSHA256"] == .string(model.sha256),case .object(let preparation)=p["preparation"],preparation["manifestSHA256"] == .string(request.prepared.manifestSHA256),preparation["graphDigest"] == .string(request.prepared.graphDigest),preparation["trainingNode"] == .string(request.prepared.trainingNode),preparation["trainingPort"] == .string(request.prepared.trainingPort),
              p["featureColumns"] == .array(request.prepared.featureColumns.map(JSONValue.string)),p["targetColumn"] == .string(request.prepared.targetColumn),case .array(let ids)=p["inputRowIDs"],!ids.isEmpty,ids.count<=1000000,Set(ids).count==ids.count,ids.allSatisfy(validID),case .object(let metrics)=m["evaluation"],metrics["count"]?.integerValue==ids.count,case .array(let rawFiles)=m["artifacts"] else {throw WorkflowArchiveError("Prediction receipt differs from captured model, input or operation.")}
        let files=try rawFiles.map {try JSONDecoder().decode(WorkflowArchiveFile.self,from:WorkflowArchiveCodec.encode($0))}
        let paths=Set(files.map(\.path));guard (6...16).contains(files.count),paths.count==files.count,Set(["model-state.json","predictions.json","portable/state.json","portable/document_classification.py","portable/inference.py","portable/manifest.json"]).isSubset(of:paths),paths.allSatisfy({["model-state.json","predictions.json"].contains($0) || $0.hasPrefix("portable/")}) else {throw WorkflowFileLeaseError.unsafePath}
        var total=0
        for file in files {total+=file.bytes;guard file.bytes>=0,file.bytes<=50_000_000,total<=200_000_000 else {throw WorkflowFileLeaseError.unsafePath};try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:50_000_000)}
        guard files.first(where:{$0.path=="model-state.json"})?.sha256==model.sha256,files.first(where:{$0.path=="portable/state.json"})?.sha256==model.sha256 else {throw WorkflowArchiveError("Prediction altered its frozen model state.")}
        let state=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["model-state.json"],limit:32_000_000))
        guard case .array(let classes)=state["classes"],m["classOrder"] == .array(classes),(2...128).contains(classes.count),Set(classes).count==classes.count else {throw WorkflowArchiveError("Prediction class order differs from its model.")}
        let accuracy:Double,loss:Double
        if request.configuration["operation"] == .string("evaluate") {
            guard let a=metrics["accuracy"]?.doubleValue,a.isFinite,(0...1).contains(a),let l=metrics["logLoss"]?.doubleValue,l.isFinite,l>=0,metrics["classOrder"] == .array(classes) else {throw WorkflowArchiveError("Invalid labeled evaluation metrics.")};accuracy=a;loss=l
        } else {
            guard metrics["accuracy"]==nil,let classified=metrics["classified"]?.integerValue,let failed=metrics["failed"]?.integerValue,classified>=0,failed>=0,classified+failed==ids.count else {throw WorkflowArchiveError("Unlabeled predictions cannot claim accuracy.")};accuracy=0;loss=0
        }
        if request.capturePredictions != true,m["predictions"] != nil {throw WorkflowArchiveError("Prediction sample retention was disabled.")}
        return .init(evaluation:.init(count:ids.count,accuracy:accuracy,logLoss:loss),provenance:p,artifacts:files,manifestSHA256:WorkflowArchiveCodec.hash(bytes))
    }
    private static func validID(_ value:JSONValue)->Bool {guard let s=value.stringValue else {return false};return !s.isEmpty && s.utf8.count<=1024}
}
