import Foundation
import KwipMLCore

enum PrebuiltWorkflowResult {
    static func verify(directory:URL,request:SklearnPreparedWorkerRequest)throws->SklearnWorkflowResult {
        let bytes=try DiagnosticFileIO.read(root:directory,parts:["manifest.json"],limit:1_000_000)
        let m=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard case .object(let provenance)=m["provenance"],provenance["populationFile"] == .string("population.json"),let populationPin=provenance["populationSHA256"]?.stringValue else {throw WorkflowArchiveError("Missing complete population custody.")}
        let populationBytes=try DiagnosticFileIO.read(root:directory,parts:["population.json"],limit:50_000_000)
        guard WorkflowArchiveCodec.hash(populationBytes)==populationPin else {throw WorkflowArchiveError("Population identities differ from their captured pin.")}
        let population=try JSONDecoder().decode([String:JSONValue].self,from:populationBytes)
        let inputOnly=m["schemaVersion"] == 2
        let operation=request.configuration["operation"]?.stringValue ?? "fit"
        guard (m["schemaVersion"] == 1 || (inputOnly && ["predict","evaluate"].contains(operation))),m["backendID"] == .string("sklearn"),m["runtimeVersion"] == .string("1.7.2"),m["device"] == .string("cpu"),m["task"] == .string("prebuilt"),m["stateFormat"] == .string("kwipml.prebuilt-model.v1"),m["configuration"] == .object(request.configuration),case .object(let p)=m["provenance"],case .object(let prep)=p["preparation"],case .array(let ids)=population["inputRowIDs"],!ids.isEmpty,ids.count<=100000,Set(ids).count==ids.count,ids.allSatisfy({$0.stringValue.map{!$0.isEmpty && $0.utf8.count<=1024} ?? false}),case .object(let metrics)=m["evaluation"],metrics["count"]?.integerValue==ids.count,case .array(let rawFiles)=m["artifacts"] else {throw WorkflowArchiveError("Prebuilt model receipt differs from its captured request.")}
        let expectedPreparation = inputOnly
            ? ["manifestSHA256":request.prepared.manifestSHA256,"graphDigest":request.prepared.graphDigest,"inputNode":request.prepared.trainingNode,"inputPort":request.prepared.trainingPort]
            : ["manifestSHA256":request.prepared.manifestSHA256,"graphDigest":request.prepared.graphDigest,"trainingNode":request.prepared.trainingNode,"trainingPort":request.prepared.trainingPort,"validationNode":request.prepared.validationNode,"validationPort":request.prepared.validationPort]
        for (key,value) in expectedPreparation {guard prep[key] == .string(value) else {throw WorkflowArchiveError("Prebuilt model preparation pin differs.")}}
        if inputOnly {guard Set(prep.keys)==Set(expectedPreparation.keys) else {throw WorkflowArchiveError("Frozen input provenance contains unexpected population roles.")}}
        guard p["featureColumns"] == .array(request.prepared.featureColumns.map(JSONValue.string)),p["targetColumn"] == .string(request.prepared.targetColumn),m["predictions"]==nil else {throw WorkflowArchiveError("Prebuilt model feature order or retention differs.")}
        if inputOnly {
            guard Set(population.keys)==["inputRowIDs"],p["inputSamples"]?.integerValue==ids.count,
                  p["trainingSamples"]==nil,p["validationSamples"]==nil,p["trainingRowIDs"]==nil,p["validationRowIDs"]==nil else {
                throw WorkflowArchiveError("Frozen input counts or population roles differ from complete identities.")
            }
        } else {
            // Immutable schema1 records use historical training/validation slot names,
            // including frozen operations. Verify them without rewriting their evidence.
            guard case .array(let train)=population["trainingRowIDs"],case .array(let validation)=population["validationRowIDs"],validation==ids,p["trainingSamples"]?.integerValue==train.count,p["validationSamples"]?.integerValue==ids.count else {throw WorkflowArchiveError("Population counts differ from complete identities.")}
        }
        let files=try rawFiles.map{try JSONDecoder().decode(WorkflowArchiveFile.self,from:WorkflowArchiveCodec.encode($0))}
        let paths=Set(files.map(\.path)),analysis=request.configuration["algorithm"] == .string("dbscan")
        let required:Set<String>=analysis ? ["population.json","predictions.json","profiles.json","preparation/manifest.json"]:["population.json","predictions.json","model-state.json","portable/state.json","portable/prebuilt_models.py","portable/prebuilt-models.json","portable/inference.py","portable/manifest.json","preparation/manifest.json"]
        guard (3...4096).contains(files.count),paths.count==files.count,required.isSubset(of:paths),paths.allSatisfy({["population.json","model-state.json","predictions.json","profiles.json","confusion.json"].contains($0) || $0.hasPrefix("portable/") || $0.hasPrefix("portable-prepared/") || $0.hasPrefix("preparation/")}),!analysis || (!paths.contains("model-state.json") && !paths.contains(where:{$0.hasPrefix("portable/")})) else {throw WorkflowFileLeaseError.unsafePath}
        var total=0
        for file in files {total+=file.bytes;guard file.bytes>=0,file.bytes<=50_000_000,total<=200_000_000 else {throw WorkflowFileLeaseError.unsafePath};try WorkflowPreviewToolContract.verifyFile(file:file,directory:directory,limit:50_000_000)}
        guard files.first(where:{$0.path=="preparation/manifest.json"})?.sha256==request.prepared.manifestSHA256 else {throw WorkflowArchiveError("Captured preparation differs.")}
        if !analysis {
            guard files.first(where:{$0.path=="model-state.json"})?.sha256==files.first(where:{$0.path=="portable/state.json"})?.sha256 else {throw WorkflowArchiveError("Exported model differs.")}
            let state=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["model-state.json"],limit:32_000_000))
            guard state["stateFormat"] == m["stateFormat"],state["featureColumns"] == p["featureColumns"],case .object(let config)=state["configuration"],config["algorithm"]==request.configuration["algorithm"],config["options"]==request.configuration["settings"] else {throw WorkflowArchiveError("Saved model configuration differs.")}
            if let model=request.model {guard p["modelSHA256"] == .string(model.sha256),files.first(where:{$0.path=="model-state.json"})?.sha256==model.sha256 else {throw WorkflowArchiveError("Frozen prediction changed model state.")}}
        }
        if case .object(let attachment)=m["learnedPreparation"] {
            guard operation=="fit",!analysis,Set(attachment.keys)==["path","sha256","bytes","contentDigest"],attachment["path"] == .string("portable-prepared/preparation.json"),let file=files.first(where:{$0.path=="portable-prepared/preparation.json"}),attachment["sha256"] == .string(file.sha256),attachment["bytes"]?.integerValue==file.bytes,files.first(where:{$0.path=="portable-prepared/state.json"})?.sha256==files.first(where:{$0.path=="model-state.json"})?.sha256 else {throw WorkflowArchiveError("Learned preparation package differs from its accepted fit.")}
            let envelope=try PrebuiltPreparationAttachment.decode(DiagnosticFileIO.read(root:directory,parts:["portable-prepared","preparation.json"],limit:PrebuiltPreparationAttachment.maximumBytes),modelBytes:DiagnosticFileIO.read(root:directory,parts:["model-state.json"],limit:32_000_000))
            guard attachment["contentDigest"] == .string(envelope.contentDigest) else {throw WorkflowArchiveError("Learned preparation identity differs.")}
        } else if m["learnedPreparation"] != nil || paths.contains(where:{$0.hasPrefix("portable-prepared/")}) {throw WorkflowArchiveError("Unqualified learned preparation package.")}
        let table=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:directory,parts:["predictions.json"],limit:50_000_000))
        guard table["columns"]==m["predictionColumns"],case .array(let rows)=table["rows"],rows.count==ids.count,case .object(let lineage)=table["lineage"],lineage["observationIDs"] == .array(ids) else {throw WorkflowArchiveError("Complete output rows differ from their captured population.")}
        try validateMetrics(metrics,algorithm:request.configuration["algorithm"]?.stringValue ?? "",operation:request.configuration["operation"]?.stringValue ?? "fit")
        for value in metrics.values {if let n=value.doubleValue {guard n.isFinite else {throw WorkflowArchiveError("Nonfinite estimator metric.")}}}
        return .init(evaluation:.init(count:ids.count,accuracy:metrics["accuracy"]?.doubleValue ?? 0,logLoss:metrics["logLoss"]?.doubleValue ?? 0),provenance:p,artifacts:files,manifestSHA256:WorkflowArchiveCodec.hash(bytes))
    }
    private static func validateMetrics(_ metrics:[String:JSONValue],algorithm:String,operation:String)throws {
        func require(_ ok:Bool)throws {guard ok else {throw WorkflowArchiveError("Estimator metrics violate their task or population contract.")}}
        guard let count=metrics["count"]?.integerValue,(1...100000).contains(count),let entry=PrebuiltModelConfiguration.algorithms[algorithm] else {throw WorkflowArchiveError("Invalid estimator population.")}
        if operation=="predict" || entry.task=="clustering" {try require(["accuracy","logLoss","mae","rmse","rSquared"].allSatisfy{metrics[$0]==nil})}
        if entry.task=="regression" {try require(metrics["accuracy"]==nil && metrics["logLoss"]==nil)}
        if operation != "predict" && entry.task=="classification" {try require(metrics["accuracy"] != nil && metrics["logLoss"] != nil && metrics["confusionMatrix"] != nil && metrics["classOrder"] != nil)}
        for key in ["accuracy","macroPrecision","macroRecall","macroF1"] {if let value=metrics[key] {try require(value.doubleValue.map{$0.isFinite && (0...1).contains($0)} ?? false)}}
        for key in ["logLoss","mae","rmse","inertia"] {if let value=metrics[key] {try require(value.doubleValue.map{$0.isFinite && $0>=0} ?? false)}}
        if let r=metrics["rSquared"],r != .null {try require(r.doubleValue.map{$0.isFinite && $0<=1.000000000001} ?? false)}
        if entry.task=="clustering" {
            guard let clusters=metrics["clusters"]?.integerValue,let noise=metrics["noiseRows"]?.integerValue,let sample=metrics["silhouetteSampleRows"]?.integerValue else {throw WorkflowArchiveError("Missing clustering diagnostics coverage.")}
            try require((0...min(count,algorithm=="dbscan" ? 2000:32)).contains(clusters) && (0...count).contains(noise) && (0...min(count,512)).contains(sample) && metrics["silhouettePopulationRows"]?.integerValue==count && metrics["silhouetteSeed"]==42 && metrics["silhouetteNoisePolicy"] == .string("exclude noise"))
            if let silhouette=metrics["silhouette"],silhouette != .null {try require(silhouette.doubleValue.map{$0.isFinite && (-1...1).contains($0)} ?? false)}
        }
        if case .array(let classes)=metrics["classOrder"],case .array(let matrix)=metrics["confusionMatrix"] {
            try require((2...128).contains(classes.count) && Set(classes).count==classes.count && matrix.count==classes.count)
            var total=0,correct=0
            for (i,row) in matrix.enumerated() {
                guard case .array(let values)=row,values.count==classes.count else {throw WorkflowArchiveError("Invalid confusion matrix shape.")}
                for (j,value) in values.enumerated() {guard let n=value.integerValue,(0...count).contains(n) else {throw WorkflowArchiveError("Invalid confusion count.")};total+=n;if i==j {correct+=n}}
            }
            try require(total==count && abs((metrics["accuracy"]?.doubleValue ?? -1)-Double(correct)/Double(count))<1e-12)
        }
    }

}
