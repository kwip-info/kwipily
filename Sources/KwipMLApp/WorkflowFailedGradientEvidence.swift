import Foundation
import KwipMLCore

struct WorkflowFailedGradientDescriptor:Codable,Equatable {
    var schemaVersion:Int=1
    let file:WorkflowArchiveFile
    let workerRunID:String
}
struct WorkflowFailedGradientEvidence:Codable,Equatable {
    var schemaVersion:Int=1
    var id:String=FailedGradientCaptureCodec.identifier
    var status:String="partialFailedTraining"
    let workflowRunID:String
    let requestSHA256:String
    let preparationManifestSHA256:String
    let graphSHA256:String
    let capability:JSONValue
    let context:FailedGradientCaptureContext
    let completedSummaries:[GradientEpochSummary]
    let interruption:InterruptedGradientEpoch

    static let path="diagnostics/failed-gradients.json"
    static let capability:JSONValue = .object(["schemaVersion":1,"id":.string(FailedGradientCaptureCodec.identifier),"supportedDevices":.array([.string("cpu"),.string("mps")]),"phase":.string("afterBackwardBeforeStep"),"limits":.object(["eventBytes":100000,"artifactBytes":8000000,"captures":1024])])
    static func require(_ condition:Bool,_ message:String)throws {if !condition {throw WorkflowArchiveError(message)}}
    /// Existing Codable summaries omit optional norms; wire contracts require null.
    static func wire<T:Encodable>(_ value:T)throws->Data {
        func fix(_ value:Any)->Any {
            if var map=value as? [String:Any] {
                for (key,value) in map {map[key]=fix(value)}
                if map["presentElementCount"] != nil {if map["l2Norm"]==nil {map["l2Norm"]=NSNull()};if map["maxAbs"]==nil {map["maxAbs"]=NSNull()}}
                if map["event"] as? String == "gradientInterrupted",map["lastBackwardBatchIndex"]==nil {map["lastBackwardBatchIndex"]=NSNull()}
                return map
            }
            if let list=value as? [Any] {return list.map(fix)}
            return value
        }
        return try JSONSerialization.data(withJSONObject:fix(JSONSerialization.jsonObject(with:WorkflowArchiveCodec.encode(value))),options:[.sortedKeys,.withoutEscapingSlashes])
    }
    static func decode(_ data:Data,plan:WorkflowExecutionPlan,workflowRunID:String,requestSHA256:String,preparationSHA256:String,trainingRows:Int,initialEpoch:Int)throws->Self {
        try require(data.count<=FailedGradientCaptureCodec.maximumArtifactBytes,"Failed-gradient artifact exceeds its bound.")
        let raw=try JSONDecoder().decode([String:JSONValue].self,from:data)
        try require(Set(raw.keys)==["schemaVersion","id","status","workflowRunID","requestSHA256","preparationManifestSHA256","graphSHA256","capability","context","completedSummaries","interruption"],"Unknown failed-gradient envelope fields.")
        let value=try JSONDecoder().decode(Self.self,from:data)
        try require(value.schemaVersion==1 && value.id==FailedGradientCaptureCodec.identifier && value.status=="partialFailedTraining" && value.capability==Self.capability,"Unsupported failed-gradient envelope/capability.")
        try require(value.workflowRunID==workflowRunID && value.requestSHA256==requestSHA256 && value.preparationManifestSHA256==preparationSHA256 && value.graphSHA256 == (try WorkflowArchiveCodec.hash(WorkflowPhaseArtifacts.encode(plan.model.graph))),"Failed-gradient captured identities differ.")
        let context=try FailedGradientCaptureCodec.decodeContext(wire(raw["context"]!),graph:plan.model.graph)
        try require(context.configuration==plan.settings.gradientCapture && context.initialEpoch==initialEpoch && context.requestedFinalEpoch==plan.settings.epochs && context.requestedBatchSize==plan.settings.batchSize && context.trainingRows==trainingRows && (plan.settings.device=="auto" || context.device==plan.settings.device),"Failed-gradient request coverage differs.")
        guard case .array(let summaries)=raw["completedSummaries"],summaries.count<10000 else {throw WorkflowArchiveError("Invalid failed-gradient prefix.")}
        var previous:[GradientEpochSummary]=[]
        for row in summaries {previous.append(try FailedGradientCaptureCodec.decodeCompleted(wire(row),context:context,previous:previous))}
        let interrupted=try FailedGradientCaptureCodec.decodeInterrupted(wire(raw["interruption"]!),context:context,previous:previous)
        try require(previous.reduce(0){$0+$1.observedBatchCount}+interrupted.observedBatchCount>0,"A failed-gradient envelope requires actual observed batches.")
        return value
    }
    struct PreparedRowCounts:Equatable {
        let trainingRows:Int
        let validationRows:Int
        let trainingTableSHA256:String
        let validationTableSHA256:String
        let trainingRowIDs:[String]
        let validationRowIDs:[String]
    }
    static func preparedRowCounts(root:URL,parts:[String]=[],selection:WorkflowPreparedSelection)throws->PreparedRowCounts {
        try selection.validate()
        let inventory=try preparedInventory(root:root,parts:parts,reference:selection.reference)
        let header=try JSONDecoder().decode([String:JSONValue].self,from:inventory.manifest)
        guard header["schemaVersion"] == .integer(1),header["kind"] == .string("dataflowExecution"),header["status"] == .string("completed"),header["graphDigest"] == .string(selection.graphDigest),(1...300).contains(inventory.entries.count) else {throw WorkflowArchiveError("Prepared count inventory differs from the captured completed selection.")}
        var required:[String:ContractDType]=[:]
        switch selection {
        case .legacy(let selected):
            for column in selected.featureColumns+[selected.targetColumn] {required[column] = .float32}
        case .objective(let selected):
            for column in selected.targetMapping.inputColumns {required[column] = .float32}
            for role in selected.targetMapping.roles {for column in role.columns {required[column]=role.dtype}}
        }
        let named:NamedInputPlan?
        switch selection {case .legacy(let s):named=s.namedInputPlan;case .objective(let s):named=s.namedInputPlan}
        var sequences:[String:NamedInputPlan.Tensor]=[:]
        if let named {
            try named.validate();required=[:]
            for tensor in named.inputs+named.roles {
                if tensor.representation=="tokens" {sequences[tensor.columns[0]]=tensor}
                else {for column in tensor.columns {required[column]=ContractDType(rawValue:tensor.dtype)}}
            }
        }
        func count(node:String,port:String,role:String)throws->(Int,String,[String]) {
            let pinned=try preparedTable(root:root,prefix:inventory.prefix,entries:inventory.entries,node:node,port:port)
            let table=try JSONDecoder().decode([String:JSONValue].self,from:pinned.bytes)
            guard table["kind"] == .string("table"),case .array(let rows)=table["rows"],(1...100000).contains(rows.count),case .array(let columns)=table["columns"],case .object(let lineage)=table["lineage"],lineage["role"] == .string(role),case .array(let ids)=lineage["observationIDs"],ids.count==rows.count else {throw WorkflowArchiveError("Prepared count table shape or partition differs.")}
            let types=columns.compactMap {if case .object(let c)=$0 {return c["dtype"]?.stringValue};return nil}
            let version=types.contains(where:{["audio","tensor"].contains($0)}) ? 5:types.contains("int64Sequence") ? 4:types.contains(where:TemporalColumnContract.types.contains) ? 3:types.contains("image") ? 2:1
            guard table["schemaVersion"] == .integer(Int64(version)) else {throw WorkflowArchiveError("Prepared count table version differs from its typed columns.")}
            let observationIDs=try ids.map {value->String in guard case .string(let id)=value,!id.isEmpty,id.unicodeScalars.count<=200 else {throw WorkflowArchiveError("Prepared count observation identity is invalid.")};return id}
            try require(Set(observationIDs).count==observationIDs.count,"Prepared count observation identities repeat.")
            for key in ["sourceDigest","partitionDigest","membershipDigest","dataVersion"] {guard case .string(let digest)=lineage[key],WorkflowArchiveCodec.isHash(digest) else {throw WorkflowArchiveError("Prepared count lineage pin is invalid.")}}
            // Reuse the existing typed preview contract, rather than introducing
            // another schema/dtype/lineage validator for the same table format.
            let preview:JSONValue = .object(["kind":.string("table"),"lineage":.object(lineage),"preview":.object(["partition":.string(role),"dataVersion":lineage["dataVersion"]!,"rowCount":.integer(Int64(rows.count)),"columns":.array(columns),"rows":.array(rows.prefix(8).map {row in if case .array(let cells)=row {return .array(cells.map {if case .string(let text)=$0 {return .string(String(text.prefix(200)))};return $0})};return row}),"diagnostics":.array([]),"diagnosticCount":.integer(0)])])
            try WorkflowPreparationPreviewValidation.validate(preview,kind:"table",operatorID:"")
            var indices:[String:(Int,String,Bool)]=[:]
            for (index,value) in columns.enumerated() {
                guard case .object(let fields)=value,case .string(let name)=fields["name"],case .string(let dtype)=fields["dtype"],case .bool(let nullable)=fields["nullable"] else {throw WorkflowArchiveError("Prepared count columns are invalid.")}
                indices[name]=(index,dtype,nullable)
            }
            for (name,dtype) in required {
                guard let column=indices[name],!column.2,(dtype == .int64 ? column.1=="int64":["float64","int64"].contains(column.1)) else {throw WorkflowArchiveError("Prepared count selected numeric column is missing or incompatible.")}
            }
            for name in sequences.keys {guard let column=indices[name],column.1=="int64Sequence",!column.2 else {throw WorkflowArchiveError("Prepared token-sequence column is absent or incompatible.")}}
            for value in rows {
                guard case .array(let cells)=value,cells.count==columns.count else {throw WorkflowArchiveError("Prepared count row width differs from its declared columns.")}
                for (name,tensor) in sequences {
                    guard case .array(let values)=cells[indices[name]!.0],values.count<=65536,values.allSatisfy({if case .integer(let id)=$0 {return id>=0 && id<Int64(tensor.vocabularySize!)};return false}) else {throw WorkflowArchiveError("Prepared token IDs differ from their captured vocabulary range.")}
                    let size=tensor.sampleShape[0],policy=tensor.lengthPolicy!
                    guard (values.count>=size || ["pad","padTruncate"].contains(policy)) && (values.count<=size || ["truncate","padTruncate"].contains(policy)) else {throw WorkflowArchiveError("Prepared token length differs from its explicit mapping policy.")}
                }
                for (name,dtype) in required {
                    let cell=cells[indices[name]!.0]
                    switch cell {
                    case .integer: break
                    case .number(let number):try require(dtype != .int64 && number.isFinite,"Prepared count selected numeric cell is incompatible.")
                    default:throw WorkflowArchiveError("Prepared count selected numeric cell is unavailable.")
                    }
                }
            }
            return(rows.count,pinned.hash,observationIDs)
        }
        let training=try count(node:selection.trainingNode,port:selection.trainingPort,role:"training")
        let validation=try count(node:selection.validationNode,port:selection.validationPort,role:"validation")
        try require(Set(training.2).isDisjoint(with:validation.2),"Prepared training and validation observations overlap.")
        return .init(trainingRows:training.0,validationRows:validation.0,trainingTableSHA256:training.1,validationTableSHA256:validation.1,trainingRowIDs:training.2,validationRowIDs:validation.2)
    }
    static func preparedInventory(root:URL,parts:[String],reference selection:PreparedDataReference)throws->(prefix:[String],manifest:Data,entries:[[String:Any]]) {
        let prefix=parts+["preparation","artifacts"]
        let manifest=try DiagnosticFileIO.read(root:root,parts:prefix+["manifest.json"],limit:2_000_000)
        try require(WorkflowArchiveCodec.hash(manifest)==selection.manifestSHA256,"Failed-gradient preparation pin changed.")
        guard let map=try JSONSerialization.jsonObject(with:manifest) as? [String:Any],let entries=map["artifacts"] as? [[String:Any]] else {throw WorkflowArchiveError("Invalid prepared inventory.")}
        return(prefix,manifest,entries)
    }
    static func preparedTable(root:URL,prefix:[String],entries:[[String:Any]],node:String,port:String)throws->(bytes:Data,hash:String) {
        let matches=entries.filter{$0["nodeID"] as? String==node && $0["port"] as? String==port && $0["kind"] as? String=="table"}
        guard matches.count==1,let file=matches.first,let name=file["path"] as? String,let size=file["bytes"] as? Int,let hash=file["sha256"] as? String else {throw WorkflowArchiveError("Prepared selected table is missing.")}
        try require((0...50_000_000).contains(size) && !name.contains("/"),"Invalid prepared selected file.")
        let bytes=try DiagnosticFileIO.read(root:root,parts:prefix+[name],limit:size)
        try require(bytes.count==size && WorkflowArchiveCodec.hash(bytes)==hash,"Prepared selected table changed.")
        return(bytes,hash)
    }
    static func trainingRows(root:URL,parts:[String]=[],selection:PreparedTrainingSelection)throws->Int {
        try trainingRows(root:root,parts:parts,reference:selection.reference)
    }
    static func trainingRows(root:URL,parts:[String]=[],selection:WorkflowPreparedSelection)throws->Int {
        try trainingRows(root:root,parts:parts,reference:selection.reference)
    }
    private static func trainingRows(root:URL,parts:[String],reference selection:PreparedDataReference)throws->Int {
        let inventory=try preparedInventory(root:root,parts:parts,reference:selection)
        let bytes=try preparedTable(root:root,prefix:inventory.prefix,entries:inventory.entries,node:selection.trainingNode,port:selection.trainingPort).bytes
        guard let table=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],let rows=table["rows"] as? [Any],(1...100000).contains(rows.count) else {throw WorkflowArchiveError("Prepared training rows are unavailable.")}
        return rows.count
    }
}

/// Evidence rejection is independent of model success/failure, never worker policy.
final class WorkflowFailedGradientCollector {
    private let plan:WorkflowExecutionPlan
    private let rows:Int
    private let originalRows:Int
    private let initialEpoch:Int
    private(set) var notice:String?
    private var context:FailedGradientCaptureContext?
    private var summaries:[GradientEpochSummary]=[]
    private var interruption:InterruptedGradientEpoch?
    private var startSeen=false
    private var errorCount=0
    private var completionSeen=false
    private var bytesRetained=0
    init(plan:WorkflowExecutionPlan,trainingRows:Int,initialEpoch:Int) {
        self.plan=plan;self.originalRows=trainingRows;self.initialEpoch=initialEpoch
        do {self.rows=try WorkflowDataLoadingVerifier.retainedTrainingRows(originalRows:trainingRows,datasetRecipe:plan.datasetRecipe,loaderRecipe:plan.loaderRecipe,capturedBatchSize:plan.settings.batchSize,trainOrigin:plan.train,inlineRoles:plan.inlineDataLoadingRoles)}
        catch {self.rows=trainingRows;self.notice="Failed-gradient evidence unavailable: "+error.localizedDescription}
    }
    func consume(_ bytes:Data,event:[String:Any],name:String) {
        guard plan.settings.gradientCapture != nil else {return}
        do {
            if name=="error" {errorCount+=1;return}
            if name=="completed" {completionSeen=true;return}
            guard ["started","gradientSummary","gradientInterrupted"].contains(name) else {
                if interruption != nil || errorCount>0 {throw WorkflowArchiveError("Progress arrived after interrupted gradient evidence.")};return
            }
            guard notice==nil else {return}
            try WorkflowFailedGradientEvidence.require(errorCount==0 && !completionSeen && interruption==nil,"Gradient observations arrived after their terminal boundary.")
            if name=="started" {
                try WorkflowFailedGradientEvidence.require(!startSeen,"Duplicate training start for failed-gradient evidence.");startSeen=true
                guard let capability=event["failedGradientCapture"] else {return}
                let cap=try JSONDecoder().decode(JSONValue.self,from:JSONSerialization.data(withJSONObject:capability))
                try WorkflowFailedGradientEvidence.require(cap==WorkflowFailedGradientEvidence.capability,"Runtime failed-gradient capability differs.")
                guard let capture=event["gradientCapture"] as? [String:Any],let config=capture["config"],let inventory=capture["inventory"],let worker=event["runID"],let device=event["device"],let start=event["initialEpoch"] else {throw WorkflowArchiveError("Failed-gradient start lacks inventory.")}
                let nested=try JSONDecoder().decode([String:JSONValue].self,from:JSONSerialization.data(withJSONObject:capture))
                try WorkflowFailedGradientEvidence.require(Set(nested.keys)==["schemaVersion","initialEpoch","config","inventory"] && nested["schemaVersion"] == .integer(1) && nested["initialEpoch"] == .integer(Int64(initialEpoch)),"Nested gradient capture version or initial epoch differs from the captured request.")
                let raw:[String:Any]=["workerRunID":worker,"device":device,"initialEpoch":start,"requestedFinalEpoch":plan.settings.epochs,"requestedBatchSize":plan.settings.batchSize,"trainingRows":rows,"configuration":config,"inventory":inventory]
                let candidate=try FailedGradientCaptureCodec.decodeContext(JSONSerialization.data(withJSONObject:raw),graph:plan.model.graph)
                try WorkflowFailedGradientEvidence.require(candidate.configuration==plan.settings.gradientCapture && candidate.initialEpoch==initialEpoch && event["totalEpochs"] as? Int==plan.settings.epochs && event["trainingSamples"] as? Int==originalRows && event["batchSize"] as? Int==min(rows,plan.settings.batchSize) && (plan.settings.device=="auto" || candidate.device==plan.settings.device),"Failed-gradient start differs from immutable request/preparation.")
                context=candidate;bytesRetained=bytes.count
            } else {
                guard let context else {if name=="gradientInterrupted" {throw WorkflowArchiveError("Interrupted gradients lack a supported matching start.")};return}
                try WorkflowFailedGradientEvidence.require(bytesRetained<=FailedGradientCaptureCodec.maximumArtifactBytes-bytes.count,"Retained gradient observations exceed their budget.");bytesRetained+=bytes.count
                if name=="gradientSummary" {summaries.append(try FailedGradientCaptureCodec.decodeCompleted(bytes,context:context,previous:summaries))}
                else {interruption=try FailedGradientCaptureCodec.decodeInterrupted(bytes,context:context,previous:summaries)}
            }
        } catch {notice="Failed-gradient evidence unavailable: "+String(error.localizedDescription.prefix(1500));context=nil;summaries=[];interruption=nil}
    }
    func result(workflowRunID:String,requestSHA256:String,preparationSHA256:String)throws->WorkflowFailedGradientEvidence? {
        guard notice==nil,let context,let interruption else {return nil}
        try WorkflowFailedGradientEvidence.require(errorCount==1 && !completionSeen,"Failed-gradient evidence lacks a single handled error boundary.")
        guard summaries.reduce(0,{$0+$1.observedBatchCount})+interruption.observedBatchCount>0 else {return nil}
        let value=WorkflowFailedGradientEvidence(workflowRunID:workflowRunID,requestSHA256:requestSHA256,preparationManifestSHA256:preparationSHA256,graphSHA256:try WorkflowArchiveCodec.hash(WorkflowPhaseArtifacts.encode(plan.model.graph)),capability:WorkflowFailedGradientEvidence.capability,context:context,completedSummaries:summaries,interruption:interruption)
        return try WorkflowFailedGradientEvidence.decode(WorkflowFailedGradientEvidence.wire(value),plan:plan,workflowRunID:workflowRunID,requestSHA256:requestSHA256,preparationSHA256:preparationSHA256,trainingRows:rows,initialEpoch:initialEpoch)
    }
}
