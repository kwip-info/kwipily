import Foundation
import Combine
import Darwin
import KwipMLCore

/// Captured learner ownership. A resume copies verified immutable state into a new operation.
@MainActor final class SelfPlaySession:ObservableObject {
    typealias Environment=GameEpisodeSession.Environment
    let id=UUID(),document:WorkflowDocument,address:WorkflowNodeAddress,validateOnly:Bool
    let environment:Environment
    @Published private(set) var isBusy=false
    @Published private(set) var message=""
    @Published private(set) var artifacts:[String:TableViewQuerySession.Artifact]=[:]
    @Published private(set) var metrics:[[String:JSONValue]]=[]
    private(set) var ports:[String:JSONValue]=[:]
    private(set) var requestIdentity=""
    private(set) var files:[WorkflowArchiveFile]=[]
    private(set) var directory:URL?
    private let reservation:WorkflowExecutionReservation
    private var token:UUID?,task:Task<Void,Never>?,worker:WorkflowWorkerHandle?
    private var cancelled=false
    @Published private(set) var campaign:[String:JSONValue]?
    @Published private(set) var pauseRequested=false
    var campaignEnabled:Bool {guard let node=document.nodes.first(where:{$0.address==address}),case .train(let c)=node.payload,case .object(let values)=c["selfPlay"]else{return false};return SelfPlayCampaignGrant.enabled(values)}
    func pauseAtBoundary(){
        guard isBusy,campaignEnabled,!pauseRequested else{return};pauseRequested=true;message="Pausing after the current iteration commits…"
        Task {[weak self] in
            for _ in 0..<100 {
                guard let self,self.isBusy else{return}
                if let directory=self.directory,FileManager.default.fileExists(atPath:directory.appendingPathComponent("request.json").path){
                    do{try await Task.detached(priority:.utility){try DiagnosticFileIO.writeImmutable(Data("pause".utf8),root:directory,parts:["pause-at-boundary"],limit:64)}.value}catch{self.message="Pause request: "+error.localizedDescription};return
                }
                try? await Task.sleep(for:.milliseconds(50))
            }
        }
    }
    init(document:WorkflowDocument,address:WorkflowNodeAddress,validateOnly:Bool,environment:Environment,reservation:WorkflowExecutionReservation){self.document=document;self.address=address;self.validateOnly=validateOnly;self.environment=environment;self.reservation=reservation}
    func start(modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,resume:SelfPlayRecord?,campaignAction:String?=nil,isCurrent:@escaping()->Bool,onFinish:@escaping()->Void) {
        guard let token=reservation.acquire()else{message="Wait for the current workspace operation.";return};self.token=token;isBusy=true;message=validateOnly ? "Validating policy/value bindings…":"Capturing Self-play…"
        let doc=document,address=address,env=environment,id=id,validateOnly=validateOnly
        task=Task {[self] in
            do {
                let prepared=try await Task.detached(priority:.userInitiated){()->([WorkflowPreviewCapture.Input],WorkflowCaptureStorageReservation?) in
                    let plan=try SelfPlayPlan.compile(doc,address:address,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
                    let recipes=try LayerParameterSettings.recipes(document:doc,model:plan.model,registry:modelRegistry,seed:Int(plan.configuration.values["initializationSeed"]?.integerValue ?? 17),initialization:nil,trainability:nil,train:address)
                    let documentBytes=try WorkflowCodec.encode(doc)
                    let isCampaign=SelfPlayCampaignGrant.enabled(plan.configuration.values) && !validateOnly
                    if let resume {guard resume.document.projectID==doc.projectID,resume.address==address else{throw WorkflowArchiveError("Resume belongs to another project or Train.")}}
                    let legacyResume=resume != nil && !isCampaign
                    let request:[String:JSONValue]=["schemaVersion":1,"trainAddress":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(address)),"documentSHA256":.string(WorkflowArchiveCodec.hash(documentBytes)),"definition":.object(plan.definition),"graph":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(plan.model.graph)),"configuration":.object(plan.configuration.values),"initialization":recipes.0?.configuration.recipeValue ?? .null,"trainability":recipes.1?.configuration.recipeValue ?? .null,"parentOperationID":legacyResume ? resume.map{.string($0.id)} ?? .null:.null,"resume":.bool(legacyResume),"validateOnly":.bool(validateOnly)]
                    var inputs:[WorkflowPreviewCapture.Input]=[.bytes(path:"input/self-play.json",data:try WorkflowArchiveCodec.encode(request)),.bytes(path:"input/document.json",data:documentBytes)]
                    if case .object(let suite)=plan.configuration.values["evaluation"],case .array(let names)=suite["opponents"] {
                        for name in names.compactMap(\.stringValue) where name.hasPrefix("checkpoint:") {
                            let operation=String(name.dropFirst(11))
                            guard let record=try SelfPlayRecord.list(root:env.root,projectID:doc.projectID).first(where:{$0.id==operation && $0.completed})else{throw WorkflowArchiveError("Select a completed same-project evaluation checkpoint.")}
                            _=try record.verify(root:env.root);let base=try record.directory(env.root)
                            for (source,destination) in [("model.pt","model.pt"),("model.json","model.json"),("inference-manifest.json","identity.json")] {
                                guard let entry=record.files.first(where:{$0.path=="artifacts/"+source})else{throw WorkflowFileLeaseError.unsafePath}
                                inputs.append(.file(path:"input/opponents/"+operation+"/"+destination,source:base.appendingPathComponent(entry.path),expectedSHA256:entry.sha256))
                            }
                        }
                    }
                    var admission:WorkflowCaptureStorageReservation?
                    if isCampaign {
                        let previous=try resume.flatMap{try SelfPlayCampaignGrant.captured($0,root:env.root)}
                        let campaignID=campaignAction=="fork" ? UUID().uuidString:previous?["campaignID"]?.stringValue ?? UUID().uuidString
                        if let limit=env.policy.storageLimitBytes {
                            admission=try WorkflowCaptureStorageReservation(workspace:env.root,accountingRoot:env.accounting,projectID:doc.projectID,operationID:id.uuidString,purpose:.selfPlay,maximumCaptureBytes:1_000_000 + (try WorkflowPreviewCapture.byteCountBound(inputs: inputs, purpose: .selfPlay)),additionalScopes:[try .init(namespace:.workflowOperations,components:[doc.projectID,campaignID])],additionalBytes:Int64({if case .object(let c)=plan.configuration.values["campaign"]{return c["maximumBytes"]?.integerValue ?? 1073741824};return 1073741824}()),limitBytes:limit)
                        }
                        var grant=try SelfPlayCampaignGrant.issue(root:env.root,project:doc.projectID,id:campaignID,action:campaignAction ?? (previous == nil ? "start":"resume"))
                        if campaignAction=="fork",let parentID=previous?["campaignID"]?.stringValue {grant["parent"] = .object(try SelfPlayCampaignGrant.issue(root:env.root,project:doc.projectID,id:parentID,action:"resume"))}
                        inputs.append(.bytes(path:"input/campaign-grant.json",data:try WorkflowArchiveCodec.encode(grant)))
                    } else if let resume {
                        guard resume.document.projectID==doc.projectID,resume.address==address else {throw WorkflowArchiveError("Resume belongs to another project or Train.")}
                        inputs += try resume.resumeInputs(root:env.root)
                    }
                    return (inputs,admission)
                }.value
                let inputs=prepared.0,admission=prepared.1
                guard isCurrent(),!cancelled else {throw CancellationError()}
                let root=try SelfPlayRecord.canonicalRoot(env.root);let directory=root.appendingPathComponent("workflow-operations/\(doc.projectID)/\(id.uuidString)");self.directory=directory
                let window=try ExecutionBudgetWindow(policy:env.policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
                worker=try await WorkflowCapturedToolExecution.launch(mode:.selfPlay,workspace:env.root,accountingRoot:env.accounting,projectID:doc.projectID,runtime:env.runtime,window:window,operationID:id.uuidString,storageLimitBytes:admission == nil ? env.policy.storageLimitBytes:nil,storageAdmission:admission,inputs:{_ in inputs},arguments:{["--self-play",$0.appendingPathComponent("input/self-play.json").path]},retainInputsOnSuccess:true,onLine:{[weak self] line in
                    guard let self,let data=line.data(using:.utf8),let event=try? JSONDecoder().decode([String:JSONValue].self,from:data)else{return}
                    if event["event"] == .string("error") {self.message=event["message"]?.stringValue ?? "Self-play failed."}
                    if event["event"] == .string("selfPlayProgress") {
                        let phase=event["phase"]?.stringValue ?? "";self.message="Iteration \(event["iteration"]?.integerValue ?? 0) · \(phase)"
                        if phase=="learning",self.metrics.count<4096 {self.metrics.append(event)}
                    }
                    if event["event"] == .string("selfPlayCompleted"),case .array(let values)=event["files"] {self.requestIdentity=event["requestIdentity"]?.stringValue ?? "";self.files=(try? values.map{try JSONDecoder().decode(WorkflowArchiveFile.self,from:JSONEncoder().encode($0))}) ?? []}
                },onExit:{[self] code,detail,output in
                    defer {finish();onFinish()}
                    do {
                        let record=SelfPlayRecord(id:id.uuidString,address:address,document:doc,files:files,finishedAt:Date(),validateOnly:validateOnly,completed:code==0 && output != nil)
                        try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:directory,parts:["self-play-record.json"],limit:2_000_000)
                        guard code==0,!cancelled,let output else {if cancelled{message="Cancelled · only committed iteration boundaries can resume."}else if !detail.isEmpty{message=detail};return}
                        let admission=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:output,parts:["admission.json"],limit:64_000))
                        if case .object(let summary)=admission["summary"],case .object(let value)=summary["campaign"]{campaign=value}
                        if case .object(let p)=admission["ports"]{ports=p}
                        for port in SelfPlayToolContract.tablePorts(document:doc,address:address) {
                            guard let file=files.first(where:{$0.path=="artifacts/"+port+".table.json"})else{throw WorkflowFileLeaseError.unsafePath}
                            artifacts[port] = .init(indexURL:output.appendingPathComponent(port+".table.json"),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table")
                        }
                        if !validateOnly {
                            Task {[weak self] in
                                let rows=try? await Task.detached(priority:.utility){()->[[String:JSONValue]] in
                                    let result=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:output,parts:["self-play.json"],limit:8_000_000))
                                    guard let values=result["measurements"]else{return []};return try JSONDecoder().decode([[String:JSONValue]].self,from:JSONEncoder().encode(values))
                                }.value
                                guard isCurrent(),let self,let rows,rows.count<=4096 else{return};self.metrics=rows
                            }
                        }
                        message=validateOnly ? "Bindings validated · no training run.":"Completed · paired heads and resume state retained. Evaluation is measured separately from collection."
                        if let campaign {message=(pauseRequested && campaign["complete"] != .bool(true) ? "Paused · ":"")+"Campaign · iteration \(campaign["iteration"]?.integerValue ?? 0) / \(campaign["target"]?.integerValue ?? 0) · \(campaign["reproducibility"]?.stringValue ?? "")"}
                    }catch{artifacts=[:];message="Self-play verification: "+error.localizedDescription}
                })
            }catch{message=error.localizedDescription;finish();onFinish()}
        }
    }
    private func finish(){isBusy=false;worker=nil;task=nil;if let token{reservation.release(token)};token=nil}
    func cancel(){guard isBusy else{return};cancelled=true;if let worker{worker.cancel()}else{task?.cancel()}}
}

struct SelfPlayRecord:Codable,Identifiable {
    var schemaVersion=1
    let id:String,address:WorkflowNodeAddress,document:WorkflowDocument,files:[WorkflowArchiveFile],finishedAt:Date,validateOnly:Bool,completed:Bool
    static func canonicalRoot(_ root:URL)throws->URL {guard let p=realpath(root.path,nil)else{throw WorkflowFileLeaseError.unsafePath};defer{free(p)};return URL(fileURLWithPath:String(cString:p))}
    func directory(_ root:URL)throws->URL {guard UUID(uuidString:id)?.uuidString==id else{throw WorkflowFileLeaseError.unsafePath};return try Self.canonicalRoot(root).appendingPathComponent("workflow-operations/\(document.projectID)/\(id)")}
    func verify(root:URL)throws->[String:TableViewQuerySession.Artifact] {
        guard schemaVersion==1,completed,!validateOnly else{throw WorkflowArchiveError("This run has no completed result tables. Its last committed boundary may still resume.")}
        let directory=try directory(root);let request=try JSONDecoder().decode(WorkflowToolRequest.self,from:DiagnosticFileIO.read(root:directory,parts:["request.json"],limit:262_144))
        guard request.toolMode == .selfPlay,request.operationID==id,request.projectID==document.projectID else{throw WorkflowFileLeaseError.unsafePath}
        try request.validate(directory:directory,verifyInputs:true)
        guard let docFile=request.inputFiles.first(where:{$0.path=="input/document.json"}),try WorkflowArchiveCodec.hash(WorkflowCodec.encode(document))==docFile.sha256 else{throw WorkflowFileLeaseError.unsafePath}
        let completion=try DiagnosticFileIO.read(root:directory,parts:["self-play-outcome.json"],limit:1_000_000)
        let event=try JSONDecoder().decode([String:JSONValue].self,from:completion)
        guard event["requestIdentity"] == .string(request.requestIdentity),let entries=event["files"],try JSONDecoder().decode([WorkflowArchiveFile].self,from:JSONEncoder().encode(entries))==files else{throw WorkflowFileLeaseError.unsafePath}
        var result:[String:TableViewQuerySession.Artifact]=[:]
        for f in files {let limit=f.path.hasPrefix("artifacts/replay/") ? try SelfPlayToolContract.replayLimit(String(f.path.dropFirst("artifacts/replay/".count))):f.path=="artifacts/model.pt" ? 100_000_000:8_000_000;try WorkflowPreviewToolContract.verifyFile(file:f,directory:directory,limit:limit)}
        for port in SelfPlayToolContract.tablePorts(document:document,address:address) {guard let f=files.first(where:{$0.path=="artifacts/"+port+".table.json"})else{throw WorkflowFileLeaseError.unsafePath};result[port] = .init(indexURL:directory.appendingPathComponent(f.path),indexSHA256:f.sha256,sourceSHA256:f.sha256,inputKind:"table")}
        return result
    }
    func resumeInputs(root:URL)throws->[WorkflowPreviewCapture.Input] {
        let directory=try directory(root),replay=directory.appendingPathComponent("artifacts/replay")
        let request=try JSONDecoder().decode(WorkflowToolRequest.self,from:DiagnosticFileIO.read(root:directory,parts:["request.json"],limit:262_144));guard request.toolMode == .selfPlay,request.operationID==id else{throw WorkflowFileLeaseError.unsafePath};try request.validate(directory:directory,verifyInputs:true)
        let pointer=try DiagnosticFileIO.read(root:replay,parts:["latest.json"],limit:128);let value=try JSONDecoder().decode([String:JSONValue].self,from:pointer)
        guard let name=value["checkpoint"]?.stringValue,name.range(of:"^iteration-[0-9]{2}$",options:.regularExpression) != nil else{throw WorkflowFileLeaseError.unsafePath}
        let manifestBytes=try DiagnosticFileIO.read(root:replay,parts:["checkpoints",name,"manifest.json"],limit:2_000_000)
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:manifestBytes)
        guard case .array(let entries)=manifest["replay"],entries.count<=400 else{throw WorkflowFileLeaseError.unsafePath}
        let chunks=try Set(entries.map{value->String in guard case .object(let c)=value,let path=c["path"]?.stringValue else{throw WorkflowFileLeaseError.unsafePath};_=try SelfPlayToolContract.replayLimit("chunks/"+path);return "chunks/"+path})
        guard let enumerator=FileManager.default.enumerator(at:replay,includingPropertiesForKeys:[.isRegularFileKey,.isSymbolicLinkKey],options:[])else{throw WorkflowFileLeaseError.unsafePath}
        var inputs:[WorkflowPreviewCapture.Input]=[];var total=0
        for case let url as URL in enumerator {
            try Task.checkCancellation();let resource=try url.resourceValues(forKeys:[.isRegularFileKey,.isSymbolicLinkKey]);guard resource.isSymbolicLink != true else{throw WorkflowFileLeaseError.unsafePath};guard resource.isRegularFile==true else{continue}
            let path=String(url.path.dropFirst(replay.path.count+1));if path=="writer.lock"{continue}
            guard path=="latest.json" || chunks.contains(path) || path.hasPrefix("checkpoints/"+name+"/")else{continue}
            let limit=try SelfPlayToolContract.replayLimit(path)
            let raw=try DiagnosticFileIO.read(root:replay,parts:path.components(separatedBy:"/"),limit:limit);total+=raw.count
            guard total<=1_000_000_000,inputs.count<599 else{throw WorkflowFileLeaseError.unsafePath}
            inputs.append(.file(path:"input/resume/"+path,source:url,expectedSHA256:WorkflowArchiveCodec.hash(raw)))
        }
        return inputs
    }
    static func list(root:URL,projectID:String)throws->[Self] {
        guard UUID(uuidString:projectID)?.uuidString==projectID else{throw WorkflowFileLeaseError.unsafePath}
        let directory=root.appendingPathComponent("workflow-operations/\(projectID)");guard FileManager.default.fileExists(atPath:directory.path)else{return []}
        let candidates=try FileManager.default.contentsOfDirectory(at:directory,includingPropertiesForKeys:[]);guard candidates.count<=20000 else{throw WorkflowFileLeaseError.unsafePath}
        return try candidates.compactMap{entry->Self? in
            try Task.checkCancellation();guard UUID(uuidString:entry.lastPathComponent)?.uuidString==entry.lastPathComponent else{return nil}
            if let raw=try? DiagnosticFileIO.read(root:entry,parts:["self-play-record.json"],limit:2_000_000),let record=try? JSONDecoder().decode(Self.self,from:raw),record.id==entry.lastPathComponent,record.document.projectID==projectID,!record.validateOnly{return record}
            // App/worker interruption may precede a terminal callback. Only captured source and
            // an existing atomic boundary qualify; nothing here asserts successful completion.
            guard let requestBytes=try? DiagnosticFileIO.read(root:entry,parts:["request.json"],limit:262_144),let request=try? JSONDecoder().decode(WorkflowToolRequest.self,from:requestBytes),request.toolMode == .selfPlay,request.projectID==projectID,request.operationID==entry.lastPathComponent,
                  let source=try? DiagnosticFileIO.read(root:entry,parts:["input","document.json"],limit:2_000_000),let document=try? WorkflowCodec.decode(source),document.projectID==projectID,
                  request.inputFiles.contains(where:{$0.path=="input/document.json" && $0.sha256==WorkflowArchiveCodec.hash(source)}),
                  let innerBytes=try? DiagnosticFileIO.read(root:entry,parts:["input","self-play.json"],limit:2_000_000),let inner=try? JSONDecoder().decode([String:JSONValue].self,from:innerBytes),inner["validateOnly"] == .bool(false),
                  ((try? DiagnosticFileIO.read(root:entry,parts:["artifacts","replay","latest.json"],limit:128)) != nil || (try? DiagnosticFileIO.read(root:entry,parts:["input","campaign-grant.json"],limit:2000000)) != nil),
                  let trainAddress=inner["trainAddress"].flatMap({try? JSONDecoder().decode(WorkflowNodeAddress.self,from:JSONEncoder().encode($0))}),let node=document.nodes.first(where:{$0.address==trainAddress && SelfPlayConfiguration.isSelfPlay($0)}) else{return nil}
            let date=(try? entry.resourceValues(forKeys:[.contentModificationDateKey]).contentModificationDate) ?? Date.distantPast
            return .init(id:request.operationID,address:node.address,document:document,files:[],finishedAt:date,validateOnly:false,completed:false)
        }.sorted{$0.finishedAt>$1.finishedAt}.prefix(100).map{$0}
    }
}
