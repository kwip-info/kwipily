import Foundation
import Combine
import KwipMLCore

/// One immutable captured game operation. Source compilation and checkpoint verification stay off main.
@MainActor final class GameEpisodeSession:ObservableObject {
    struct Environment {let root:URL;let accounting:URL;let runtime:RuntimeLocation;let policy:ResolvedExecutionPreferences}
    let id=UUID()
    let document:WorkflowDocument
    let address:WorkflowNodeAddress
    let validateOnly:Bool
    @Published private(set) var isBusy=false
    @Published private(set) var message=""
    @Published private(set) var artifacts:[String:TableViewQuerySession.Artifact]=[:]
    private(set) var ports:[String:JSONValue]=[:]
    @Published private(set) var summary:[String:JSONValue]=[:]
    private var task:Task<Void,Never>?
    private var worker:WorkflowWorkerHandle?
    private var token:UUID?
    private var cancelled=false
    private(set) var requestIdentity=""
    private(set) var boundReferences:[WorkflowNodeAddress:WorkflowFrozenModelReference]=[:]
    private var files:[WorkflowArchiveFile]?
    let environment:Environment
    private let reservation:WorkflowExecutionReservation
    init(document:WorkflowDocument,address:WorkflowNodeAddress,validateOnly:Bool,environment:Environment,reservation:WorkflowExecutionReservation){self.document=document;self.address=address;self.validateOnly=validateOnly;self.environment=environment;self.reservation=reservation}
    func start(modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,references:[WorkflowNodeAddress:WorkflowFrozenModelReference],paired:[WorkflowNodeAddress:SelfPlayRecord]=[:],isCurrent:@escaping()->Bool,onFinish:@escaping()->Void) {
        guard !isBusy,let token=reservation.acquire() else {message="Wait for the current workspace operation.";return}
        if let node=document.nodes.first(where:{$0.address==address}),case .gameSession(let raw)=node.payload,let config=try? GameSessionConfiguration(raw) {
            for role in ["A","B"] where config.source(role)=="saved" {
                for edge in document.edges where edge.to.address==address && edge.to.port=="policy"+role {if let ref=references[edge.from.address]{boundReferences[edge.from.address]=ref}}
            }
        }
        self.token=token;isBusy=true;message=validateOnly ? "Validating rules and model bindings…":"Capturing frozen game policies…"
        let document=self.document,address=self.address,validateOnly=self.validateOnly,env=environment
        task=Task { [self] in
            do {
                let inputs=try await Task.detached(priority:.userInitiated) {()->[WorkflowPreviewCapture.Input] in
                    try Task.checkCancellation()
                    let plan=try GameSessionPlan.compile(document,address:address,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
                    var policies:[String:JSONValue]=[:],files:[WorkflowPreviewCapture.Input]=[]
                    for (role,model) in plan.canvasPolicies {
                        var p:[String:JSONValue]=["graph":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(model.graph)),"sourceIdentity":.string(model.semanticSHA256)]
                        if let recipe=plan.initializations[role] {p["initialization"]=recipe}
                        policies[role] = .object(p)
                    }
                    let retained=try SelfPlayRecord.list(root:env.root,projectID:document.projectID)
                    let privateBindings=(try? DiagnosticFileIO.read(root:env.root,parts:["self-play-bindings",document.projectID,"bindings.json"],limit:1_000_000)).flatMap{try? JSONDecoder().decode([WorkflowNodeAddress:String].self,from:$0)} ?? [:]
                    for (role,node) in plan.savedPolicies {
                        let sourceNode=document.nodes.first{$0.address==node}
                        let direct=sourceNode.map(SelfPlayConfiguration.isSelfPlay) == true
                        let selected=paired[node] ?? retained.first{record in references[node]==nil && record.completed && (direct ? record.address==node:record.id==privateBindings[node])}
                        if let record=selected {
                            if direct,!WorkflowOutputConstruction.matches(document,captured:record.document,provider:node){throw WorkflowArchiveError("The connected Self-play model changed; train it again or select its recorded checkpoint through Load model.")}
                            _=try record.verify(root:env.root);let base=try record.directory(env.root)
                            let inner=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:base,parts:["input","self-play.json"],limit:2_000_000))
                            guard let graph=inner["graph"],let checkpoint=record.files.first(where:{$0.path=="artifacts/model.pt"})else{throw WorkflowFileLeaseError.unsafePath}
                            policies[role] = .object(["graph":graph,"sourceIdentity":.string(checkpoint.sha256),"checkpointSHA256":.string(checkpoint.sha256)])
                            files.append(.file(path:"input/"+role+".pt",source:base.appendingPathComponent(checkpoint.path),expectedSHA256:checkpoint.sha256));continue
                        }
                        guard let reference=references[node],reference.projectID==document.projectID,reference.hasValidPins,let directoryID=UUID(uuidString:reference.directoryID),case .verified(let parent)=WorkflowHistoryReader.load(.init(projectID:reference.projectID,directoryID:directoryID),root:env.root),parent.outcome.status == .completed,parent.outcome.runID==reference.runID,parent.outcome.requestIdentity==reference.requestIdentity,parent.outcome.modelManifestSHA256==reference.modelManifestSHA256,parent.outcome.preparationManifestSHA256==reference.preparationManifestSHA256,let checkpoint=parent.outcome.artifacts.first(where:{$0.path=="training/artifacts/checkpoint.pt"}),checkpoint.sha256==reference.checkpointSHA256 else {throw WorkflowArchiveError("Select a verified completed canvas-model checkpoint for Policy \(role). Imported task models require a separate game adapter.")}
                        let base=env.root.appendingPathComponent("workflow-runs/\(reference.projectID)/\(reference.directoryID)")
                        let outcome=try DiagnosticFileIO.read(root:base,parts:["outcome.json"],limit:256_000)
                        guard WorkflowArchiveCodec.hash(outcome)==reference.outcomeSHA256 else {throw WorkflowArchiveError("Saved model outcome changed.")}
                        policies[role] = .object(["graph":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(parent.plan.model.graph)),"sourceIdentity":.string(parent.plan.model.semanticSHA256),"checkpointSHA256":.string(checkpoint.sha256)])
                        files.append(.file(path:"input/"+role+".pt",source:base.appendingPathComponent(checkpoint.path),expectedSHA256:checkpoint.sha256))
                    }
                    let request:[String:JSONValue]=["schemaVersion":1,"definition":.object(plan.definition),"configuration":.object(plan.configuration.values),"policies":.object(policies),"validateOnly":.bool(validateOnly)]
                    files.append(.bytes(path:"input/session.json",data:try WorkflowArchiveCodec.encode(request)))
                    return files
                }.value
                try Task.checkCancellation();guard isCurrent(),!cancelled else {throw CancellationError()}
                let window=try ExecutionBudgetWindow(policy:env.policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
                worker=try await WorkflowCapturedToolExecution.launch(mode:.gameSession,workspace:env.root,accountingRoot:env.accounting,projectID:document.projectID,runtime:env.runtime,window:window,operationID:id.uuidString,storageLimitBytes:env.policy.storageLimitBytes,inputs:{_ in inputs},arguments:{["--game-session",$0.appendingPathComponent("input/session.json").path]},retainInputsOnSuccess:true,onLine:{[weak self] line in
                    guard let self,let raw=line.data(using:.utf8),let e=try? JSONDecoder().decode([String:JSONValue].self,from:raw) else{return}
                    if e["event"] == .string("error") {self.message=e["message"]?.stringValue ?? "Game session failed."}
                    if e["event"] == .string("gameSessionProgress") {self.message="Episodes \(e["completed"]?.integerValue ?? 0) / \(e["requested"]?.integerValue ?? 0) · \(e["moves"]?.integerValue ?? 0) moves"}
                    if e["event"] == .string("gameSessionCompleted"),case .array(let values)=e["files"] {self.requestIdentity=e["requestIdentity"]?.stringValue ?? "";self.files=try? values.map{try JSONDecoder().decode(WorkflowArchiveFile.self,from:JSONEncoder().encode($0))}}
                },onExit:{[self] code,detail,output in
                    defer {finish();onFinish()}
                    guard code==0,!cancelled,let output,let files else {if cancelled {message="Game session cancelled; no completed tables published."} else if !detail.isEmpty {message=detail};return}
                    do {
                        let result=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:output,parts:["admission.json"],limit:64_000))
                        if case .object(let s)=result["summary"] {summary=s};if case .object(let p)=result["ports"] {ports=p}
                        for port in ["summary","episodes","moves"] {
                            guard let f=files.first(where:{$0.path=="artifacts/"+port+".table.json"}) else {throw WorkflowFileLeaseError.unsafePath}
                            artifacts[port] = .init(indexURL:output.appendingPathComponent(port+".table.json"),indexSHA256:f.sha256,sourceSHA256:f.sha256,inputKind:"table")
                        }
                        let record=GameEpisodeRecord(id:id.uuidString,address:address,document:document,files:files,finishedAt:Date(),validateOnly:validateOnly)
                        try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:output.deletingLastPathComponent(),parts:["game-session-record.json"],limit:2_000_000)
                        message=validateOnly ? "Bindings validated · no episodes run.":"Completed · \(summary["terminal"]?.integerValue ?? 0) terminal · \(summary["truncated"]?.integerValue ?? 0) truncated · \(summary["failed"]?.integerValue ?? 0) failed. Frozen policies unchanged."
                    } catch {artifacts=[:];message="Result verification failed: "+error.localizedDescription}
                })
            } catch {message=cancelled ? "Game session cancelled.":error.localizedDescription;finish();onFinish()}
        }
    }
    private func finish(){isBusy=false;worker=nil;task=nil;if let token {reservation.release(token)};token=nil}
    func cancel(){guard isBusy else{return};cancelled=true;message="Cancelling game session…";if let worker {worker.cancel()}else{task?.cancel()}}
}
