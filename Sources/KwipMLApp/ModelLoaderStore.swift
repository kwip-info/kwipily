import Foundation
import AppKit
import Combine
import Security
import KwipMLCore
import KwipMLNetworking

struct ModelHubSnapshot:Equatable {
    struct File:Equatable {let path:String;let bytes:Int;let sha256:String?}
    let repository:String;let requestedRevision:String;let revision:String;let subfolder:String;let files:[File]
    static func decode(_ data:Data,configuration:LoadModelConfiguration)throws->Self {
        guard data.count<=2_000_000,let raw=try JSONSerialization.jsonObject(with:data) as? [String:Any],let sha=raw["sha"] as? String,sha.count==40,sha.allSatisfy({$0.isHexDigit && !$0.isUppercase}),let siblings=raw["siblings"] as? [[String:Any]],siblings.count<=10000 else {throw WorkflowArchiveError("Hub returned invalid or oversized model metadata.")}
        let prefix=configuration.subfolder.isEmpty ? "":configuration.subfolder+"/"
        var files:[File]=[],total=0
        for row in siblings {
            guard let path=row["rfilename"] as? String,path.hasPrefix(prefix) else {continue}
            let leaf=String(path.dropFirst(prefix.count))
            guard ModelHubRequest.allowedAssetFile(leaf) else {continue}
            guard let size=row["size"] as? Int,size>0,size<=2_000_000_000 else {throw WorkflowArchiveError("Model file size is unavailable or exceeds admission.")}
            let hash=(row["lfs"] as? [String:Any])?["sha256"] as? String
            if let hash, !WorkflowArchiveCodec.isHash(hash) {throw WorkflowFileLeaseError.unsafePath}
            total+=size;guard total<=2_010_000_000,files.count<128 else {throw WorkflowArchiveError("Selected model exceeds the 2 GB /128-file admission limit.")}
            files.append(.init(path:path,bytes:size,sha256:hash))
        }
        let typed=files.contains(where:{$0.path==prefix+"rl_agent_config.json"})
        let required:Set<String>=typed ? ["rl_agent_config.json","encoder/config.json","tokenizer/tokenizer.json","tokenizer/tokenizer_config.json"]:["config.json",((raw["config"] as? [String:Any])?["architectures"] as? [String]) == ["ASTForAudioClassification"] ? "preprocessor_config.json":"tokenizer.json"]
        guard required.allSatisfy({name in files.contains{$0.path==prefix+name}}),files.contains(where:{$0.path.hasSuffix(".safetensors")}) else {throw WorkflowArchiveError("This snapshot lacks safe weights or its task-matched tokenizer/audio processor. Repository code and pickle weights are not supported.")}
        return .init(repository:configuration.repository,requestedRevision:configuration.revision,revision:sha,subfolder:configuration.subfolder,files:files.sorted{$0.path<$1.path})
    }
}

/// Only this narrow inspector observes progress. Canvas receives semantic binding
/// changes at completion, never per-byte/per-token redraw notifications.
@MainActor final class ModelLoaderStore:ObservableObject {
    let owner:WorkflowAuthoringStore
    let root:URL
    let accountingRoot:URL?
    let bindings:WorkflowPrivateTrainBindings<ModelAssetReference>
    @Published private(set) var isBusy=false
    @Published private(set) var fineTuningInspectedConfiguration:[String:JSONValue]?
    func clearFineTuningInspection(){fineTuningPreview=nil;fineTuningInspectedConfiguration=nil}
    @Published private(set) var fineTuningPreview:[String:JSONValue]?
    @Published private(set) var trainingPauseRequested=false
    func pauseFineTuning(){guard isBusy,trainingAddress != nil else{return};trainingPauseRequested=true;operation?.pauseAfterEpoch()}
    @Published private(set) var importedResumes:[ModelImportedResume]=[]
    @Published private(set) var fineTuningRecoveries:[ModelFineTuningRecovery]=[]
    @Published private(set) var evaluationRecords:[ModelCausalEvaluationRecord]=[]
    @Published private(set) var evaluationSummary:[String:JSONValue]?
    private(set) var evaluationTableMetadata:[String:JSONValue]?
    private(set) var currentEvaluation:ModelCausalEvaluationRecord?
    @Published private(set) var trainingRecords:[ModelFineTuningRecord]=[]
    @Published private(set) var trainingProgress=TrainingProgressLedger()
    @Published private(set) var trainingMetrics:[EpochMetric]=[]
    @Published private(set) var trainingAddress:WorkflowNodeAddress?
    @Published private(set) var trainingRunID:String?
    @Published private(set) var trainingPhase="idle"
    private var trainingDocuments:[String:WorkflowDocument]=[:]
    private var trainingReloadGeneration=0
    private var standaloneOperation=false
    var blocksWorkspaceOperations:Bool {isBusy && standaloneOperation}
    @Published private(set) var activeAddress:WorkflowNodeAddress?
    @Published private(set) var status=""
    @Published private(set) var resolved:[WorkflowNodeAddress:ModelHubSnapshot]=[:]
    @Published private(set) var tokenPreview:[WorkflowNodeAddress:[String:JSONValue]]=[:]
    @Published private(set) var generationRecord:ModelAssetGenerationRecord?
    private(set) var generationMetadata:[String:JSONValue]?
    private(set) var generationRuntime:RuntimeLocation?
    private(set) var failures:[WorkflowNodeAddress:(payload:WorkflowNodePayload,message:String)]=[:]
    private(set) var statusAddress:WorkflowNodeAddress?
    func message(for address:WorkflowNodeAddress)->String {statusAddress==address ? status:""}
    func failure(_ address:WorkflowNodeAddress)->String? {guard let value=failures[address],owner.document.nodes.first(where:{$0.address==address})?.payload==value.payload else {return nil};return value.message}
    var onBindingChange:(()->Void)?
    private var task:Task<Void,Never>?
    private var operation:ModelAssetOperation?
    private var inspectionAddress:WorkflowNodeAddress?
    func cancelFineTuningInspection(_ address:WorkflowNodeAddress){if inspectionAddress==address {cancel()}}
    private let client=NetworkBrokerClient()
    init(owner:WorkflowAuthoringStore,root:URL,accountingRoot:URL?,ephemeralSelections:Bool=false)throws {self.owner=owner;self.root=root;self.accountingRoot=accountingRoot;bindings=try .init(owner:owner,directory:root,ephemeral:ephemeralSelections);if bindings.bindings.values.contains(where:{$0.metadata["trainingOperation"] == .bool(true)}) {reloadTraining()}}
    func configuration(_ address:WorkflowNodeAddress)throws->LoadModelConfiguration {guard let node=owner.document.nodes.first(where:{$0.address==address}),case .frozenModel(let values)=node.payload else {throw WorkflowArchiveError("Choose Load model.")};return try .decode(values)}
    func reference(_ address:WorkflowNodeAddress)->ModelAssetReference? {
        guard let reference=bindings.reference(for:address) else {return nil}
        if let node=owner.document.nodes.first(where:{$0.address==address}),FineTuningConfiguration.isExternal(node) {
            guard let captured=trainingDocuments[reference.directoryID],WorkflowOutputConstruction.matches(owner.document,captured:captured,provider:address),let edge=owner.document.edges.first(where:{$0.to.address==address && $0.to.port=="model"}),bindings.reference(for:edge.from.address)?.identity==reference.metadata["baseIdentity"]?.stringValue else {return nil}
            return reference
        }
        guard let c=try? configuration(address),reference.matches(c) else {return nil};return reference
    }
    func show(_ message:String,for address:WorkflowNodeAddress) {statusAddress=address;status=message;onBindingChange?()}
    func cancel() {task?.cancel();operation?.cancel()}
    func invalidate() {cancel();onBindingChange=nil}
    @discardableResult private func start(_ address:WorkflowNodeAddress,onTerminal:@escaping(Bool)->Void={_ in},reservation:WorkflowExecutionReservation?=nil,work:@escaping()async throws->Void)->Bool {
        guard !isBusy,owner.isCurrent(owner.lease()),!owner.isBusy,!owner.hasPendingInspectorEdits else {status="Apply pending changes and wait for the current operation.";return false}
        // Claim before publishing busy state or yielding to result readers.
        let token=reservation?.acquire()
        if reservation != nil,token == nil {status="Wait for the current workspace operation.";return false}
        let capturedPayload=owner.document.nodes.first(where:{$0.address==address})?.payload
        if case .generateText=capturedPayload {standaloneOperation=false} else if case .train=capturedPayload {standaloneOperation=false} else if case .evaluateTest=capturedPayload {standaloneOperation=false} else {standaloneOperation=true}
        isBusy=true;activeAddress=address;statusAddress=address;failures.removeValue(forKey:address);onBindingChange?()
        task=Task { [self] in
            var succeeded=false
            defer {
                isBusy=false;standaloneOperation=false;activeAddress=nil;operation=nil;task=nil
                if let token {reservation?.release(token)}
                onBindingChange?();onTerminal(succeeded)
            }
            do {try await work();succeeded=true} catch {
                if Task.isCancelled || error is CancellationError {status="Model operation cancelled."}
                else if case NetworkBrokerFailure.signInRequired=error {status="Hub access was refused. Check the repository and its named read credential."}
                else if case NetworkBrokerFailure.unavailable=error {status="Hub request failed or timed out. Check the network and repository, then resolve again."}
                else {status=error.localizedDescription}
                if case .train=capturedPayload {
                    trainingPhase=Task.isCancelled || error is CancellationError ? "cancelled":"failed"
                    if !owner.runHistoryPolicy.metricsAndDiagnostics {trainingProgress = .init();trainingMetrics=[]}
                }
                if !Task.isCancelled,!(error is CancellationError),let capturedPayload,owner.document.nodes.first(where:{$0.address==address})?.payload==capturedPayload {failures[address]=(capturedPayload,status)}
            }
        }
        return true
    }
    func resolve(_ address:WorkflowNodeAddress,bearer:String?,maximumSeconds:Double=45) {
        start(address) { [self] in
            let configuration=try configuration(address),lease=owner.lease()
            guard configuration.source == .huggingFace else {throw WorkflowArchiveError("Choose the Hugging Face source.")}
            status="Resolving repository revision…"
            let request=ModelHubRequest(action:.resolve,repository:configuration.repository,revision:configuration.revision,bearer:bearer)
            let data=try await ModelHubBudget.read(remainingSeconds:maximumSeconds){try await NetworkBrokerClient().modelHub(request)}
            let snapshot=try ModelHubSnapshot.decode(data,configuration:configuration)
            try Task.checkCancellation();guard owner.isCurrent(lease),try self.configuration(address)==configuration else {throw CancellationError()}
            resolved[address]=snapshot;status="Revision pinned · \(snapshot.revision.prefix(12)) · \(snapshot.files.count) files"
        }
    }
    func chooseLocal(_ address:WorkflowNodeAddress,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences) {
        guard !isBusy else {return}
        if (try? configuration(address).source) == .documentPackage {chooseDocumentPackage(address,policy:policy);return}
        let panel=NSOpenPanel();panel.canChooseFiles=false;panel.canChooseDirectories=true;panel.allowsMultipleSelection=false;panel.prompt="Inspect model";panel.message="Choose an existing safetensors model folder from any location. Include safe weights and the task-matched configuration, tokenizer or audio processor. Laya uses its complete encoder/tokenizer subfolders. GGUF / LM Studio quantized files need a separate adapter and are not supported yet."
        guard panel.runModal() == .OK,let source=panel.url else {return}
        let started=source.startAccessingSecurityScopedResource()
        if FileManager.default.fileExists(atPath:source.appendingPathComponent("kwip-adapter.json").path) {
            let basePanel=NSOpenPanel();basePanel.canChooseFiles=false;basePanel.canChooseDirectories=true;basePanel.allowsMultipleSelection=false;basePanel.prompt="Choose matching base";basePanel.message="This is a LoRA adapter, not a standalone model. Choose its exact original base folder. The app verifies every saved base-file pin before attaching it; nothing is downloaded."
            guard basePanel.runModal() == .OK,let base=basePanel.url else {if started {source.stopAccessingSecurityScopedResource()};return}
            let baseStarted=base.startAccessingSecurityScopedResource()
            start(address) { [self] in
                defer {if started {source.stopAccessingSecurityScopedResource()};if baseStarted {base.stopAccessingSecurityScopedResource()}}
                let lease=owner.lease()
                let inventory=try await Task.detached(priority:.userInitiated) {(try ModelAssetFiles.inventory(base),try ModelAdapterFiles.inventory(source))}.value
                let op=ModelAssetOperation();operation=op;status="Verifying adapter and matching base…"
                let result=try await op.run(action:"attachAdapter",settings:["maximumResidentBytes":.integer(Int64(policy.budget.resources?.maximumResidentBytes ?? 2_000_000_000))],source:base,files:inventory.0,projectID:owner.projectID,root:root,accountingRoot:accountingRoot,runtime:runtime,policy:policy,adapter:source,adapterFiles:inventory.1,isCurrent:{[weak self] in self?.owner.isCurrent(lease)==true},onProgress:{[weak self] in self?.status=$0})
                try await importFolder(result.directory.appendingPathComponent("output/model"),address:address,runtime:runtime,policy:policy)
            }
            if !isBusy {if started {source.stopAccessingSecurityScopedResource()};if baseStarted {base.stopAccessingSecurityScopedResource()}}
            return
        }
        start(address) { [self] in
            defer {if started {source.stopAccessingSecurityScopedResource()}}
            try await importFolder(source,address:address,runtime:runtime,policy:policy)
        }
        if !isBusy,started {source.stopAccessingSecurityScopedResource()}
    }
    func importFolder(_ source:URL,address:WorkflowNodeAddress,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,budgetWindow:ExecutionBudgetWindow?=nil)async throws {
        let window=try budgetWindow ?? ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
        let config=try configuration(address),lease=owner.lease(),revision=bindings.revision
        guard config.source != .savedRun else {throw WorkflowArchiveError("Choose Local folder before importing a model.")}
        let id=UUID().uuidString,cache=root.appendingPathComponent("model-assets/\(owner.projectID)/\(id)")
        let root=self.root,accounting=accountingRoot,project=owner.projectID
        status="Snapshotting selected model files…"
        let worker=Task.detached(priority:.userInitiated) {()->([WorkflowArchiveFile],WorkflowFrozenSnapshotDirectory,WorkflowStorageAdmission.Reservation?) in
            let files=try ModelAssetFiles.inventory(source),size=files.reduce(Int64(16_000_000)){$0+Int64($1.bytes)}
            let reservation:WorkflowStorageAdmission.Reservation?
            if let limit=policy.storageLimitBytes {reservation=try WorkflowStorageAdmission(workspace:root,accountingRoot:accounting).reserve(scopes:[.init(namespace:.modelAssets,components:[project,id])],peakBytes:size,limitBytes:limit)} else {reservation=nil}
            do {
                try FileManager.default.createDirectory(at:cache,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
                let ownership=try WorkflowFrozenSnapshotDirectory(directory:cache,removeOnDeinit:false)
                do {try ModelAssetFiles.copy(files,from:source,to:cache);return (files,ownership,reservation)} catch {ownership.remove();throw error}
            } catch {reservation?.close();throw error}
        }
        let (files,cacheOwnership,reservation)=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        var retained=false
        defer {reservation?.close();if !retained {cacheOwnership.remove()}}
        guard owner.isCurrent(lease) else {throw CancellationError()}
        status="Verifying architecture, tokenizer and weights…"
        let op=ModelAssetOperation();operation=op
        let result=try await op.run(action:"inspect",settings:[:],source:cache,files:files,projectID:project,root:root,accountingRoot:accounting,runtime:runtime,policy:policy,budgetWindow:window,isCurrent:{[weak self] in self?.owner.isCurrent(lease)==true},onProgress:{[weak self] in self?.status=$0})
        let metadata=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:result.directory,parts:["output","model-metadata.json"],limit:2_000_000))
        var reference=ModelAssetReference(projectID:project,directoryID:id,metadata:metadata,sourceConfiguration:config.values)
        guard reference.hasValidPins,reference.files==files.filter({$0.path != "kwip-manifest.json"}),reference.identity == (try ModelAssetFiles.sourceIdentity(reference.files)) else {throw WorkflowFileLeaseError.unsafePath}
        let existing=bindings.bindings.values.first(where:{$0.identity==reference.identity})
        if let existing {reference = .init(projectID:project,directoryID:existing.directoryID,metadata:metadata,sourceConfiguration:config.values)}
        guard owner.isCurrent(lease),try configuration(address)==config,bindings.setReference(reference,for:address,lease:lease,expectedRevision:revision) else {throw WorkflowArchiveError(bindings.error ?? "Model selection changed during inspection.")}
        retained = existing == nil
        status="Model verified · \(metadata["architecture"]?.stringValue ?? "causal model") · tokenizer matched";onBindingChange?()
    }
    func download(_ address:WorkflowNodeAddress,bearer:String?,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences) {
        start(address) { [self] in
            let window=try ExecutionBudgetWindow(policy:policy.budget,startedAtUptime:ProcessInfo.processInfo.systemUptime)
            let config=try configuration(address),lease=owner.lease()
            guard let snapshot=resolved[address],snapshot.repository==config.repository,snapshot.requestedRevision==config.revision,snapshot.subfolder==config.subfolder else {throw WorkflowArchiveError("Resolve the current repository revision first.")}
            let temporary=root.appendingPathComponent("model-download-staging/"+UUID().uuidString)
            let demand=snapshot.files.reduce(Int64(0)){$0+Int64($1.bytes)}
            let storageRoot=root,accounting=accountingRoot
            let preparation=Task.detached(priority:.utility) {()->WorkflowStorageAdmission.Reservation? in
                let admission:WorkflowStorageAdmission.Reservation?
                if let limit=policy.storageLimitBytes {admission=try WorkflowStorageAdmission(workspace:storageRoot,accountingRoot:accounting).reserve(scopes:[.init(namespace:.modelDownloadStaging,components:[temporary.lastPathComponent])],peakBytes:demand,limitBytes:limit)} else {admission=nil}
                do {try Task.checkCancellation();try FileManager.default.createDirectory(at:temporary,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700]);return admission}
                catch {admission?.close();throw error}
            }
            let admission=try await withTaskCancellationHandler(operation:{try await preparation.value},onCancel:{preparation.cancel()})
            defer {admission?.close()}
            let held=try WorkflowFrozenSnapshotDirectory(directory:temporary);defer {held.remove()}
            let scratch=try WorkflowScratchCleanup.register(directory:temporary);defer {scratch.close()}
            for (index,file) in snapshot.files.enumerated() {
                status="Downloading · \(index+1)/\(snapshot.files.count) files"
                let prefix=snapshot.subfolder.isEmpty ? "":snapshot.subfolder+"/"
                let relative=String(file.path.dropFirst(prefix.count))
                guard ModelHubRequest.allowedAssetFile(relative) else{throw WorkflowFileLeaseError.unsafePath}
                let destination=temporary.appendingPathComponent(relative)
                try FileManager.default.createDirectory(at:destination.deletingLastPathComponent(),withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
                guard FileManager.default.createFile(atPath:destination.path,contents:nil,attributes:[.posixPermissions:0o600]) else {throw WorkflowFileLeaseError.unsafePath}
                let output=try FileHandle(forWritingTo:destination)
                do {
                    var offset:Int64=0
                    while offset<Int64(file.bytes) {
                        try Task.checkCancellation();guard owner.isCurrent(lease) else {throw CancellationError()}
                        let request=ModelHubRequest(action:.file,repository:snapshot.repository,revision:snapshot.revision,file:file.path,offset:offset,total:Int64(file.bytes),bearer:bearer)
                        let remaining=try window.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)
                        let bytes=try await ModelHubBudget.read(remainingSeconds:remaining){try await NetworkBrokerClient().modelHub(request)}
                        try await Task.detached(priority:.utility) {try output.write(contentsOf:bytes)}.value;offset+=Int64(bytes.count)
                    }
                    try await Task.detached(priority:.utility) {
                        try output.synchronize();try output.close()
                        if let sha=file.sha256 {try WorkflowPreviewToolContract.verifyFile(file:.init(path:relative,bytes:file.bytes,sha256:sha),directory:temporary,limit:file.bytes)}
                    }.value
                } catch {try? output.close();throw error}
            }
            try await importFolder(temporary,address:address,runtime:runtime,policy:policy,budgetWindow:window)
        }
    }
    func preview(_ address:WorkflowNodeAddress,text:String,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences) {
        start(address) { [self] in
            let config=try configuration(address),lease=owner.lease()
            guard let source=reference(address) else {throw WorkflowArchiveError("Load and verify a model first.")}
            let op=ModelAssetOperation();operation=op;status="Tokenizing bounded sample…"
            let settings:[String:JSONValue]=["texts":.array([.string(text)]),"maximumTokens":.integer(Int64(config.maximumTokens)),"truncation":.string(config.truncation),"padding":.string(config.padding),"paddingSide":.string(config.paddingSide)]
            let result=try await op.run(action:"tokenize",settings:settings,source:source.directory(root:root),files:source.files,projectID:owner.projectID,root:root,accountingRoot:accountingRoot,runtime:runtime,policy:policy,isCurrent:{[weak self] in self?.owner.isCurrent(lease)==true},onProgress:{[weak self] in self?.status=$0})
            var packet=try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:result.directory,parts:["output","tokens.json"],limit:8_000_000));packet["sampleText"] = .string(text);packet["configuration"] = .object(config.values);tokenPreview[address]=packet;status="Token preview ready."
        }
    }
    @discardableResult func exportModel(_ reference:ModelAssetReference,address:WorkflowNodeAddress,shardBytes:Int,access:WorkflowModelDestinationAccess,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,isCurrent:@escaping()->Bool,onTerminal:@escaping(Bool)->Void={_ in},onSaved:@escaping(URL)->Void)->Bool {
        guard !isBusy else {access.close();return false}
        let started=start(address,onTerminal:onTerminal) { [self] in
            defer {access.close()}
            try access.check();status="Preparing safe model export…"
            let op=ModelAssetOperation();operation=op
            let result=try await op.run(action:"export",settings:["shardBytes":.integer(Int64(shardBytes))],source:reference.directory(root:root),files:reference.files,projectID:owner.projectID,root:root,accountingRoot:accountingRoot,runtime:runtime,policy:policy,isCurrent:isCurrent,onProgress:{[weak self] in self?.status=$0})
            let files=result.files.filter{$0.path.hasPrefix("output/model/")}.map {WorkflowArchiveFile(path:String($0.path.dropFirst(13)),bytes:$0.bytes,sha256:$0.sha256)}
            guard !files.isEmpty else {throw WorkflowArchiveError("Model export produced no verified files.")}
            let work=Task.detached(priority:.userInitiated) {()->WorkflowPendingExportPublication in
                let pending=try WorkflowPendingExportPublication(destination:access.destination,directory:true)
                do {for file in files {try Task.checkCancellation();try pending.writeModelFile(file,from:result.directory.appendingPathComponent("output/model"))};return pending} catch {pending.close();throw error}
            }
            let pending=try await withTaskCancellationHandler(operation:{try await work.value},onCancel:{work.cancel()})
            defer {pending.close()}
            try Task.checkCancellation();try access.check();guard isCurrent() else {throw CancellationError()}
            try pending.commit();status=reference.metadata["task"] == .string("typedDecisions") ? "Typed-decision package saved · model, tokenizer and questions":"Hugging Face model saved · complete model and tokenizer";onSaved(pending.destination)
        }
        if !started {access.close()};return started
    }

    func generationIsCurrent(_ endpoint:WorkflowEndpoint)->Bool {
        guard let record=generationRecord,endpoint.address==record.address,endpoint.port=="table",generationMetadata != nil,reference(record.sourceAddress)==record.reference else {return false}
        return WorkflowOutputConstruction.matches(owner.document,captured:record.document,provider:record.address)
    }
    func restoreGeneration(runtime:RuntimeLocation) {
        guard generationRecord==nil,task==nil else {return}
        let root=self.root,project=owner.projectID,lease=owner.lease()
        Task { [weak self] in
            do {
                let work=Task.detached(priority:.utility) {()->(ModelAssetGenerationRecord,[String:JSONValue])? in
                    guard let data=try DiagnosticFileIO.readIfPresent(root:root,parts:["workflow-model-assets",project,"latest-generation.json"],limit:2_000_000) else {return nil}
                    let record=try JSONDecoder().decode(ModelAssetGenerationRecord.self,from:data)
                    guard record.document.projectID==project else {throw WorkflowFileLeaseError.unsafePath}
                    return (record,try record.verify(root:root))
                }
                guard let result=try await work.value,let self,self.owner.isCurrent(lease),self.generationRecord==nil else {return}
                self.generationRecord=result.0;self.generationMetadata=result.1.filter{$0.key != "rows"};self.generationRuntime=runtime;self.onBindingChange?()
            } catch {self?.status="Imported-model generation could not be restored: "+error.localizedDescription}
        }
    }
    @discardableResult func generate(address:WorkflowNodeAddress,sourceAddress:WorkflowNodeAddress,configuration:TextGenerationConfiguration,promptArtifact:TableViewQuerySession.Artifact?,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,reservation:WorkflowExecutionReservation,onTerminal:@escaping(Bool)->Void)->Bool {
        return start(address,onTerminal:onTerminal,reservation:reservation) { [self] in
            generationRecord=nil;generationMetadata=nil
            let lease=owner.lease(),document=owner.document
            guard let reference=reference(sourceAddress) else {throw WorkflowArchiveError("Load and verify the connected model first.")}
            // Token limits may change without reimporting immutable model weights.
            // Read this captured canvas configuration, not its original import defaults.
            let modelConfig:LoadModelConfiguration
            if let node=owner.document.nodes.first(where:{$0.address==sourceAddress}),case .frozenModel(let values)=node.payload {
                modelConfig=try LoadModelConfiguration.decode(values)
            } else {modelConfig=try LoadModelConfiguration.decode(reference.sourceConfiguration)}
            var extra:[String:Data]=[:]
            if configuration.tableMode {
                guard let artifact=promptArtifact,artifact.inputKind=="table" else {throw WorkflowArchiveError("Run and capture the complete text prompt table first.")}
                let bytes=try await Task.detached(priority:.userInitiated) {()->Data in
                    let data=try DiagnosticFileIO.read(root:artifact.indexURL.deletingLastPathComponent(),parts:[artifact.indexURL.lastPathComponent],limit:8_000_000)
                    guard WorkflowArchiveCodec.hash(data)==artifact.sourceSHA256 else {throw WorkflowFileLeaseError.unsafePath};return data
                }.value
                extra["input/prompts.table.json"]=bytes
            }
            var full=configuration.values.filter{["promptMode","prompt","promptColumn","textColumn","idsColumn","countColumn","stopColumn"].contains($0.key)}
            full["settings"] = .object(configuration.workerSettings)
            let settings:[String:JSONValue]=["configuration":.object(full),"maximumTokens":.integer(Int64(modelConfig.maximumTokens)),"device":configuration.values["device"]!]
            status="Loading imported model for generation…"
            let op=ModelAssetOperation();operation=op
            do {
                let result=try await op.run(action:"generate",settings:settings,source:reference.directory(root:root),files:reference.files,projectID:owner.projectID,root:root,accountingRoot:accountingRoot,runtime:runtime,policy:policy,extraInputs:extra,isCurrent:{[weak self] in guard let self else {return false};return self.owner.isCurrent(lease) && self.reference(sourceAddress)==reference},onProgress:{[weak self] in self?.status=$0})
                let record=ModelAssetGenerationRecord(schemaVersion:1,operationID:result.operationID,address:address,document:document,sourceAddress:sourceAddress,reference:reference,finishedAt:Date(),files:result.files)
                let work=Task.detached(priority:.userInitiated) {()->[String:JSONValue] in
                    let table=try record.verify(root:self.root)
                    try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:result.directory,parts:["model-generation-outcome.json"],limit:2_000_000)
                    try DiagnosticFileIO.writeAtomic(WorkflowArchiveCodec.encode(record),root:self.root,parts:["workflow-model-assets",document.projectID,"latest-generation.json"])
                    return table
                }
                let table=try await work.value;guard owner.isCurrent(lease) else {throw CancellationError()}
                generationRuntime=runtime;generationMetadata=table.filter{$0.key != "rows"};generationRecord=record;status="Generation completed.";onBindingChange?()
            } catch {throw error}
        }
    }

    func fineTuningSettings(address:WorkflowNodeAddress,config:[String:JSONValue],draft:[String:JSONValue]?=nil,policy:ResolvedExecutionPreferences)throws->[String:JSONValue] {
        guard let original=try FineTuningConfiguration.decode(config) else{throw WorkflowArchiveError("Choose imported fine-tuning.")}
        var settings=try FineTuningConfiguration(draft ?? original.values).values
        for key in ["startMode","resumeRecord","resumeEpoch"] {settings.removeValue(forKey:key)}
        let run=try WorkflowTrainRunSetup.configuration(config,schemaVersion:owner.document.schemaVersion) ?? [:]
        let device=run["device"]?.stringValue ?? "auto"
        settings["device"] = .string(settings["task"] == .string("typedDecisions") && device=="auto" ? "cpu":device);settings["epochs"]=run["epochs"] ?? 1;settings["batchSize"]=run["batchSize"] ?? 8;settings["seed"]=run["seed"] ?? 0
        settings["maximumResidentBytes"] = .integer(Int64(policy.budget.resources?.maximumResidentBytes ?? (settings["task"] == .string("typedDecisions") ? 16_000_000_000:2_000_000_000)))
        settings["optimizer"]=config["optimizer"] ?? .object(["schemaVersion":1,"componentID":.string("optimizer_adamw"),"configurationVersion":1,"config":.object(["lr":run["learningRate"] ?? 0.001])])
        settings["trainingControl"]=try run["trainingControl"] ?? ( JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(TrainingControlPolicy.factory)))
        settings["reporting"]=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(TrainingReportingPolicy.current))
        let capture = policy.sampleCapture.applyingHistory(outputs:owner.runHistoryPolicy.componentOutputs,
            diagnostics:owner.runHistoryPolicy.metricsAndDiagnostics,
            requiresPredictionTable:owner.document.edges.contains{$0.from.address==address && $0.from.port=="predictions"})
        settings["capturePredictions"] = .bool(capture.predictionRows)
        settings["scheduler"]=config["scheduler"] ?? .null
        if config["gradientPolicy"] != nil {throw WorkflowArchiveError("Remove canvas derivative overrides before this imported task run; use its explicit accumulation control.")}
        return settings
    }

    func train(address:WorkflowNodeAddress,training:TableViewQuerySession.Artifact,validation:TableViewQuerySession.Artifact,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,reservation:WorkflowExecutionReservation,budgetWindow:ExecutionBudgetWindow?=nil,onTerminal:@escaping(Bool)->Void) {
        start(address,onTerminal:onTerminal) { [self] in
            guard let token=reservation.acquire() else {throw WorkflowArchiveError("Wait for the current workspace operation.")};defer {reservation.release(token)}
            let lease=owner.lease(),document=owner.document,revision=bindings.revision,historyPolicy=owner.runHistoryPolicy,reporting=TrainingReportingPolicy.current
            defer {if !historyPolicy.metricsAndDiagnostics {trainingProgress = .init()}}
            guard let node=document.nodes.first(where:{$0.address==address}),case .train(let config)=node.payload,let taskConfig=try FineTuningConfiguration.decode(config),let edge=document.edges.first(where:{$0.to.address==address && $0.to.port=="model"}),let loadedBase=reference(edge.from.address),loadedBase.metadata["trainingOperation"] != .bool(true) else {throw WorkflowArchiveError("Connect a verified Load model base before fine-tuning.")}
            guard config["initialization"]==nil,config["trainability"]==nil,config["objective"]==nil,config["datasetAccess"]==nil,config["batching"]==nil else {throw WorkflowArchiveError("Loaded causal models use their model-matched objective and text batches. Remove incompatible canvas initialization, objective or data-reader overrides.")}
            let run=try WorkflowTrainRunSetup.resolve(in:document,train:address).configuration
            guard run["experiment"]==nil,run["preparationCachePolicy"]==nil || run["preparationCachePolicy"] == .string("off"),config["trainingMode"]==nil || config["trainingMode"] == .string("fresh") else {throw WorkflowArchiveError("Use a single fine-tuning run without a canvas checkpoint or preparation-cache override.")}
            var settings=try fineTuningSettings(address:address,config:config,policy:policy)
            let startMode=taskConfig.values["startMode"]?.stringValue ?? "base"
            var base=loadedBase
            var resume:ModelResumeCapture?
            if startMode != "base" {
                guard let id=taskConfig.values["resumeRecord"]?.stringValue else{throw WorkflowArchiveError("Select a recorded same-project fine-tuning boundary.")}
                let root=self.root,epoch=Int(taskConfig.values["resumeEpoch"]?.integerValue ?? 0)
                if startMode=="resume",let imported=importedResumes.first(where:{$0.id==id}) {
                    guard epoch==0 else{throw WorkflowArchiveError("An imported package contains one boundary; select latest (0).")}
                    resume=try await Task.detached{try imported.capture(root:root,base:loadedBase)}.value;settings["resumeMode"] = .string("exact")
                } else if startMode=="resume",let recovery=fineTuningRecoveries.first(where:{$0.id==id && $0.address==address && $0.base.identity==loadedBase.identity}) {
                    resume=try await Task.detached{try recovery.capture(root:root,epoch:epoch)}.value;settings["resumeMode"] = .string("exact")
                } else {
                    guard let prior=trainingRecords.first(where:{$0.id==id && $0.address==address && $0.document.projectID==owner.projectID})else{throw WorkflowArchiveError("Choose a retained same-project training run.")}
                    if startMode=="resume" {
                        guard prior.base.identity==loadedBase.identity else{throw WorkflowArchiveError("Connect the original matching base for exact continuation.")}
                        resume=try await Task.detached{try ModelResumeCapture.capture(prior,root:root,epoch:epoch)}.value;settings["resumeMode"] = .string("exact")
                    }else {
                        var metadata=try await Task.detached{try prior.verify(root:root)}.value;metadata["trainingOperation"] = .bool(true)
                        base=ModelAssetReference(projectID:owner.projectID,directoryID:prior.id,metadata:metadata,sourceConfiguration:loadedBase.sourceConfiguration)
                    }
                }
            }
            guard training.inputKind=="table",validation.inputKind=="table" else {throw WorkflowArchiveError("Prepare complete Training and Validation tables first.")}
            let tables:[ModelTableCapture.Input]=[.init(role:.training,url:training.indexURL,sha256:training.sourceSHA256,retainedSegments:training.retainedSegments),.init(role:.validation,url:validation.indexURL,sha256:validation.sourceSHA256,retainedSegments:validation.retainedSegments)]
            trainingPauseRequested=false;trainingAddress=address;trainingProgress = .init();trainingMetrics=[];trainingRunID=nil;trainingPhase="preparing";status="Loading verified model for fine-tuning…"
            let op=ModelAssetOperation();operation=op
            let initialEpoch=Int(resume?.report["completedEpoch"]?.integerValue ?? 0)
            let result:ModelAssetOperation.Result
            do {result=try await op.run(action:"train",settings:settings,source:base.directory(root:root),files:base.files,projectID:owner.projectID,root:root,accountingRoot:accountingRoot,runtime:runtime,policy:policy,tableInputs:tables,historyPolicy:historyPolicy,budgetWindow:budgetWindow,resume:resume,isCurrent:{[weak self] in self?.owner.isCurrent(lease)==true && self?.reference(edge.from.address)==loadedBase},onProgress:{[weak self] in self?.status=$0},onTrainingProgress:{[weak self] point in
                guard let self else {return};try self.trainingProgress.append(point,worker:point.runID,maximumEpoch:Int(settings["epochs"]?.integerValue ?? 1),initialEpoch:initialEpoch);if point.kind=="validation",let train=point.trainingLoss,let valid=point.validationLoss {self.trainingMetrics.append(.init(epoch:point.epoch,trainLoss:train,validationLoss:valid,accuracy:nil))};if self.trainingPhase != "training" {self.trainingPhase="training"};self.trainingRunID=point.runID;self.status="Fine-tuning · epoch \(point.epoch) · \(point.updates) updates"
            })} catch {
                if let directory=op.recoveryDirectory {
                    let root=self.root,originalBase=base
                    let recovered=await Task.detached{try? ModelFineTuningRecovery.collect(directory:directory,address:address,document:document,base:originalBase,root:root)}.value
                    if let recovered {fineTuningRecoveries.insert(recovered,at:0)}
                }
                throw error
            }
            let progressFile:WorkflowArchiveFile?
            if historyPolicy.metricsAndDiagnostics {
                let request=try WorkflowToolRequest.decode(DiagnosticFileIO.read(root:result.directory,parts:["request.json"],limit:256_000))
                let archive=TrainingProgressArchive(schemaVersion:1,requestIdentity:request.requestIdentity,workerRunID:result.operationID,policy:reporting,observed:trainingProgress.observed,points:trainingProgress.points)
                let directory=result.directory
                progressFile=try await Task.detached(priority:.utility) {()->WorkflowArchiveFile in
                    let bytes=try JSONEncoder().encode(archive)
                    try DiagnosticFileIO.writeImmutable(bytes,root:directory,parts:["diagnostics","training-progress.json"],limit:8_000_000)
                    return .init(path:"diagnostics/training-progress.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))
                }.value
            } else {progressFile=nil;trainingProgress = .init()}
            let record=ModelFineTuningRecord(schemaVersion:1,operationID:result.operationID,address:address,document:document,sourceAddress:edge.from.address,base:base,finishedAt:Date(),files:result.files,progressFile:progressFile)
            let root=self.root
            let verified=Task.detached(priority:.userInitiated) {()->[String:JSONValue] in
                let metadata:[String:JSONValue]
                do {metadata=try record.verify(root:root)}catch {throw WorkflowArchiveError("Training receipt verification: "+error.localizedDescription)}
                do {try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:result.directory,parts:["model-training-outcome.json"],limit:2_000_000)}catch {throw WorkflowArchiveError("Training receipt publication: "+error.localizedDescription)}
                return metadata
            }
            var metadata=try await verified.value
            metadata["fineTuningMode"]=taskConfig.values["mode"];metadata["trainingOperation"] = .bool(true);metadata["baseIdentity"] = .string(base.identity)
            var inference=base.sourceConfiguration;inference["maximumTokens"]=taskConfig.values["maximumTokens"]
            let trained=ModelAssetReference(projectID:owner.projectID,directoryID:result.operationID,metadata:metadata,sourceConfiguration:inference)
            trainingDocuments[result.operationID]=document
            reservation.release(token)
            guard owner.isCurrent(lease),bindings.setReference(trained,for:address,lease:lease,expectedRevision:revision) else {throw WorkflowArchiveError(bindings.error ?? "Training inputs changed before publication.")}
            trainingReloadGeneration+=1;trainingRecords.insert(record,at:0);trainingPhase="completed";status=metadata["paused"] == .bool(true) ? "Fine-tuning paused at completed epoch · selected model ready":"Fine-tuning completed · selected model ready";onBindingChange?()
        }
    }
    func reloadTraining() {
        trainingReloadGeneration+=1
        let generation=trainingReloadGeneration,revision=bindings.revision,root=self.root,project=owner.projectID
        let activeIDs=Set(bindings.bindings.values.filter{$0.metadata["trainingOperation"] == .bool(true)}.map(\.directoryID))
        Task { [weak self] in
            let value=await Task.detached(priority:.utility) {()->([ModelFineTuningRecord],[String:WorkflowDocument],[String],[ModelFineTuningRecovery],[ModelImportedResume],[ModelCausalEvaluationRecord]) in
                let records=ModelFineTuningRecord.scan(root:root,project:project)
                var verified:[String:WorkflowDocument]=[:],issues:[String]=[]
                for record in records where activeIDs.contains(record.id) {
                    if Task.isCancelled {break}
                    do {_=try record.verify(root:root);verified[record.id]=record.document}catch {issues.append(error.localizedDescription)}
                }
                return (records,verified,issues,ModelFineTuningRecovery.scan(root:root,project:project),ModelImportedResume.scan(root:root,project:project),ModelCausalEvaluationRecord.scan(root:root,project:project))
            }.value
            guard let self,self.owner.projectID==project,self.trainingReloadGeneration==generation,self.bindings.revision==revision else {return}
            self.importedResumes=value.4;self.evaluationRecords=value.5;self.fineTuningRecoveries=value.3;self.trainingRecords=value.0;self.trainingDocuments=value.1;if !value.2.isEmpty {self.status="Saved fine-tuning model could not be restored: "+value.2.prefix(3).joined(separator:"; ")};self.onBindingChange?()
        }
    }

    @discardableResult func exportTraining(_ reference:ModelAssetReference,address:WorkflowNodeAddress,content:WorkflowSaveModelConfiguration.Content,access:WorkflowModelDestinationAccess,isCurrent:@escaping()->Bool,onTerminal:@escaping(Bool)->Void={_ in},onSaved:@escaping(URL)->Void)->Bool {
        guard !isBusy else {access.close();return false}
        let started=start(address,onTerminal:onTerminal) { [self] in
            defer {access.close()}
            let root=self.root,id=reference.directoryID
            guard let record=trainingRecords.first(where:{$0.id==id}) else {throw WorkflowArchiveError("Refresh History to load this fine-tuning result before exporting.")}
            let work=Task.detached(priority:.userInitiated) {()->WorkflowPendingExportPublication in
                let report=try record.report(root:root)
                let lora:Bool;if case .object(let settings)=report["settings"] {lora=settings["mode"] == .string("lora")}else {throw WorkflowFileLeaseError.unsafePath}
                guard content == .adapter || content == .continuation,content != .adapter || lora else {throw WorkflowArchiveError("Choose a supported source-specific export.")}
                if content == .continuation,report["continuation"] != .string("cpuEpochBoundary") {throw WorkflowArchiveError("This device result supports weights reuse, not exact CPU continuation.")}

                let group=content == .adapter ? "adapter":"latest",prefix="output/training/"+group+"/"
                let files=record.files.filter{$0.path.hasPrefix(prefix)}.map{WorkflowArchiveFile(path:String($0.path.dropFirst(prefix.count)),bytes:$0.bytes,sha256:$0.sha256)}
                guard !files.isEmpty else {throw WorkflowArchiveError("This run has no verified requested boundary.")}
                let source=record.directory(root).appendingPathComponent("output/training/"+group)
                let pending=try WorkflowPendingExportPublication(destination:access.destination,directory:true)
                do {
                    if content == .continuation,lora {
                        let basePrefix="output/training/base/"
                        let baseFiles=record.files.filter{$0.path.hasPrefix(basePrefix)}.map{WorkflowArchiveFile(path:String($0.path.dropFirst(basePrefix.count)),bytes:$0.bytes,sha256:$0.sha256)}
                        guard !baseFiles.isEmpty,(baseFiles+files).reduce(0,{$0+$1.bytes})<=2_010_000_000 else{throw WorkflowArchiveError("Portable LoRA resume needs its verified base within the 2 GB model allowance.")}
                        try pending.writeModelGroup("latest",files:files,from:source)
                        try pending.writeModelGroup("base",files:baseFiles,from:record.directory(root).appendingPathComponent("output/training/base"))
                        let bytes=try WorkflowArchiveCodec.encode(report);try pending.writeResumeMetadata(name:"training.json",bytes:bytes)
                        let entries=(files.map{WorkflowArchiveFile(path:"latest/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}+baseFiles.map{WorkflowArchiveFile(path:"base/"+$0.path,bytes:$0.bytes,sha256:$0.sha256)}+[.init(path:"training.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))]).sorted{$0.path<$1.path}
                        try pending.writeResumeMetadata(name:"result-manifest.json",bytes:WorkflowArchiveCodec.encode(["schemaVersion":JSONValue.integer(1),"files":try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(entries))]))
                    }else{for file in files {try Task.checkCancellation();try pending.writeModelFile(file,from:source)}}
                    return pending
                }catch {pending.close();throw error}
            }
            status="Verifying fine-tuned export…"
            let pending=try await withTaskCancellationHandler(operation:{try await work.value},onCancel:{work.cancel()});defer {pending.close()}
            try Task.checkCancellation();try access.check();guard isCurrent() else {throw CancellationError()};try pending.commit();status="Model contents saved.";onSaved(pending.destination)
        }
        if !started {access.close()};return started
    }

    func predictionArtifact(_ address:WorkflowNodeAddress)->TableViewQuerySession.Artifact? {
        if let record=currentEvaluation,record.address==address,evaluationIsCurrent(record),let file=record.files.first(where:{$0.path=="output/table.json"}) {return .init(indexURL:record.directory(root).appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table")}
        guard let reference=reference(address),let record=trainingRecords.first(where:{$0.id==reference.directoryID && $0.address==address}),let file=record.files.first(where:{$0.path=="output/table.json"}) else {return nil}
        return .init(indexURL:record.directory(root).appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table")
    }

    func outputArtifact(_ endpoint:WorkflowEndpoint)->TableViewQuerySession.Artifact? {
        if endpoint.port=="predictions" {return predictionArtifact(endpoint.address)}
        guard generationIsCurrent(endpoint),let record=generationRecord,let file=record.files.first(where:{$0.path=="output/table.json"}) else {return nil}
        return .init(indexURL:record.directory(root).appendingPathComponent(file.path),indexSHA256:file.sha256,sourceSHA256:file.sha256,inputKind:"table")
    }

}


extension ModelLoaderStore {
    /// Import inert, bounded state only. Never copy or run code from a selected package.
    func chooseDocumentPackage(_ address:WorkflowNodeAddress,policy:ResolvedExecutionPreferences) {
        let panel=NSOpenPanel();panel.canChooseFiles=false;panel.canChooseDirectories=true;panel.allowsMultipleSelection=false
        panel.prompt="Choose model";panel.message="Choose a Kwip Prebuilt model or document classifier package. Only inert model state and optional learned preparation are imported; package code is never executed. The worker validates the complete processor and classifier before prediction."
        guard panel.runModal() == .OK,let source=panel.url else {return}
        let scoped=source.startAccessingSecurityScopedResource()
        start(address) { [self] in
            defer {if scoped {source.stopAccessingSecurityScopedResource()}}
            try await importDocumentClassifierState(source,address:address,policy:policy)
        }
        if !isBusy,scoped {source.stopAccessingSecurityScopedResource()}
    }
    func importDocumentClassifierState(_ source:URL,address:WorkflowNodeAddress,policy:ResolvedExecutionPreferences)async throws {
            let lease=owner.lease(),revision=bindings.revision,config=try configuration(address),project=owner.projectID,id=UUID().uuidString,root=self.root,accounting=self.accountingRoot
            guard config.source == .documentPackage else {throw WorkflowArchiveError("Choose Prebuilt / document model package first.")}
            let directory=root.appendingPathComponent("model-assets/\(project)/\(id)")
            let worker=Task.detached(priority:.userInitiated) {()->ModelAssetReference in
                let bytes=try DiagnosticFileIO.read(root:source,parts:["state.json"],limit:32_000_000)
                let state=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
                let prebuilt=state["stateFormat"] == .string("kwipml.prebuilt-model.v1")
                let classes:[JSONValue],vocabulary:[JSONValue]
                if prebuilt {
                    guard case .object(let c)=state["configuration"],let a=c["algorithm"]?.stringValue,let entry=PrebuiltModelConfiguration.algorithms[a],entry.predict,case .object(let options)=c["options"],case .array(let names)=state["featureColumns"],(1...128).contains(names.count),state["featureCount"]?.integerValue==names.count else {throw WorkflowArchiveError("Invalid inert Prebuilt model package.")}
                    try PrebuiltModelConfiguration.validateSettings(algorithm:a,settings:options)
                    classes=[];vocabulary=[]
                } else {
                    guard state["stateFormat"] == .string("kwipml.document-classifier.v1"),case .array(let c)=state["classes"],(2...128).contains(c.count),case .object(let processor)=state["processor"],case .array(let v)=processor["vocabulary"],(1...65536).contains(v.count) else {throw WorkflowArchiveError("Choose a supported document or Prebuilt model inference package.")};classes=c;vocabulary=v
                }
                let preparationBytes=try DiagnosticFileIO.readIfPresent(root:source,parts:["preparation.json"],limit:PrebuiltPreparationAttachment.maximumBytes)
                let attachment=try preparationBytes.map {try PrebuiltPreparationAttachment.decode($0,modelBytes:bytes)}
                guard attachment == nil || prebuilt else {throw WorkflowArchiveError("Learned preparation requires a Prebuilt model.")}
                var files=[WorkflowArchiveFile(path:"state.json",bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes))]
                if let preparationBytes {files.append(.init(path:"preparation.json",bytes:preparationBytes.count,sha256:WorkflowArchiveCodec.hash(preparationBytes)))}
                let reservation=try policy.storageLimitBytes.map {try WorkflowStorageAdmission(workspace:root,accountingRoot:accounting).reserve(scopes:[.init(namespace:.modelAssets,components:[project,id])],peakBytes:Int64(files.reduce(0){$0+$1.bytes}+1_000_000),limitBytes:$0)}
                defer {reservation?.close()}
                do {
                    try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
                    try bytes.write(to:directory.appendingPathComponent("state.json"),options:.withoutOverwriting)
                    if let preparationBytes {try preparationBytes.write(to:directory.appendingPathComponent("preparation.json"),options:.withoutOverwriting)}
                    var metadata:[String:JSONValue]=["identity":.string(try ModelAssetFiles.sourceIdentity(files)),"files":try JSONDecoder().decode(JSONValue.self,from:WorkflowArchiveCodec.encode(files)),"architecture":.string(prebuilt ? "Prebuilt model":"TF-IDF · Logistic regression"),"task":.string(prebuilt ? "prebuilt":"documentClassification"),"classes":.array(classes),"vocabularySize":.integer(Int64(vocabulary.count))]
                    if prebuilt {metadata["prebuiltConfiguration"]=state["configuration"];metadata["featureColumns"]=state["featureColumns"]}
                    if let attachment {metadata["learnedPreparation"] = .object(attachment.summary)}
                    return ModelAssetReference(projectID:project,directoryID:id,metadata:metadata,sourceConfiguration:config.values)
                } catch {try? FileManager.default.removeItem(at:directory);throw error}
            }
            let reference=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
            var retained=false;defer {if !retained {try? FileManager.default.removeItem(at:directory)}}
            try Task.checkCancellation()
            guard owner.isCurrent(lease),try configuration(address)==config,reference.hasValidPins,bindings.setReference(reference,for:address,lease:lease,expectedRevision:revision) else {throw WorkflowArchiveError("Model selection changed during import.")}
            retained=true;status="Model state imported · full state validation runs before inference.";onBindingChange?()
    }

}

extension ModelLoaderStore {
    func inspectFineTuning(address:WorkflowNodeAddress,draft:[String:JSONValue],training:TableViewQuerySession.Artifact,validation:TableViewQuerySession.Artifact,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,reservation:WorkflowExecutionReservation,onTerminal:@escaping(Bool)->Void) {
        start(address,onTerminal:onTerminal) { [self] in
            guard let token=reservation.acquire() else{throw WorkflowArchiveError("Wait for the current workspace operation.")};defer{reservation.release(token)}
            let lease=owner.lease()
            guard let node=owner.document.nodes.first(where:{$0.address==address}),case .train(let config)=node.payload,let edge=owner.document.edges.first(where:{$0.to.address==address && $0.to.port=="model"}),let base=reference(edge.from.address) else{throw WorkflowArchiveError("Load the connected model before inspecting preparation.")}
            let settings=try fineTuningSettings(address:address,config:draft,policy:policy)
            let tables:[ModelTableCapture.Input]=[.init(role:.training,url:training.indexURL,sha256:training.sourceSHA256,retainedSegments:training.retainedSegments),.init(role:.validation,url:validation.indexURL,sha256:validation.sourceSHA256,retainedSegments:validation.retainedSegments)]
            inspectionAddress=address;defer{inspectionAddress=nil}
            fineTuningPreview=nil;status="Inspecting model-matched loss preparation…"
            let op=ModelAssetOperation();operation=op
            let result=try await op.run(action:"prepare",settings:settings,source:base.directory(root:root),files:base.files,projectID:owner.projectID,root:root,accountingRoot:accountingRoot,runtime:runtime,policy:policy,tableInputs:tables,isCurrent:{[weak self] in self?.owner.isCurrent(lease)==true},onProgress:{[weak self] in self?.status=$0})
            let preview=try await Task.detached {try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:result.directory,parts:["output","tokens.json"],limit:8_000_000))}.value
            guard owner.isCurrent(lease) else{throw CancellationError()}
            fineTuningPreview=preview;fineTuningInspectedConfiguration=draft;status="Prepared inputs inspected · no training run."
        }
    }
}

extension ModelLoaderStore {
    func evaluationIsCurrent(_ record:ModelCausalEvaluationRecord)->Bool {WorkflowOutputConstruction.matches(owner.document,captured:record.document,provider:record.address) && reference(record.sourceAddress)?.identity==record.model.identity}
    func evaluateCausal(address:WorkflowNodeAddress,training:TableViewQuerySession.Artifact,evaluation:TableViewQuerySession.Artifact,population:TableViewQuerySession.Artifact,runtime:RuntimeLocation,policy:ResolvedExecutionPreferences,exposureRoot:URL?,reservation:WorkflowExecutionReservation,onTerminal:@escaping(Bool)->Void) {
        start(address,onTerminal:onTerminal){[self] in
            guard let token=reservation.acquire()else{throw WorkflowArchiveError("Wait for the current workspace operation.")};defer{reservation.release(token)}
            let lease=owner.lease(),document=owner.document,plan=try WorkflowEvaluationCompiler.compile(document,evaluate:address,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
            guard let source=document.nodes.first(where:{$0.address==plan.modelSource.address}),case .train(let config)=source.payload,let model=reference(source.address),[JSONValue.string("causalLanguageModel"),.string("typedDecisions")].contains(model.metadata["task"] ?? .null) else{throw WorkflowArchiveError("Connect a verified imported Train model. Its task/target mapping is used for evaluation.")}
            var settings=try fineTuningSettings(address:source.address,config:config,policy:policy);settings["evaluationRole"] = .string(plan.settings.partitionRole);settings["batchSize"] = .integer(Int64(plan.settings.batchSize));settings["device"] = .string(plan.settings.device)
            let op=ModelAssetOperation();operation=op;status="Scoring the captured "+plan.settings.partitionRole+" population…"
            let result=try await op.run(action:"evaluate",settings:settings,source:model.directory(root:root),files:model.files,projectID:document.projectID,root:root,accountingRoot:accountingRoot,runtime:runtime,policy:policy,tableInputs:[.init(role:.training,url:training.indexURL,sha256:training.sourceSHA256,retainedSegments:training.retainedSegments),.init(role:.evaluation,url:evaluation.indexURL,sha256:evaluation.sourceSHA256,retainedSegments:evaluation.retainedSegments),.init(role:.evaluationPopulation,url:population.indexURL,sha256:population.sourceSHA256,retainedSegments:population.retainedSegments)],isCurrent:{[weak self] in self?.owner.isCurrent(lease)==true && self?.reference(source.address)==model},onProgress:{[weak self] in self?.status=$0})
            let record=ModelCausalEvaluationRecord(operationID:result.operationID,address:address,document:document,sourceAddress:source.address,model:model,files:result.files,finishedAt:Date()),root=self.root
            let summary=try await Task.detached(priority:.userInitiated){()->[String:JSONValue] in
                let summary=try record.disclose(root:root,exposureRoot:exposureRoot)
                try DiagnosticFileIO.writeImmutable(WorkflowArchiveCodec.encode(record),root:result.directory,parts:["model-evaluation-outcome.json"],limit:2_000_000)
                return summary
            }.value
            guard owner.isCurrent(lease)else{throw CancellationError()}
            let tableDirectory=result.directory
            let table=try await Task.detached{try JSONDecoder().decode([String:JSONValue].self,from:DiagnosticFileIO.read(root:tableDirectory,parts:["output","table.json"],limit:8_000_000))}.value
            var metadata=table.filter{$0.key != "rows"};if case .array(let rows)=table["rows"]{metadata["rowCount"] = .integer(Int64(rows.count))}
            evaluationRecords.insert(record,at:0);currentEvaluation=record;evaluationSummary=summary;evaluationTableMetadata=metadata;status=model.metadata["task"] == .string("typedDecisions") ? "Typed-decision evaluation completed · loss, Brier score and accuracy":"Causal evaluation completed · token-weighted loss and perplexity";onBindingChange?()
        }
    }
}

extension ModelLoaderStore {
    func chooseResumePackage(address:WorkflowNodeAddress,policy:ResolvedExecutionPreferences) {
        guard !isBusy,let edge=owner.document.edges.first(where:{$0.to.address==address && $0.to.port=="model"}),let base=reference(edge.from.address)else{show("Load the original matching base first.",for:address);return}
        let panel=NSOpenPanel();panel.canChooseFiles=false;panel.canChooseDirectories=true;panel.allowsMultipleSelection=false;panel.prompt="Inspect CPU resume package";panel.message="Choose a saved CPU training boundary. Only bounded safetensors and reviewed metadata are imported. The original data and matching base are still required."
        guard panel.runModal() == .OK,let source=panel.url else{return};let scoped=source.startAccessingSecurityScopedResource()
        let started=start(address){[self] in
            defer{if scoped{source.stopAccessingSecurityScopedResource()}}
            let root=self.root,accounting=accountingRoot,lease=owner.lease()
            let worker=Task.detached(priority:.userInitiated){try ModelImportedResume.importPackage(source,base:base,root:root,accountingRoot:accounting,limit:policy.storageLimitBytes)}
            let record=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
            guard owner.isCurrent(lease)else{throw CancellationError()};importedResumes.insert(record,at:0);status="CPU resume package imported · select its epoch boundary in Configure fine-tuning."
        }
        if !started,scoped{source.stopAccessingSecurityScopedResource()}
    }
}
