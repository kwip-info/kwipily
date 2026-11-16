import AppKit
import Combine
import CryptoKit
import Darwin
import KwipMLCore

/// Fixed stages are safe to expose; endpoint URLs, credentials and worker prose are not.
enum WorkflowAPICaptureStage:String,CaseIterable {
    case configuration,connection,snapshot,broker,verification,publication
    var code:String {"api_source_"+rawValue+"_failed"}
    var message:String {"API source capture failed during "+rawValue+"; no complete snapshot was published."}
}

typealias DatabaseSourceCaptureExecutor = @MainActor ([String: JSONValue], DatabaseSourceCredentials, FileHandle) async throws -> Data

/// Local source access and recorded header guidance for one workflow owner. Rendering
/// reads cached declarations only; grants, snapshots and workers require explicit actions.
@MainActor final class WorkflowSourceProvider: ObservableObject, WorkflowDataSources {
    private weak var workspace: WorkspaceStore?
    let owner: WorkflowAuthoringStore
    let bindings: WorkflowPrivateSourceBindings
    var databaseOutputs:DatabaseOutputStore? {outputDestinations?.databaseOutputs}
    var projectID: String { owner.projectID }
    var sessionID: UUID { owner.sessionID }
    @Published private(set) var isInspecting = false
    @Published private(set) var statusMessage = "Choose a local table source, then inspect its recorded columns."
    var changes: AnyPublisher<Void, Never> { objectWillChange.eraseToAnyPublisher() }
    var isBusy: Bool { !isCurrent || isInspecting || (workspace?.isBusy ?? true) || owner.isBusy }

    var pickSource: () -> URL? = {
        let panel = NSOpenPanel()
        panel.title = "Choose Table Input file"
        panel.canChooseFiles = true; panel.canChooseDirectories = false; panel.allowsMultipleSelection = false
        return panel.runModal() == .OK ? panel.url : nil
    }
    var pickFileInputSource:(Bool)->URL? = { directory in
        let panel=NSOpenPanel();panel.title=directory ? "Choose File Input folder":"Choose File Input file"
        panel.canChooseFiles = !directory;panel.canChooseDirectories=directory;panel.allowsMultipleSelection=false
        return panel.runModal() == .OK ? panel.url:nil
    }
    var rememberSource: (URL) throws -> Void
    var snapshotSource: (String) throws -> SourceSnapshot
    var inspectionExecute: SourceInspectionExecutor?
    var credentialStorage: DatabaseSourceCredentialStorage = DatabaseSourceKeychain()
    var databaseCaptureExecutor: DatabaseSourceCaptureExecutor = { options, credentials, output in
        try await DatabaseSourceBrokerClient().capture(options: options, credentials: credentials, output: output)
    }
    var outputDestinations: DataOutputDestinationProvider? {
        didSet {
            outputDestinationObservation = outputDestinations?.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
            objectWillChange.send()
        }
    }
    private var outputDestinationObservation: AnyCancellable?
    private var active = true
    private var observations: [AnyCancellable] = []
    private struct Header {
        let path: String
        let config: [String: JSONValue]
        let bindingRevision: UInt64
        let snapshotSHA256: String
        let columns: [String]
        let result: SourceInspectionResult
        let completedAt: Date
    }
    private var headers: [WorkflowNodeAddress: Header] = [:]
    private var apiFailure:(detail:DataFlowFailureDetail,run:UUID,revision:UInt64,binding:UInt64)?
    var apiFailureEvidence:WorkflowPreparationFailureEvidence? {guard let f=apiFailure,isCurrent else{return nil};return .init(detail:f.detail,runID:f.run,capturedRevision:f.revision,isCurrent:owner.document.revision==f.revision && bindings.revision==f.binding)}
    private func recordAPIFailure(_ address:WorkflowNodeAddress,run:UUID=UUID(),stage:WorkflowAPICaptureStage?=nil) {guard let n=owner.document.nodes.first(where:{$0.address==address}) else{return};apiFailure=(.init(schemaVersion:1,code:stage?.code ?? "api_source_failed",nodeID:address.localID,nodeLabel:n.name.isEmpty ? "Table Input":n.name,operatorID:"kwip.data.sourceCSV",componentTitle:"Table Input",inputRowCount:nil,cause:stage?.message ?? "API source capture failed; no complete snapshot was published.",suggestion:"Check the request URL, private connection, JSON paths, pagination and resource limits. Fetch snapshot again after correcting the saved settings."),run,owner.document.revision,bindings.revision)}
    private var apiFetches:[WorkflowNodeAddress:(config:[String:JSONValue],receipt:[String:JSONValue])]=[:]
    func apiFetchReceipt(for address:WorkflowNodeAddress)->[String:JSONValue]? {if let saved=apiFetches[address],sourceNode(address).map(config)==saved.config {return saved.receipt};return sourceInspectionPreview(for:address)?.apiFetch}
    private struct InspectionStatusRecord {
        var lease: WorkflowEditorLease
        let bindingRevision: UInt64
        let path: String?
        let config: [String: JSONValue]
        let status: SourceInspectionStatus
    }
    private var inspectionStatuses: [WorkflowNodeAddress: InspectionStatusRecord] = [:]

    var capturedPredictionSchema:((WorkflowEndpoint)->[DataColumnSchema]?)?
    var capturedSchema:((DataFlowEndpoint)->[DataColumnSchema]?)?
    /// Compare refreshed cached hashes only. External files are not read while editing.
    func observedSourcesMatch(_ outputs:[String:[String:JSONValue]])->Bool {
        for (address,header) in headers {
            // A prefix sample cannot establish or refute the full execution digest.
            if header.result.parsing?.sampleOnly == true { continue }
            guard let port=outputs[address.localID]?["table"],case .object(let value)=port,
                  case .object(let lineage)=value["lineage"],case .string(let hash)=lineage["sourceDigest"] else {continue}
            if hash != header.snapshotSHA256 {return false}
        }
        return true
    }
    private final class Inspection {
        let address: WorkflowNodeAddress
        let lease: WorkflowEditorLease
        let bindingRevision: UInt64
        let path: String
        let config: [String: JSONValue]
        let window: ExecutionBudgetWindow
        var snapshotSHA256: String?
        var storageLimitBytes:Int64?
        var workspace: WorkspaceStore?
        var reservationID: UUID?
        var worker: WorkflowWorkerHandle?
        var launchTask: Task<Void, Never>?
        var verificationTask: Task<Void, Never>?
        var cancelled = false
        var terminalReceived = false
        init(address: WorkflowNodeAddress, lease: WorkflowEditorLease, bindingRevision: UInt64,
             path: String, config: [String: JSONValue], window: ExecutionBudgetWindow) {
            self.address = address; self.lease = lease; self.bindingRevision = bindingRevision
            self.path = path; self.config = config; self.window = window
        }
    }
    private var scheduledSnapshots:[DatabaseSourceSnapshot]=[]
    private var inspection: Inspection?
    private var databaseCaptureTask: Task<Void, Never>?

    init(workspace: WorkspaceStore, owner: WorkflowAuthoringStore, bindings: WorkflowPrivateSourceBindings) throws {
        guard bindings.owner === owner else { throw DataFlowContractError("Private sources must belong to this exact workflow owner.") }
        self.workspace = workspace; self.owner = owner; self.bindings = bindings
        rememberSource = { [weak workspace] url in
            guard let workspace else { throw DataFlowContractError("The source workspace is no longer available.") }
            try workspace.sourceAccess.remember(url)
        }
        snapshotSource = { [weak workspace] path in
            guard let workspace else { throw DataFlowContractError("The source workspace is no longer available.") }
            return try workspace.sourceAccess.withAccess(to: path) { url in
                try SourceSnapshot.prepare(source: url, directory: workspace.workspaceDirectory.appendingPathComponent("source-snapshots"))
            }
        }
        observations = [owner.objectWillChange, bindings.objectWillChange, workspace.objectWillChange].map { publisher in
            publisher.sink { [weak self] _ in self?.objectWillChange.send() }
        }
        observations.append(owner.$document.dropFirst().sink { [weak self] _ in
            guard let self, owner.isPublishingLayoutMove else { return }
            // Position changes cannot alter already verified source bytes or parsing settings.
            for address in self.inspectionStatuses.keys {
                self.inspectionStatuses[address]?.lease = owner.lease()
            }
        })
    }
    private var isCurrent: Bool {
        active && workspace?.activeProjectID == projectID && owner.isCurrent(owner.lease())
    }
    private func sourceNode(_ address: WorkflowNodeAddress) -> WorkflowNode? {
        guard address.domain == .data else { return nil }
        return owner.document.nodes.first {
            return $0.address == address && DataFlowSourceOperators.isSource($0)
        }
    }
    private func config(_ node: WorkflowNode) -> [String: JSONValue] {
        if case .dataOperator(_, let config) = node.payload { return config }; return [:]
    }
    private func canChoose(_ address: WorkflowNodeAddress, lease: WorkflowEditorLease, revision: UInt64) -> Bool {
        !isBusy && owner.isCurrent(lease) && bindings.revision == revision && sourceNode(address) != nil
            && owner.selection.nodeAddresses == [address]
    }

    @discardableResult func chooseSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool {
        let revision = bindings.revision
        guard canChoose(address, lease: lease, revision: revision) else {
            statusMessage = "Select the current Table Input and wait for active work before choosing a file."; return false
        }
        guard let selected=sourceNode(address) else {return false}
        let fileInput:Bool;if case .dataOperator(DataFlowSourceOperators.fileInput,_)=selected.payload {fileInput=true}else{fileInput=false}
        let chosen=fileInput ? pickFileInputSource(config(selected)["sourceMode"]?.stringValue == "directory"):pickSource()
        guard let url = chosen else { return false }
        guard url.isFileURL, canChoose(address, lease: lease, revision: revision) else {
            statusMessage = "The workflow, selection, or binding changed while the picker was open. Choose the source again."; return false
        }
        do {
            try rememberSource(url)
            // The grant operation may be supplied by a host; it cannot revive stale authority.
            guard canChoose(address, lease: lease, revision: revision) else {
                statusMessage = "The grant was saved, but the workflow changed before its source reference could be committed."; return false
            }
            guard bindings.setPath(url.path, for: address, lease: lease, expectedRevision: revision) else {
                statusMessage = "The grant was saved, but the source reference was not committed: " + (bindings.error ?? "Private source save failed.")
                return false
            }
            headers[address] = nil
            inspectionStatuses[address] = nil
            statusMessage = "Source linked on this Mac. Inspecting its columns…"
            _ = inspectSource(for: address, lease: owner.lease())
            return true
        } catch { statusMessage = error.localizedDescription; return false }
    }

    /// Explicit refresh captures one selected remote table. Workers only receive the verified local IPC snapshot.
    @discardableResult func connectDatabaseSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool {
        guard !isBusy, owner.isCurrent(lease), owner.selection.nodeAddresses == [address], let node = sourceNode(address), let workspace else { return false }
        let configuration = config(node), revision = bindings.revision
        @MainActor func databaseStatus(_ phase: SourceInspectionStatus.Phase, _ detail: String) {
            setInspectionStatus(for: address, lease: lease, bindingRevision: revision, path: bindings.path(for: address),
                config: configuration, phase: phase, detail: detail)
        }
        do {
            let isAPI = try TableInputConfiguration.format(of:configuration) == .api
            guard try isAPI || TableInputConfiguration.format(of: configuration) == .database else { return false }
            if isAPI {apiFailure=nil}
            let options = try isAPI ? APITableSettings.normalized(TableInputConfiguration.options(in:configuration)):DatabaseSourceSettings.normalized(TableInputConfiguration.options(in: configuration))
            if options["backend"] == .string("sqlite") { return chooseSource(for: address, lease: lease) }
            let connectionID = try isAPI ? APITableSettings.origin(options):DatabaseSourceSettings.connectionIdentity(options)
            let selectedProfile=databaseOutputs?.profile(for:address)
            let selectedCredential:DatabaseSourceCredentials? = try isAPI ? databaseOutputs?.secret(for:address,config:options,write:false):selectedProfile == nil ? credentialStorage.read(projectID:projectID,nodeID:address.localID,connectionID:connectionID):databaseOutputs?.secret(for:address,config:options,write:false)
            guard let credentials=selectedCredential else {
                statusMessage = "Save credentials for this database endpoint before connecting."
                databaseStatus(.failed, statusMessage); return false
            }
            let policy = try workspace.executionPolicy(for: projectID)
            let storage = workspace.workspaceDirectory, accountingRoot = workspace.applicationDirectory
            let captureExecutor = databaseCaptureExecutor
            headers[address] = nil
            isInspecting = true; owner.setExecutionBusy(true); workspace.isPreparingData = true
            workspace.cancelDataPreparation = { [weak self] in self?.databaseCaptureTask?.cancel() }
            let reservationID = workspace.dataPreparationReservationID
            statusMessage = isAPI ? "Fetching API table snapshot…":"Capturing selected database table…"
            databaseStatus(.inspecting, statusMessage)
            databaseCaptureTask = Task { [self] in
                var accepted = false
                var committedRevision: UInt64?
                var snapshot: DatabaseSourceSnapshot?
                defer {
                    snapshot?.finish(retain: accepted)
                    databaseCaptureTask = nil; isInspecting = false
                    if workspace.dataPreparationReservationID == reservationID {
                        workspace.isPreparingData = false; workspace.cancelDataPreparation = nil; owner.setExecutionBusy(false)
                        if accepted, !Task.isCancelled, isCurrent, bindings.revision == committedRevision,
                           owner.selection.nodeAddresses == [address], sourceNode(address).map(config) == configuration,
                           bindings.path(for: address) == snapshot?.destination.path {
                            _ = inspectSource(for: address, lease: owner.lease())
                        }
                    }
                }
                @MainActor func checkAuthority() throws {
                    try Task.checkCancellation()
                    guard isCurrent, owner.isCurrent(lease), bindings.revision == revision,
                          owner.selection.nodeAddresses == [address], sourceNode(address).map(config) == configuration else { throw WorkflowFileLeaseError.unsafePath }
                }
                do {
                    try checkAuthority()
                    let create = Task.detached(priority: .userInitiated) {
                        try DatabaseSourceSnapshot(workspace: storage, accountingRoot: accountingRoot, storageLimitBytes: policy.storageLimitBytes)
                    }
                    let captured = try await withTaskCancellationHandler(operation: { try await create.value }, onCancel: { create.cancel() })
                    snapshot = captured
                    try checkAuthority()
                    let receipt = try await isAPI ? DatabaseSourceBrokerClient().capture(options:options,credentials:credentials,output:captured.output,memoryLimit:policy.budget.resources?.maximumResidentBytes,operationSeconds:policy.budget.maximumRunSeconds):captureExecutor(options, credentials, captured.output)
                    try checkAuthority()
                    let verification = Task.detached(priority: .userInitiated) { try captured.verify(receipt: receipt) }
                    try await withTaskCancellationHandler(operation: { try await verification.value }, onCancel: { verification.cancel() })
                    try checkAuthority()
                    try rememberSource(captured.destination)
                    // A host grant callback can mutate selection, the graph or the
                    // binding. It cannot revive the original authority afterward.
                    try checkAuthority()
                    guard bindings.commitCapturedPath(captured.destination.path, for: address, lease: lease, expectedRevision: revision,
                                                     workspace: workspace, reservationID: reservationID) else { throw WorkflowArchiveError("Database source binding changed during capture.") }
                    if isAPI,let r=try? JSONDecoder().decode([String:JSONValue].self,from:receipt),case .object(let fetch)=r["apiFetch"] {apiFetches[address]=(configuration,fetch)}
                    accepted = true; committedRevision = bindings.revision
                    headers[address] = nil; inspectionStatuses[address] = nil
                    statusMessage = isAPI ? "API snapshot ready. Inspecting its recorded rows…":"Database snapshot ready. Inspecting its recorded rows…"
                } catch {
                    if let storageError = error as? WorkflowStorageAdmission.Failure { statusMessage = storageError.localizedDescription }
                    else { statusMessage = error is CancellationError ? "Database capture cancelled." : "Database capture failed. Check the endpoint, table, credentials and TLS settings." }
                    if isAPI,!Task.isCancelled {recordAPIFailure(address)}
                    databaseStatus(error is CancellationError ? .cancelled : .failed, statusMessage)
                }
            }
            return true
        } catch {
            statusMessage = "Database source is unavailable. Check its settings, credentials and managed storage allowance."
            databaseStatus(.failed, statusMessage); return false
        }
    }

    @discardableResult func inspectSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool {
        guard !isBusy, owner.isCurrent(lease), let node = sourceNode(address), let path = bindings.path(for: address) else {
            statusMessage = "Choose a current Table Input and wait for active work before inspecting it."; return false
        }
        let configuration = config(node), revision = bindings.revision
        headers[address] = nil
        func refuse(_ message: String) -> Bool {
            statusMessage = message
            setInspectionStatus(for: address, lease: lease, bindingRevision: revision,
                path: path, config: configuration, phase: .failed, detail: message)
            return false
        }
        guard let workspace else { return refuse("The source workspace is no longer available.") }
        guard let runtime = workspace.runtime else { return refuse(workspace.runtimeError ?? "Runtime unavailable.") }
        let parsingConfiguration: [String: JSONValue]
        let fileInput:Bool;if case .dataOperator(DataFlowSourceOperators.fileInput,_)=node.payload {fileInput=true}else{fileInput=false}
        do { parsingConfiguration = try fileInput ? FileInputConfiguration.validate(configuration):TableInputConfiguration.validate(configuration) }
        catch { return refuse(error.localizedDescription) }
        let separator = parsingConfiguration["separator"]?.stringValue ?? ","
        let completeFile = (try? TableInputConfiguration.format(of: parsingConfiguration)) != .csv
        do {
            let policy = try workspace.executionPolicy(for: projectID)
            let window = try ExecutionBudgetWindow(policy: policy.budget, startedAtUptime: ProcessInfo.processInfo.systemUptime)
            let operation = Inspection(address: address, lease: lease, bindingRevision: revision,
                                       path: path, config: configuration, window: window)
            let accountingRoot = workspace.applicationDirectory
            let storage = workspace.workspaceDirectory, permissions = workspace.sourceAccess.directory
            operation.workspace = workspace;operation.storageLimitBytes=policy.storageLimitBytes
            inspection = operation; isInspecting = true
            owner.setExecutionBusy(true); workspace.isPreparingData = true
            // The reservation owns this provider until joined capture, actual exit and final verification.
            workspace.cancelDataPreparation = { [self] in cancel() }
            operation.reservationID = workspace.dataPreparationReservationID
            statusMessage = "Capturing source columns for inspection…"
            setInspectionStatus(operation, phase: .inspecting, detail: statusMessage)
            let configuredExecute = inspectionExecute
            let operationProjectID = projectID
            operation.launchTask = Task { [self] in
                do {
                    try Task.checkCancellation()
                    if fileInput {
                        let capture=Task.detached(priority:.userInitiated) {
                            try FileInputCollectionCapture.capture(path:path,configuration:parsingConfiguration,workspace:storage,accountingRoot:accountingRoot,sourcePermissions:permissions,storageLimitBytes:policy.storageLimitBytes,previewLimit:8)
                        }
                        let snapshot=try await withTaskCancellationHandler(operation:{try await capture.value},onCancel:{capture.cancel()});defer{snapshot.remove()}
                        guard contextCurrent(operation) else {throw CancellationError()}
                        operation.snapshotSHA256=snapshot.source.sha256
                        let worker=try await SourceInspectionExecution.launch(workspace:storage,projectID:projectID,runtime:runtime,window:window,sourceURL:snapshot.source.url,sourceSHA256:snapshot.source.sha256,separator:",",configuration:parsingConfiguration,collectionMembers:snapshot.source.collectionMembers,isFileInput:true,onExit:{[self] code,stderr,result in receiveExit(operation,code:code,stderr:stderr,result:result)})
                        if inspection === operation && !operation.terminalReceived {operation.worker=worker;if operation.cancelled{worker.cancel()}}
                        operation.launchTask=nil;return
                    }
                    let capture = Task.detached(priority: .userInitiated) {
                        try SourceInspectionCapture.capture(path: path, workspace: storage, accountingRoot: accountingRoot,
                            sourcePermissions: permissions, window: window,
                            projectID: operationProjectID, storageLimitBytes: policy.storageLimitBytes, completeFile: completeFile)
                    }
                    let snapshot = try await withTaskCancellationHandler(operation: { try await capture.value }, onCancel: { capture.cancel() })
                    defer { snapshot.remove() }
                    try Task.checkCancellation()
                    guard contextCurrent(operation) else { throw CancellationError() }
                    operation.snapshotSHA256 = snapshot.sha256
                    statusMessage = "Inspecting source columns from a captured snapshot…"
                    setInspectionStatus(operation, phase: .inspecting, detail: statusMessage)
                    let execute: SourceInspectionExecutor = configuredExecute ?? { workspace, project, runtime, window, source, hash, separator, exit in
                        try await SourceInspectionExecution.launch(workspace: workspace, projectID: project, runtime: runtime,
                            window: window, sourceURL: source, sourceSHA256: hash, separator: separator,
                            configuration: parsingConfiguration, storageAdmission: snapshot.storageReservation, onExit: exit)
                    }
                    let worker = try await execute(storage, projectID, runtime, window,
                        snapshot.sourceURL, snapshot.sha256, separator, { [self] code, stderr, result in
                            receiveExit(operation, code: code, stderr: stderr, result: result)
                        })
                    if inspection === operation && !operation.terminalReceived {
                        operation.worker = worker
                        if operation.cancelled { worker.cancel() }
                    }
                    operation.launchTask = nil
                } catch {
                    operation.launchTask = nil
                    if !operation.terminalReceived {
                        operation.terminalReceived = true
                        finish(operation, message: error.localizedDescription)
                    }
                }
            }
            return true
        } catch { return refuse(error.localizedDescription) }
    }
    private func contextCurrent(_ operation: Inspection) -> Bool {
        inspection === operation && !operation.cancelled && isCurrent && owner.isCurrent(operation.lease)
            && operation.reservationID == operation.workspace?.dataPreparationReservationID
            && bindings.revision == operation.bindingRevision
            && bindings.path(for: operation.address) == operation.path
            && sourceNode(operation.address).map(config) == operation.config
    }
    private func receiveExit(_ operation: Inspection, code: Int32, stderr: String, result: SourceInspectionResult?) {
        guard inspection === operation, !operation.terminalReceived else { return }
        operation.terminalReceived = true; operation.worker = nil
        guard contextCurrent(operation) else { finish(operation, message: "Source inspection belongs to an earlier context; refresh its columns.", phase: .stale); return }
        guard code == 0, let result, result.sourceSHA256 == operation.snapshotSHA256 else {
            let message = code == 124
                ? "Source inspection exceeded its time allowance. Reduce the input size or adjust the execution budget."
                : (CSVSourceDiagnostic.decodeTransport(stderr) != nil ? stderr
                    : (stderr.isEmpty ? "Source inspection did not finish with verified columns." : "Source inspection failed: " + String(stderr.prefix(500))))
            finish(operation, message: message); return
        }
        guard let workspace = operation.workspace else { finish(operation, message: "Source workspace is unavailable."); return }
        let permissions = workspace.sourceAccess.directory
        let path = operation.path, window = operation.window
        let verificationStorage=workspace.workspaceDirectory,verificationAccounting=workspace.applicationDirectory,verificationLimit=operation.storageLimitBytes
        statusMessage = "Verifying the sampled source bytes…"
        setInspectionStatus(operation, phase: .inspecting, detail: statusMessage)
        operation.verificationTask = Task { [self] in
            do {
                try Task.checkCancellation()
                let verification = Task.detached(priority: .userInitiated) {
                    if result.collection != nil {
                        let captured=try FileInputCollectionCapture.capture(path:path,configuration:operation.config,workspace:verificationStorage,accountingRoot:verificationAccounting,sourcePermissions:permissions,storageLimitBytes:verificationLimit,previewLimit:8)
                        defer{captured.remove()};guard captured.source.sha256==result.sourceSHA256 else {throw WorkflowArchiveError("File collection changed during inspection. Refresh the source.")}
                    } else {
                        try SourceInspectionCapture.verifyCurrent(path: path, sourcePermissions: permissions,
                            window: window, expectedSHA256: result.sourceSHA256, completeFile: (try? TableInputConfiguration.format(of: operation.config)) != .csv)
                    }
                }
                try await withTaskCancellationHandler(operation: { try await verification.value }, onCancel: { verification.cancel() })
                try Task.checkCancellation()
                guard contextCurrent(operation) else { throw DataFlowContractError("Source context changed during final verification. Refresh its columns.") }
                _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
                headers[operation.address] = Header(path: operation.path, config: operation.config,
                    bindingRevision: operation.bindingRevision, snapshotSHA256: result.sourceSHA256, columns: result.columns, result: result, completedAt: Date())
                finish(operation, message: "Recorded columns from a bounded source sample. Later records are not inspected; source changes require an explicit refresh.", phase: .verified)
            } catch { finish(operation, message: error.localizedDescription) }
        }
    }
    private func finish(_ operation: Inspection, message: String, phase: SourceInspectionStatus.Phase = .failed) {
        guard inspection === operation else { return }
        inspection = nil; operation.worker = nil; operation.verificationTask = nil; isInspecting = false
        if let workspace = operation.workspace,
           operation.reservationID == workspace.dataPreparationReservationID {
            workspace.isPreparingData = false; workspace.cancelDataPreparation = nil
            owner.setExecutionBusy(false)
        }
        operation.workspace = nil
        statusMessage = operation.cancelled ? "Source inspection cancelled." : (CSVSourceDiagnostic.decodeTransport(message)?.detail ?? message)
        setInspectionStatus(operation, phase: operation.cancelled ? .cancelled : phase,
            detail: operation.cancelled ? statusMessage : message)
    }
    func cancel() {
        databaseCaptureTask?.cancel()
        guard let operation = inspection else { return }
        operation.cancelled = true
        operation.launchTask?.cancel(); operation.verificationTask?.cancel(); operation.worker?.cancel()
        statusMessage = "Cancelling source inspection; waiting for active work to finish…"
        setInspectionStatus(operation, phase: .inspecting, detail: statusMessage)
    }
    func invalidate() {
        active = false; cancel(); observations.removeAll(); outputDestinationObservation = nil; objectWillChange.send()
    }
    private func matchingOutputDestinations(for address: WorkflowNodeAddress) -> DataOutputDestinationProvider? {
        guard isCurrent, address.domain == .data,
              owner.document.nodes.contains(where: { node in
                  guard node.address == address, case .dataOperator("kwip.data.outputDelimited", _) = node.payload else { return false }; return true
              }), let outputDestinations, outputDestinations.projectID == projectID,
              outputDestinations.sessionID == sessionID else { return nil }
        return outputDestinations
    }
    func outputDestinationSummary(for address: WorkflowNodeAddress) -> String {
        matchingOutputDestinations(for: address)?.summary(for: address) ?? "Output destination selection is unavailable in this editor."
    }
    @discardableResult func chooseOutputDestination(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool {
        guard !isBusy, owner.isCurrent(lease), owner.selection.nodeAddresses == [address],
              let destinations = matchingOutputDestinations(for: address) else {
            statusMessage = "Select an idle table output in this workflow before choosing a destination."; return false
        }
        let chosen = destinations.chooseDestination(for: address, lease: lease)
        statusMessage = destinations.statusMessage
        return chosen
    }
    @discardableResult func chooseOutputDestinations(for addresses:Set<WorkflowNodeAddress>,lease:WorkflowEditorLease)->Bool {
        guard !addresses.isEmpty,!isBusy,owner.isCurrent(lease),owner.selection.nodeAddresses==addresses,
              let destinations=outputDestinations,addresses.allSatisfy({matchingOutputDestinations(for:$0) === destinations}) else {
            statusMessage="Select idle table reports in this workflow before choosing a folder.";return false
        }
        let chosen=destinations.chooseDestinations(for:addresses,lease:lease)
        statusMessage=destinations.statusMessage;return chosen
    }
    private func setInspectionStatus(_ operation: Inspection, phase: SourceInspectionStatus.Phase, detail: String) {
        setInspectionStatus(for: operation.address, lease: operation.lease,
            bindingRevision: operation.bindingRevision, path: operation.path, config: operation.config,
            phase: phase, detail: detail)
    }
    private func setInspectionStatus(for address: WorkflowNodeAddress, lease: WorkflowEditorLease,
                                     bindingRevision: UInt64, path: String?, config: [String: JSONValue],
                                     phase: SourceInspectionStatus.Phase, detail: String) {
        guard isCurrent, owner.isCurrent(lease), bindings.revision == bindingRevision,
              bindings.path(for: address) == path, sourceNode(address).map(self.config) == config else { return }
        objectWillChange.send()
        inspectionStatuses[address] = .init(lease: lease, bindingRevision: bindingRevision,
            path: path, config: config, status: .init(phase: phase, detail: detail))
    }
    func sourceInspectionStatus(for address: WorkflowNodeAddress) -> SourceInspectionStatus? {
        guard isCurrent, let record = inspectionStatuses[address], owner.isCurrent(record.lease),
              record.bindingRevision == bindings.revision, bindings.path(for: address) == record.path,
              sourceNode(address).map(config) == record.config else { return nil }
        return record.status
    }
    /// Cached verified evidence only. Refresh explicitly revalidates external source bytes.
    func sourceInspectionCompletedAt(for address: WorkflowNodeAddress) -> Date? {
        guard sourceInspectionPreview(for: address) != nil else { return nil }
        return headers[address]?.completedAt
    }
    func sourceInspectionPreview(for address: WorkflowNodeAddress) -> SourceInspectionResult? {
        guard isCurrent, let header = headers[address],
              header.bindingRevision == bindings.revision,
              bindings.path(for: address) == header.path,
              sourceNode(address).map(config) == header.config,
              header.result.sourceSHA256 == header.snapshotSHA256 else { return nil }
        return header.result
    }
    func bindingDisplayName(for address: WorkflowNodeAddress) -> String? {
        guard isCurrent else { return nil }
        return bindings.path(for: address).map { URL(fileURLWithPath: $0).lastPathComponent }
    }
    func recordedSchema(for address: WorkflowNodeAddress, in document: WorkflowDocument) -> [DataColumnSchema] {
        projectedSchema(for:address,in:document,inputPort:nil)
    }
    func recordedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument,inputPort:String)->[DataColumnSchema] {
        projectedSchema(for:address,in:document,inputPort:inputPort)
    }
    private func projectedSchema(for address:WorkflowNodeAddress,in document:WorkflowDocument,inputPort:String?)->[DataColumnSchema] {
        guard isCurrent, document == owner.document else { return [] }
        let isTrain=address.domain == .control && document.nodes.contains {node in guard node.address==address else {return false};switch node.payload {case .train,.estimator:return true;default:return false}}
        guard address.domain == .data || isTrain else {return []}
        var queryID=address.localID,queryPort=inputPort
        var nodes = document.nodes.compactMap { node -> DataFlowNode? in
            guard node.address.domain == .data, case .dataOperator(let id, let configuration) = node.payload else { return nil }
            return .init(id: node.address.localID, operatorID: id, config: configuration)
        }
        var edges = document.edges.filter { $0.from.address.domain == .data && $0.to.address.domain == .data }.map {
            DataFlowEdge(from: .init(node: $0.from.address.localID, port: $0.from.port), to: .init(node: $0.to.address.localID, port: $0.to.port))
        }
        if isTrain {
            guard let role=inputPort,["training","validation","table"].contains(role),let edge=document.edges.first(where:{$0.to.address==address && $0.to.port==role && $0.kind == .table && $0.from.address.domain == .data}) else {return []}
            queryID="__train-mapping-schema-"+address.localID
            guard !nodes.contains(where:{$0.id==queryID}) else {return []}
            nodes.append(.init(id:queryID,operatorID:"kwip.data.view",config:[:]))
            edges.append(.init(from:.init(node:edge.from.address.localID,port:edge.from.port),to:.init(node:queryID,port:"table")))
            queryPort="table"
        }
        var predictionSchemas:[String:[DataColumnSchema]]=[:]
        for (index,edge) in document.edges.enumerated() where edge.to.address.domain == .data && edge.from.address.domain == .control && ["predictions","profiles","confusion","table"].contains(edge.from.port) && edge.kind == .table {
            guard let schema=capturedPredictionSchema?(edge.from) else {continue}
            let id="__prediction-schema-\(index)"
            guard !nodes.contains(where:{$0.id==id}) else {return []}
            nodes.append(.init(id:id,operatorID:DataFlowSourceOperators.capturedTable,config:["sha256":.string(String(repeating:"0",count:64))]))
            edges.append(.init(from:.init(node:id,port:"table"),to:.init(node:edge.to.address.localID,port:edge.to.port)))
            predictionSchemas[id]=schema
        }
        return DataNodeSchemaProjection.upstreamSchema(for: queryID, in: .init(nodes: nodes, edges: edges), inputPort:queryPort, sourceSchema: { localID in
            if let schema=predictionSchemas[localID] {return schema}
            let source = WorkflowNodeAddress(domain: .data, localID: localID)
            guard let header = self.headers[source], header.bindingRevision == self.bindings.revision,
                  self.bindings.path(for: source) == header.path,
                  self.sourceNode(source).map(self.config) == header.config else { return [] }
            return header.result.columnSchema ?? header.columns.map { .init(name: $0, dtype: "string", nullable: nil) }
        },capturedSchema:capturedSchema)
    }
}


extension WorkflowSourceProvider {
    /// Explicit Run Flow refresh; the normal worker receives only the pinned IPC snapshot.
    func releaseScheduledSources(){for snapshot in scheduledSnapshots {snapshot.finish(retain:false)};scheduledSnapshots=[]}
    func refreshAPISnapshotsForRun(runID:UUID,budget:ExecutionBudgetWindow,sourceAddresses:Set<WorkflowNodeAddress>?=nil,acceptedPolicy:ResolvedExecutionPreferences?=nil) async throws {
        apiFailure=nil
        guard let workspace,!isBusy,!owner.hasPendingInspectorEdits else {throw WorkflowArchiveError("Use the idle current canvas and apply pending edits before fetching API sources.")}
        let lease=owner.lease(),document=owner.document
        let available=document.nodes.filter {if case .dataOperator(let id,let c)=$0.payload {return id==DataFlowSourceOperators.tableInput && c["format"] == .string("api")};return false}
        if let sourceAddresses, !sourceAddresses.isSubset(of:Set(available.map(\.address))) {throw DataFlowContractError("The requested API source scope is no longer available.")}
        let nodes=available.filter {sourceAddresses?.contains($0.address) ?? true}
        guard !nodes.isEmpty else{return}
        let policy=try acceptedPolicy ?? workspace.executionPolicy(for:projectID)
        isInspecting=true;owner.setExecutionBusy(true);workspace.isPreparingData=true
        let reservation=workspace.dataPreparationReservationID
        defer {isInspecting=false;if workspace.dataPreparationReservationID==reservation {workspace.isPreparingData=false;workspace.cancelDataPreparation=nil;owner.setExecutionBusy(false)}}
        for node in nodes {
            try Task.checkCancellation()
            guard owner.isCurrent(lease),owner.document==document else {throw CancellationError()}
            guard case .dataOperator(_,let config)=node.payload,case .object(let raw)=config["options"] else {throw WorkflowFileLeaseError.unsafePath}
            var pendingSnapshot:DatabaseSourceSnapshot?
            var accepted=false
            var stage=WorkflowAPICaptureStage.configuration
            defer {if workspace.scheduledExecutionID != nil,accepted,let snapshot=pendingSnapshot {scheduledSnapshots.append(snapshot)}else {pendingSnapshot?.finish(retain:accepted)}}
            do {
                let c=try APITableSettings.normalized(raw)
                stage = .connection
                guard let store=databaseOutputs else {throw WorkflowArchiveError("API connection authority is unavailable.")}
                let secret=try store.secret(for:node.address,config:c,write:false)
                statusMessage="Fetching API source · "+(node.name.isEmpty ? "Table Input":node.name)
                stage = .snapshot
                let snapshot=try await Task.detached {try DatabaseSourceSnapshot(workspace:workspace.workspaceDirectory,accountingRoot:workspace.applicationDirectory,storageLimitBytes:policy.storageLimitBytes)}.value
                pendingSnapshot=snapshot
                let revision=bindings.revision
                stage = .broker
                let receipt=try await DatabaseSourceBrokerClient().capture(options:c,credentials:secret,output:snapshot.output,memoryLimit:policy.budget.resources?.maximumResidentBytes,operationSeconds:Int(try budget.remainingSeconds(atUptime:ProcessInfo.processInfo.systemUptime)))
                stage = .verification
                try await Task.detached {try snapshot.verify(receipt:receipt)}.value
                try Task.checkCancellation()
                guard owner.isCurrent(lease),owner.document==document,bindings.revision==revision else {throw CancellationError()}
                stage = .publication
                if workspace.scheduledExecutionID==nil {try rememberSource(snapshot.destination)}
                guard bindings.commitCapturedPath(snapshot.destination.path,for:node.address,lease:lease,expectedRevision:revision,workspace:workspace,reservationID:reservation) else {throw WorkflowArchiveError(bindings.error ?? "API source changed during capture.")}
                if let r=try? JSONDecoder().decode([String:JSONValue].self,from:receipt),case .object(let fetch)=r["apiFetch"] {apiFetches[node.address]=(config,fetch)}
                accepted=true;headers[node.address]=nil;inspectionStatuses[node.address]=nil
            } catch {
                if !Task.isCancelled {recordAPIFailure(node.address,run:runID,stage:stage)}
                setInspectionStatus(for:node.address,lease:lease,bindingRevision:bindings.revision,path:bindings.path(for:node.address),config:config,phase:Task.isCancelled ? .cancelled:.failed,detail:stage.message)
                throw error
            }
        }
    }
}
