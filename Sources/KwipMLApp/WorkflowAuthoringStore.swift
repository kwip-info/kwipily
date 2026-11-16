import Foundation
import Combine
import KwipMLCore

struct WorkflowEditorLease: Equatable {
    let projectID: String
    let sessionID: UUID
    let generation: UInt64
}

struct WorkflowBlockLease: Equatable {
    let owner: WorkflowEditorLease
    let root: WorkflowNodeAddress
    let nestedPath: [String]
}

/// The sole mutable source for a workflow. Persisted transactions precede published edits;
/// execution callbacks borrow a lease rather than a second mutable model or data document.
@MainActor final class WorkflowAuthoringStore: ObservableObject {
    /// Scoped to synchronous document publication; never applies to configuration or wiring edits.
    private(set) var isPublishingLayoutMove = false
    let projectID: String
    let sessionID = UUID()
    @Published private(set) var document: WorkflowDocument
    @Published private(set) var selection = WorkflowSelection()
    @Published private(set) var error: String?
    @Published private var retiredToolRepairDetail: String?
    var retiredToolRepairMessage: String? {
        (RetiredToolAuthoringUpgrade.requiresUpgrade(document) || document.nodes.contains {if case .modelDefinition=$0.payload {return true};return false}) ? retiredToolRepairDetail : nil
    }
    @Published private(set) var isBusy = false
    @Published private var pendingDataEdits = Set<DataEditToken>()
    var hasPendingInspectorEdits: Bool { !pendingDataEdits.isEmpty }
    var hasPendingDataEdits: Bool { hasPendingInspectorEdits }
    private struct DataEditToken: Hashable {
        let contextID: UUID
        let address: WorkflowNodeAddress
        let key: String
    }
    /// Registration requires current editing authority. Existing tokens survive the
    /// originating context's own commits until that field commits or the context retires.
    @discardableResult func setInspectorEditPending(contextID: UUID, address: WorkflowNodeAddress,
                                               key: String, pending: Bool, lease token: WorkflowEditorLease) -> Bool {
        guard isCurrent(token), !isBusy, document.drafts.isEmpty,
              selection.nodeAddresses == [address], selection.edgeIDs.isEmpty,
              !key.isEmpty, key.count <= 128,
              inlineSelectionIsValid(address) else { return false }
        let item = DataEditToken(contextID: contextID, address: address, key: key)
        if pending {
            guard pendingDataEdits.contains(item) || pendingDataEdits.count < 512 else { return false }
            pendingDataEdits.insert(item)
        } else { pendingDataEdits.remove(item) }
        return true
    }
    /// UUID-scoped cleanup cannot remove a newer context's pending fields, even if
    /// this editor has already advanced past the retired context's source lease.
    func clearDataEdits(contextID: UUID) {
        pendingDataEdits = pendingDataEdits.filter { $0.contextID != contextID }
    }

    @discardableResult func setDataEditPending(contextID:UUID,address:WorkflowNodeAddress,key:String,pending:Bool,lease:WorkflowEditorLease)->Bool {
        guard address.domain == .data else {return false}
        return setInspectorEditPending(contextID:contextID,address:address,key:key,pending:pending,lease:lease)
    }
    func clearInspectorEdits(contextID:UUID) {clearDataEdits(contextID:contextID)}

    private let persistence: WorkflowSourcePersistence
    private(set) var canvasVersions: CanvasVersionJournal?
    private var runHistoryStore:RunHistoryPolicyStore?
    @Published private(set) var runHistoryPolicyError:String?
    @Published private(set) var runHistoryPolicy=RunHistoryPolicy.all
    func setRunHistoryPolicy(_ policy:RunHistoryPolicy)throws {
        guard active,!isBusy,!hasPendingInspectorEdits,let store=runHistoryStore else {throw Failure("Finish the active run and drafts before changing History capture.")}
        try store.save(policy);runHistoryPolicy=policy;runHistoryPolicyError=nil
    }
    func captureHistoryDecision(id:String,policy:RunHistoryPolicy?=nil)throws {
        guard runHistoryPolicyError==nil else {throw Failure("Run History policy is unreadable. Review capture settings before running.")}
        try runHistoryStore?.capture(id:id,policy:policy ?? runHistoryPolicy)
    }
    func configureCanvasVersions(root:URL)throws {
        let store=RunHistoryPolicyStore(root:root,projectID:projectID)
        runHistoryStore=store
        do {runHistoryPolicy=try store.load();runHistoryPolicyError=nil}
        catch {runHistoryPolicyError=error.localizedDescription;throw error}
        let journal=try CanvasVersionJournal(root:root,projectID:projectID)
        if journal.versions.isEmpty && document.drafts.isEmpty {_ = try journal.save(document,label:"Initial canvas",dependencies:pinnedFlowDocument?.dependencies)}
        canvasVersions=journal
    }
    func saveCanvasVersion(label:String="")throws {
        guard active,!isBusy,!hasPendingInspectorEdits,document.drafts.isEmpty else {throw Failure("Finish the active run and apply or revert drafts before saving a canvas version.")}
        guard let canvasVersions else {throw Failure("Canvas History is unavailable. Reopen this project before saving.")}
        try requireSavedCanvasAuthority()
        _ = try canvasVersions.save(document,label:label,dependencies:pinnedFlowDocument?.dependencies)
        objectWillChange.send()
    }
    @discardableResult func restoreCanvasVersion(_ id:String,lease token:WorkflowEditorLease)->Bool {
        do {
            try check(token)
            guard !hasPendingInspectorEdits,document.drafts.isEmpty,let journal=canvasVersions else {throw Failure("Apply or revert drafts before restoring a canvas version.")}
            try requireSavedCanvasAuthority()
            var next=try journal.document(version:id)
            guard !RetiredToolAuthoringUpgrade.requiresUpgrade(next),next.schemaVersion==document.schemaVersion else {throw Failure("This historical canvas needs a separate compatibility upgrade. The current canvas is unchanged.")}
            guard document.revision<UInt64.max else {throw Failure("Canvas revision limit reached.")}
            next.revision=document.revision+1
            let previous=historyState(document:document)
            var envelope=pinnedFlowDocument
            if envelope != nil {
                guard let dependencies=journal.versions.first(where:{$0.id==id})?.dependencies else {throw Failure("This version predates file-backed container pins. Open a recovery copy to upgrade it explicitly.")}
                envelope?.dependencies=dependencies
                envelope?.draftDependencies=[]
            }
            try journal.requireRestoreCapacity(current:document,target:next,currentDependencies:pinnedFlowDocument?.dependencies,targetDependencies:envelope?.dependencies)
            // Preserve the present source before changing canonical owner authority.
            _ = try journal.save(document,label:"Before restore",dependencies:pinnedFlowDocument?.dependencies)
            try persist(next,restoring:envelope)
            append(previous,to:&undoStates);redoStates.removeAll()
            generation+=1;document=next;selection=WorkflowSelection()
            do {_ = try journal.save(next,label:"Restored version",dependencies:pinnedFlowDocument?.dependencies);error=nil}
            catch {self.error="Canvas restored; saving its History checkpoint failed: "+error.localizedDescription}
            return true
        } catch {self.error=error.localizedDescription;return false}
    }
    private func requireSavedCanvasAuthority()throws {
        if let root=flowRootURL,let pinned=pinnedFlowDocument {
            guard try FlowDocumentCodec.reference(FlowDocumentStore.readHead(at:root))==FlowDocumentCodec.reference(pinned) else {throw Failure("This canvas changed in another window. Reopen it before saving or restoring History.")}
        } else {
            let url=URL(fileURLWithPath:try persistence.currentRecord().path)
            let bytes=try DiagnosticFileIO.read(root:url.deletingLastPathComponent(),parts:[url.lastPathComponent],limit:WorkflowCodec.maximumBytes)
            let stored=try WorkflowCodec.decode(bytes)
            guard try WorkflowCodec.encode(stored)==WorkflowCodec.encode(document) else {throw Failure("This canvas changed in another window. Reopen it before saving or restoring History.")}
        }
    }
    var flowRootURL: URL? { persistence.fileURL }
    var pinnedFlowDocument: FlowDocument? { persistence.state.flowDocument }
    func currentProjectRecord() throws -> ProjectRecord { try persistence.currentRecord() }
    let modelRegistry: ComponentRegistry
    let dataRegistry: DataFlowOperatorRegistry
    private var generation: UInt64 = 0
    private var active = true
    private struct State {
        let document: WorkflowDocument
        let selection: WorkflowSelection
        let envelope: FlowDocument?
    }
    var inlineOwnedDefinitions:[CompositeDefinitionReference:String]=[:]
    private var undoStates: [State] = []
    private var redoStates: [State] = []
    private var historyAvailable: Bool {
        active && !isBusy && document.drafts.isEmpty && document.revision < UInt64.max && generation < UInt64.max
    }
    var canUndo: Bool { historyAvailable && !undoStates.isEmpty }
    var canRedo: Bool { historyAvailable && !redoStates.isEmpty }

    init(library: ProjectLibrary, projectID: String, modelRegistry: ComponentRegistry,
         dataRegistry: DataFlowOperatorRegistry) throws {
        let persistence = try WorkflowSourcePersistence(library: library, projectID: projectID)
        let document = persistence.state.document
        self.persistence = persistence
        self.projectID = document.projectID
        self.document = document
        self.modelRegistry = modelRegistry
        self.dataRegistry = dataRegistry
    }

    /// Called only when opening the current editable project, never for History or
    /// replay. A recovery copy precedes the atomic save; migration is not an Undo
    /// step that could reintroduce retired controls into the current editor.
    @discardableResult func migrateDataLoadingForAuthoring() throws -> Bool {
        try migrateRetiredToolsForAuthoring()
    }

    @discardableResult func migrateRetiredToolsForAuthoring() throws -> Bool {
        guard active, !isBusy, undoStates.isEmpty, redoStates.isEmpty,
              document.revision < UInt64.max else { throw Failure("Reopen the project before upgrading retired tools.") }
        var next: WorkflowDocument
        do { next = try RetiredToolAuthoringUpgrade.plan(document)
            if next.guidedBuild == nil || (next.guidedBuild?.courseID == "kwip.course.universal-classifier" && (next.guidedBuild?.version ?? 0)>=4) {next=try WorkflowModelBoundary.consolidate(next)} }
        catch {
            retiredToolRepairDetail = error.localizedDescription
            return false
        }
        guard next != document else { return false }
        let source = try persistence.currentRecord()
        let sourceURL = URL(fileURLWithPath: source.path)
        let backup = sourceURL.appendingPathExtension("before-retired-tools-r\(document.revision)")
        let bytes = try DiagnosticFileIO.read(root: sourceURL.deletingLastPathComponent(), parts: [sourceURL.lastPathComponent], limit: WorkflowCodec.maximumBytes)
        if FileManager.default.fileExists(atPath: backup.path) {
            guard try DiagnosticFileIO.read(root: backup.deletingLastPathComponent(), parts: [backup.lastPathComponent], limit: WorkflowCodec.maximumBytes) == bytes else { throw Failure("A different retired-tool recovery copy already exists; the original project is unchanged.") }
        } else { try bytes.write(to: backup, options: [.withoutOverwriting]) }
        guard try DiagnosticFileIO.read(root: backup.deletingLastPathComponent(), parts: [backup.lastPathComponent], limit: WorkflowCodec.maximumBytes) == bytes else { throw Failure("The retired-tool recovery copy could not be verified; the original project is unchanged.") }
        next.revision = document.revision + 1
        try persist(next)
        generation += 1; document = next; error = nil; retiredToolRepairDetail = nil
        return true
    }

    func useAuthoringRetentionPreferences(at url: URL) {
        let store = AppPreferencesStore(fileURL: url)
        persistence.authoringRetentionPreferences = {
            do { return try store.load().effectiveAuthoringRetention }
            catch { throw AppPreferencesError.invalid("Authoring evidence policy could not be read. Review KwipML → Settings before retrying. " + error.localizedDescription) }
        }
    }

    func lease() -> WorkflowEditorLease {
        .init(projectID: projectID, sessionID: sessionID, generation: generation)
    }
    /// Busy does not invalidate a captured execution's completion callback.
    func isCurrent(_ token: WorkflowEditorLease) -> Bool {
        active && token.projectID == projectID && token.sessionID == sessionID && token.generation == generation
    }
    func setExecutionBusy(_ value: Bool) {
        guard active else { return }
        isBusy = value
    }
    func invalidate() {
        active = false
        pendingDataEdits.removeAll()
        undoStates.removeAll(); redoStates.removeAll()
        isBusy = false
    }

    @discardableResult func select(_ next: WorkflowSelection, lease token: WorkflowEditorLease) -> Bool {
        do {
            guard isCurrent(token) else { throw Failure("This selection refers to an earlier workflow session or state.") }
            selection = WorkflowSelection(
                nodeAddresses: Set(next.nodeAddresses.filter {inlineSelectionIsValid($0)}),
                edgeIDs: Set(next.edgeIDs.filter {id in document.edges.contains {$0.id==id} || WorkflowInlineContainerProjection.internalEdge(in:document,displayID:id) != nil}))
            error = nil
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    /// A title is source metadata: it participates in the same persisted history and
    /// lease generation as graph edits, without changing selection or any graph domain.
    @discardableResult func rename(_ name: String, lease token: WorkflowEditorLease) -> Bool {
        do {
            try check(token)
            guard document.drafts.isEmpty else {
                throw Failure("Apply or discard the open Block draft before renaming the workflow.")
            }
            guard name != document.name else { error = nil; return true }
            guard document.revision < UInt64.max else {
                throw Failure("The workflow revision limit has been reached.")
            }
            var next = document
            next.name = name
            next.revision += 1
            // Codec bounds and destination safety are checked by the existing save path.
            let previous = historyState(document: document)
            try persist(next)
            append(previous, to: &undoStates)
            redoStates.removeAll()
            generation += 1
            document = next
            error = nil
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    /// Cache policy is an explicit authoring action. A genuine first change
    /// migrates schema 1 and edits its Run setup in one persisted history step.
    @discardableResult func setPreparationCachePolicy(_ policy:WorkflowPreparationCachePolicy,
                                                       for address:WorkflowNodeAddress,
                                                       lease token:WorkflowEditorLease)->Bool {
        do {
            try check(token)
            guard !hasPendingInspectorEdits else {
                throw Failure("Commit or discard unfinished inspector values before changing preparation cache policy.")
            }
            guard document.drafts.isEmpty else {
                throw Failure("Apply or discard the open Block draft before changing preparation cache policy.")
            }
            if let index=document.nodes.firstIndex(where:{$0.address == address}),case .train(let trainConfig)=document.nodes[index].payload {
                guard address.domain == .control,document.schemaVersion >= 16,
                      var run=try WorkflowTrainRunSetup.configuration(trainConfig,schemaVersion:document.schemaVersion) else {
                    throw Failure("Reopen this project to move Run setup into Train.")
                }
                let current=try WorkflowControlContracts.preparationCachePolicy(run,schemaVersion:document.schemaVersion)
                guard current != policy else {error=nil;return true}
                guard document.revision<UInt64.max else {throw Failure("The workflow revision limit has been reached.")}
                run["preparationCachePolicy"] = .string(policy.rawValue)
                var next=document
                next.nodes[index].payload = .train(config:try WorkflowTrainRunSetup.setting(run,in:trainConfig,schemaVersion:document.schemaVersion))
                try commitControlMigration(next)
                return true
            }
            guard address.domain == .control,let index=document.nodes.firstIndex(where:{$0.address == address}),
                  case .runSetup(let config)=document.nodes[index].payload else {
                throw Failure("Choose an existing control-domain Run setup component.")
            }
            let current=try WorkflowControlContracts.preparationCachePolicy(config,schemaVersion:document.schemaVersion)
            guard current != policy else {error=nil;return true}
            guard document.revision<UInt64.max else {throw Failure("The workflow revision limit has been reached.")}
            var next=try WorkflowCodec.migrateToVersion2(document)
            guard case .runSetup(var updated)=next.nodes[index].payload else {throw Failure("Run setup changed during migration.")}
            updated["preparationCachePolicy"] = .string(policy.rawValue)
            next.nodes[index].payload = .runSetup(config:updated)
            try commitControlMigration(next)
            return true
        } catch {self.error=error.localizedDescription;return false}
    }

    /// Mode and schema migration are one persisted, undoable authoring action.
    @discardableResult func setTrainingMode(_ mode:WorkflowTrainingMode,
                                                       for address:WorkflowNodeAddress,
                                                       lease token:WorkflowEditorLease)->Bool {
        do {
            try check(token)
            guard !hasPendingInspectorEdits else {
                throw Failure("Commit or discard unfinished inspector values before changing training mode.")
            }
            guard document.drafts.isEmpty else {
                throw Failure("Apply or discard the open Block draft before changing training mode.")
            }
            guard address.domain == .control,let index=document.nodes.firstIndex(where:{$0.address == address}),
                  case .train(let config)=document.nodes[index].payload else {
                throw Failure("Choose an existing control-domain Train component.")
            }
            let current=try WorkflowControlContracts.trainingMode(config,schemaVersion:document.schemaVersion)
            guard current != mode else {error=nil;return true}
            guard document.revision<UInt64.max else {throw Failure("The workflow revision limit has been reached.")}
            var next=try WorkflowCodec.migrateToVersion3(document)
            guard case .train(var updated)=next.nodes[index].payload else {throw Failure("Train changed during migration.")}
            updated["trainingMode"] = .string(mode.rawValue)
            next.nodes[index].payload = .train(config:updated)
            try commitControlMigration(next)
            return true
        } catch {self.error=error.localizedDescription;return false}
    }

    /// Specification and schema upgrade commit together; ordinary run values remain unchanged.
    @discardableResult func setExperimentSpecification(_ specification:WorkflowExperimentSpecification?,
                                                       for address:WorkflowNodeAddress,
                                                       lease token:WorkflowEditorLease)->Bool {
        do {
            try check(token)
            guard !hasPendingInspectorEdits else {
                throw Failure("Commit or discard unfinished inspector values before changing experiments.")
            }
            guard document.drafts.isEmpty else {
                throw Failure("Apply or discard the open Block draft before changing experiments.")
            }
            if let index=document.nodes.firstIndex(where:{$0.address == address}),case .train(let trainConfig)=document.nodes[index].payload {
                guard address.domain == .control,document.schemaVersion >= 16,
                      var run=try WorkflowTrainRunSetup.configuration(trainConfig,schemaVersion:document.schemaVersion) else {
                    throw Failure("Reopen this project to move Run setup into Train.")
                }
                try specification?.validate()
                let current=try run["experiment"].map(WorkflowExperimentSpecification.decode)
                guard current != specification else {error=nil;return true}
                guard document.revision<UInt64.max else {throw Failure("The workflow revision limit has been reached.")}
                run["experiment"]=try specification.map {try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode($0))}
                var next=document
                next.nodes[index].payload = .train(config:try WorkflowTrainRunSetup.setting(run,in:trainConfig,schemaVersion:document.schemaVersion))
                try commitControlMigration(next)
                return true
            }
            guard address.domain == .control,let index=document.nodes.firstIndex(where:{$0.address == address}),
                  case .runSetup(let config)=document.nodes[index].payload else {
                throw Failure("Choose an existing control-domain Run setup component.")
            }
            try specification?.validate()
            let current=try config["experiment"].map(WorkflowExperimentSpecification.decode)
            guard current != specification else {error=nil;return true}
            guard document.revision<UInt64.max else {throw Failure("The workflow revision limit has been reached.")}
            var next=try WorkflowCodec.migrateToVersion4(document)
            var updated=config
            updated["experiment"]=try specification.map {
                try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode($0))
            }
            next.nodes[index].payload = .runSetup(config:updated)
            try commitControlMigration(next)
            return true
        } catch {self.error=error.localizedDescription;return false}
    }

    private func commitControlMigration(_ proposed:WorkflowDocument)throws {
        var next=proposed
        next.revision=document.revision+1
        let previous = historyState(document: document)
        try persist(next)
        append(previous,to:&undoStates)
        redoStates.removeAll();generation+=1;document=next;error=nil
    }

    @discardableResult func apply(_ command: WorkflowCommand, lease token: WorkflowEditorLease) -> WorkflowEditResult? {
        let profile = InteractionProfiler.shared.begin(.authoringCommand)
        defer { InteractionProfiler.shared.end(profile) }
        do {
            try check(token)
            let retainedSelection=selection
            let effectiveCommand=try authorizingInlineRetirement(inlineCommand(command))
            let result = try WorkflowEditing.apply(effectiveCommand, to: document, selection: selection,
                                                   modelRegistry: modelRegistry, dataRegistry: dataRegistry)
            if result.document != document {
                let previous = historyState(document: document)
                try persist(result.document)
                append(previous, to: &undoStates)
                redoStates.removeAll()
                generation += 1
                if case .move = command { isPublishingLayoutMove = true }
                document = result.document
                recordInlineOwnership(effectiveCommand)
                isPublishingLayoutMove = false
            }
            selection = result.selection
            switch command {
            case .rename(let address,_),.configure(let address,_),.setLayerParameters(let address,_),.setActivation(let address,_,_),.setPooling(let address,_,_),.setConvolution(let address,_,_),.setNormalization(let address,_,_),.setLinear(let address,_,_),.setRecurrent(let address,_,_),.setBilinear(let address,_),.setEmbedding(let address,_),.setEmbeddingBag(let address,_),.setAttention(let address,_),.setReshape(let address,_,_),.setTensorOperation(let address,_,_):
                if retainedSelection.nodeAddresses == [address],inlineSelectionIsValid(address) {selection=retainedSelection}
            default:break
            }
            error = nil
            return result
        } catch { self.error = error.localizedDescription; return nil }
    }

    /// MCP and other reviewed multi-command edits use the same canonical editor
    /// and persistence boundary. Validate the complete sequence before one save,
    /// source publication and Undo insertion; intermediate documents never escape.
    struct StagedBatch {
        let result:WorkflowEditResult
        let ownedDefinitions:[CompositeDefinitionReference:String]
    }
    /// Pure simulation over the prospective document. Only validated definitions
    /// created by this transaction acquire provisional retirement proof; imported
    /// references and caller-provided prefixes never gain that authority.
    func stageBatch(_ commands:[WorkflowCommand],lease token:WorkflowEditorLease)throws->StagedBatch {
        try check(token)
        guard !hasPendingInspectorEdits,!commands.isEmpty,commands.count<=100 else{throw Failure("Finish pending edits and use a bounded nonempty transaction.")}
        var next=document,selected=selection,proofs=inlineOwnedDefinitions
        for command in commands {
            let mapped=try inlineCommand(command,in:next)
            let authorized=try authorizingInlineRetirement(mapped,in:next,ownedDefinitions:proofs)
            let previous=next
            let result=try WorkflowEditing.apply(authorized,to:next,selection:selected,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
            next=result.document;selected=result.selection
            if next != previous,case .inlineContainer(let root,_,_)=authorized,
               let current=try? WorkflowInlineContainer.definition(in:next,root:root),
               current.reference.id.hasPrefix("inline-private-") {
                let retained=Set(next.definitions.map(\.reference))
                proofs=proofs.filter{retained.contains($0.key)}
                proofs[current.reference]=try inlineDefinitionHash(current)
            }
        }
        if next != document {
            guard document.revision<UInt64.max else{throw Failure("Canvas revision limit reached.")}
            next.revision=document.revision+1
        }
        return .init(result:.init(document:next,selection:selected,copies:.init()),ownedDefinitions:proofs)
    }
    @discardableResult func applyBatch(_ commands:[WorkflowCommand],lease token:WorkflowEditorLease)->WorkflowEditResult? {
        do {
            let staged=try stageBatch(commands,lease:token),next=staged.result.document
            if next != document {
                let previous=historyState(document:document)
                try persist(next)
                append(previous,to:&undoStates);redoStates.removeAll();generation+=1
                isPublishingLayoutMove=commands.allSatisfy{if case .move=$0{return true};return false}
                document=next;inlineOwnedDefinitions=staged.ownedDefinitions
                isPublishingLayoutMove=false
            }
            selection=staged.result.selection;error=nil
            return staged.result
        }catch{isPublishingLayoutMove=false;self.error=error.localizedDescription;return nil}
    }

    func blockLease(root: WorkflowNodeAddress, nestedPath: [String] = []) -> WorkflowBlockLease {
        .init(owner: lease(), root: root, nestedPath: nestedPath)
    }

    @discardableResult func beginDraft(root: WorkflowNodeAddress, nestedPath: [String] = [],
                                       lease token: WorkflowEditorLease) -> Bool {
        do {
            try check(token)
            let next = try WorkflowBlockEditing.beginDraft(document, root: root, nestedPath: nestedPath)
            try publishDraft(next)
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    @discardableResult func updateDraft(_ proposed: CompositeDefinition, lease token: WorkflowBlockLease) -> Bool {
        do {
            try checkDraft(token)
            let next = try WorkflowBlockEditing.updateDraft(document, proposed: proposed,
                expectedBaseRevision: document.revision, root: token.root, nestedPath: token.nestedPath)
            try publishDraft(next)
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    @discardableResult func discardDraft(lease token: WorkflowBlockLease) -> Bool {
        do {
            try checkDraft(token)
            let next = try WorkflowBlockEditing.discardDraft(document, root: token.root,
                nestedPath: token.nestedPath, expectedBaseRevision: document.revision)
            try publishDraft(next)
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    @discardableResult func applyDraft(lease token: WorkflowBlockLease) -> Bool {
        do {
            try checkDraft(token)
            let next = try WorkflowBlockEditing.applyDraft(document, root: token.root,
                nestedPath: token.nestedPath, expectedBaseRevision: document.revision, registry: modelRegistry)
            // Undo returns to the exact mixed source before this draft, not to an
            // unresolved proposal. No intermediate draft edits enter history.
            let baseline = try WorkflowBlockEditing.discardDraft(document, root: token.root,
                nestedPath: token.nestedPath, expectedBaseRevision: document.revision)
            var baselineEnvelope = persistence.state.flowDocument
            baselineEnvelope?.draftDependencies = []
            let previous = State(document: baseline, selection: selection, envelope: baselineEnvelope)
            try persist(next)
            append(previous, to: &undoStates)
            redoStates.removeAll()
            generation += 1
            document = next
            selection = WorkflowSelection(
                nodeAddresses: selection.nodeAddresses.intersection(Set(next.nodes.map(\.address))),
                edgeIDs: selection.edgeIDs.intersection(Set(next.edges.map(\.id))))
            error = nil
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    private func checkDraft(_ token: WorkflowBlockLease) throws {
        try check(token.owner)
        guard token.root.domain == .model,
              let draft = document.drafts.first,
              draft.projectID == projectID,
              draft.baseRevision == document.revision,
              draft.instancePath == [token.root.localID] + token.nestedPath else {
            throw Failure("This child editor does not own the current workflow Block draft.")
        }
    }

    private func publishDraft(_ next: WorkflowDocument) throws {
        guard next != document else { error = nil; return }
        var durable = next
        if persistence.fileURL != nil {
            guard document.revision < UInt64.max else { throw Failure("Parent revision is exhausted.") }
            durable.revision = document.revision + 1
            for index in durable.drafts.indices { durable.drafts[index].baseRevision = durable.revision }
        }
        try persist(durable)
        generation += 1
        document = durable
        error = nil
    }

    func flowAdoptionHead(definition: CompositeDefinitionReference, lease token: WorkflowEditorLease) throws -> FlowDocumentRevision {
        try check(token)
        return try persistence.adoptionHead(definition: definition)
    }

    func prepareFlowAdoption(definition: CompositeDefinitionReference,
                             lease token: WorkflowEditorLease) throws -> WorkflowSourcePersistence.Adoption {
        try check(token)
        guard !hasPendingInspectorEdits, document.drafts.isEmpty else {
            throw Failure("Finish pending inspector edits before reviewing a subflow update.")
        }
        let head = try persistence.adoptionHead(definition: definition)
        let proposal = try persistence.prepareAdoption(definition: definition, expectedHead: head)
        try validateFlowAdoption(proposal.after.document)
        return proposal
    }

    @discardableResult func adoptFlowRevision(_ proposal: WorkflowSourcePersistence.Adoption,
                                             lease token: WorkflowEditorLease) -> Bool {
        do {
            try check(token)
            guard !hasPendingInspectorEdits, document.drafts.isEmpty, generation < UInt64.max else {
                throw Failure("Finish pending edits before adopting this subflow update.")
            }
            try validateFlowAdoption(proposal.after.document)
            let previous = historyState(document: document)
            _ = try persistence.commitAdoption(proposal)
            append(previous, to: &undoStates); redoStates.removeAll()
            generation += 1; document = persistence.state.document; error = nil
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    private func validateFlowAdoption(_ next: WorkflowDocument) throws {
        for definition in next.definitions {
            try CompositeDefinitionValidation.validate(definition, definitions: next.definitions, registry: modelRegistry)
        }
        for node in next.nodes {
            guard case .modelBlock(let reference, let overrides) = node.payload else { continue }
            guard let definition = next.definitions.first(where: { $0.reference == reference }) else {
                throw FlowDocumentError.missingDependency
            }
            let fields = definition.parameters.map(\.field)
            guard Set(overrides.keys).isSubset(of: Set(fields.map(\.key))),
                  fields.allSatisfy({ $0.accepts(overrides[$0.key] ?? $0.defaultValue) }) else {
                throw Failure("Existing instance overrides are incompatible with this subflow update.")
            }
            for edge in next.edges {
                if edge.from.address == node.address && !definition.outputs.contains(where: { $0.id == edge.from.port }) {
                    throw Failure("This update removes an output used by an existing connection.")
                }
                if edge.to.address == node.address && !definition.inputs.contains(where: { $0.id == edge.to.port }) {
                    throw Failure("This update removes an input used by an existing connection.")
                }
            }
        }
    }

    @discardableResult func undo(lease token: WorkflowEditorLease) -> Bool { restore(token, undo: true) }
    @discardableResult func redo(lease token: WorkflowEditorLease) -> Bool { restore(token, undo: false) }

    private func restore(_ token: WorkflowEditorLease, undo: Bool) -> Bool {
        do {
            try check(token)
            guard document.drafts.isEmpty else {
                throw Failure("Apply or discard the open Block draft before using workflow history.")
            }
            guard let previous = (undo ? undoStates : redoStates).last else {
                throw Failure("No available workflow history action.")
            }
            guard document.revision < UInt64.max else { throw Failure("The workflow revision limit has been reached.") }
            var next = previous.document
            next.revision = document.revision + 1
            let current = historyState(document: document)
            try persist(next, restoring: previous.envelope)
            if undo {
                undoStates.removeLast(); append(current, to: &redoStates)
            } else {
                redoStates.removeLast(); append(current, to: &undoStates)
            }
            generation += 1
            document = next
            selection = previous.selection
            error = nil
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    private func check(_ token: WorkflowEditorLease) throws {
        guard isCurrent(token) else { throw Failure("This editor refers to an earlier workflow session or state. Reopen it before applying changes.") }
        guard !isBusy else { throw Failure("Stop the active execution before editing the workflow.") }
    }
    private func historyState(document: WorkflowDocument) -> State {
        State(document: document, selection: selection, envelope: persistence.state.flowDocument)
    }
    private func persist(_ next: WorkflowDocument, restoring envelope: FlowDocument? = nil) throws {
        guard generation < UInt64.max else { throw Failure("The workflow editor generation limit has been reached.") }
        _ = try persistence.save(next, restoring: envelope)
    }
    private func append(_ state: State, to history: inout [State]) {
        history.append(state)
        if history.count > 100 { history.removeFirst(history.count - 100) }
    }
    private struct Failure: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }
}
