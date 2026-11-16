import Foundation
import Combine
import KwipMLCore

/// Borrowed source authority. The projection and undo entries are presentation/history;
/// the existing draft session remains the only editable and persisted lesson source.
@MainActor final class ModuleCompositionLearningCanvasAuthority: ObservableObject, ModelCanvasAuthority {
    private let drafts: TensorLearningDraftSession
    private let registry: ComponentRegistry
    private let current: () -> Bool
    private let busy: () -> Bool
    private let draftIdentity: UUID
    private let task: ModuleCompositionLearningTask
    private var observation: AnyCancellable?
    private var editRevision: UInt64 = 0
    private var undoStates: [ModelAuthoringDocument] = []
    private var redoStates: [ModelAuthoringDocument] = []
    private var retired = false
    @Published private var projection: ModelProjectDocument
    @Published private(set) var error: String?
    let projectID: String
    let sessionID = UUID()
    var isBusy: Bool { retired || busy() || !current() || drafts.identity != draftIdentity || drafts.draft?.projectID != projectID || drafts.draft?.operation != task.rawValue }
    var allowsRootCanvas: Bool { true }
    var canvasDocument: ModelProjectDocument { projection }
    var canvasChanges: AnyPublisher<Void,Never> { objectWillChange.eraseToAnyPublisher() }
    var canvasDocuments: AnyPublisher<ModelProjectDocument,Never> { $projection.eraseToAnyPublisher() }
    var editStatus: String { drafts.hasUnsavedChanges ? "Changes need saving." : "Saved with this lesson." }

    init(session: TensorLearningDraftSession, registry: ComponentRegistry,
         isCurrent: @escaping () -> Bool, isBusy: @escaping () -> Bool) throws {
        guard let draft = session.draft, let task = ModuleCompositionLearningTask(rawValue:draft.operation),
              let model = draft.compositionModel else { throw LearningContractError("Open a current composition lesson first.") }
        try ModuleCompositionLearningDraft.validate(model:model,task:task,projectID:draft.projectID,complete:false)
        drafts=session;self.registry=registry;current=isCurrent;busy=isBusy;self.task=task
        draftIdentity=session.identity;projectID=draft.projectID;projection = .blockModel(model)
        observation=session.$draft.sink { [weak self] value in
            guard let self, self.drafts.identity == self.draftIdentity,
                  value?.projectID == self.projectID,value?.operation == self.task.rawValue,
                  let model=value?.compositionModel else { return }
            let next=ModelProjectDocument.blockModel(model)
            if next != self.projection {
                if self.editRevision < UInt64.max { self.editRevision += 1 }
                self.projection=next
            }
        }
    }
    func invalidate() { retired=true;objectWillChange.send() }
    func modelCanvasRegistry() throws -> ComponentRegistry { registry }
    func lease(path:[String]) -> ModelEditorLease { .init(projectID:projectID,sessionID:sessionID,editRevision:editRevision,instancePath:path) }
    func scopeCanUndo(path:[String])->Bool { path.isEmpty && !isBusy && !hasDraft && !undoStates.isEmpty }
    func scopeCanRedo(path:[String])->Bool { path.isEmpty && !isBusy && !hasDraft && !redoStates.isEmpty }
    private var hasDraft:Bool { drafts.draft?.compositionModel?.drafts.isEmpty != true }

    func applyRootCommand(_ command:GraphCommand,selection:GraphSelection,lease token:ModelEditorLease)->GraphCommandResult? {
        do {
            let model=try checked(token,root:true)
            let result=try ModelRootEditing.apply(command,to:.blockModel(model),selection:selection,registry:registry)
            guard case .blockModel(let next)=result.document else {throw failure("Keep this lesson's Block model source.")}
            guard try persist(next,history:model) else {return nil}
            return result.command
        } catch {self.error=error.localizedDescription;return nil}
    }
    func editRootScope(_ mutation:(inout CompositeGraph)->Void,lease token:ModelEditorLease)->Bool {
        perform {
            var model=try checked(token,root:true)
            guard model.drafts.isEmpty else {throw failure("Apply or discard the open Block draft first.")}
            let before=model;mutation(&model.source.root)
            if model != before {try advance(&model)}
            return try persist(model,history:before)
        }
    }
    func beginDraft(path:[String],lease token:ModelEditorLease)->Bool {
        perform {let model=try checked(token,root:true);return try persist(ModelBlockEditing.beginDraft(model,path:path))}
    }
    func updateDraft(_ proposed:CompositeDefinition,lease token:ModelEditorLease)->Bool {
        perform {
            let model=try checked(token);_ = try draft(model,token:token)
            return try persist(ModelBlockEditing.updateDraft(model,proposed:proposed,expectedBaseRevision:model.revision,path:token.instancePath))
        }
    }
    func editDraftScope(_ mutation:(inout CompositeGraph)->Void,lease token:ModelEditorLease)->Bool {
        perform {
            let model=try checked(token);var proposed=try draft(model,token:token).proposed
            mutation(&proposed.body)
            return try persist(ModelBlockEditing.updateDraft(model,proposed:proposed,expectedBaseRevision:model.revision,path:token.instancePath))
        }
    }
    func applyDraftCommand(_ command:GraphCommand,selection:GraphSelection,lease token:ModelEditorLease)->GraphCommandResult? {
        do {
            let model=try checked(token);var proposed=try draft(model,token:token).proposed
            let result=try ModelScopeEditing.apply(command,to:proposed.body,definitions:model.source.definitions,name:proposed.title,selection:selection,registry:registry)
            proposed.body=result.scope
            let next=try ModelBlockEditing.updateDraft(model,proposed:proposed,expectedBaseRevision:model.revision,path:token.instancePath)
            guard try persist(next) else {return nil};return result.command
        } catch {self.error=error.localizedDescription;return nil}
    }
    func applyDraft(lease token:ModelEditorLease)->Bool {
        perform {
            let model=try checked(token);_ = try draft(model,token:token)
            let next=try ModelBlockEditing.applyDraft(model,path:token.instancePath,expectedBaseRevision:model.revision,
                registry:registry,definitionValidator:ModuleCompositionLearningDraft.definitionValidator(task:task))
            let baseline=try ModelBlockEditing.discardDraft(model,path:token.instancePath,expectedBaseRevision:model.revision)
            return try persist(next,history:baseline)
        }
    }
    func discardDraft(lease token:ModelEditorLease)->Bool {
        perform {
            let model=try checked(token);_ = try draft(model,token:token)
            return try persist(ModelBlockEditing.discardDraft(model,path:token.instancePath,expectedBaseRevision:model.revision))
        }
    }
    func scopeUndo(lease token:ModelEditorLease)->Bool { restore(token,undo:true) }
    func scopeRedo(lease token:ModelEditorLease)->Bool { restore(token,undo:false) }
    private func restore(_ token:ModelEditorLease,undo:Bool)->Bool {
        perform {
            let model=try checked(token,root:true)
            guard model.drafts.isEmpty,var previous=(undo ? undoStates:redoStates).last else {throw failure("Resolve the open Block draft before using Undo or Redo.")}
            previous.revision=model.revision;try advance(&previous)
            guard try persist(previous) else {return false}
            if undo {undoStates.removeLast();redoStates.append(model)} else {redoStates.removeLast();undoStates.append(model)}
            return true
        }
    }
    private func checked(_ token:ModelEditorLease,root:Bool=false)throws->ModelAuthoringDocument {
        guard !isBusy else {throw failure("Finish the current check or reopen this lesson before editing.")}
        guard token.projectID == projectID,token.sessionID == sessionID,token.editRevision == editRevision,
              !root || token.instancePath.isEmpty,let model=drafts.draft?.compositionModel,
              projection == .blockModel(model),editRevision < UInt64.max else {
            throw failure("This editor refers to an earlier lesson state. Reopen the Block before applying changes.")
        }
        return model
    }
    private func draft(_ model:ModelAuthoringDocument,token:ModelEditorLease)throws->ModelBlockDraft {
        guard let value=model.drafts.first,value.projectID == projectID,value.instancePath == token.instancePath,value.baseRevision == model.revision else {
            throw failure("This editor does not own the current Block draft.")
        }
        return value
    }
    private func advance(_ model:inout ModelAuthoringDocument)throws {
        guard model.revision < UInt64.max else {throw failure("The lesson model revision limit has been reached.")};model.revision += 1
    }
    private func persist(_ next:ModelAuthoringDocument,history:ModelAuthoringDocument?=nil)throws->Bool {
        try ModuleCompositionLearningDraft.validate(model:next,task:task,projectID:projectID,complete:false)
        guard next != drafts.draft?.compositionModel else {error=nil;return true}
        guard drafts.update({$0.compositionModel=next}) else {throw failure(drafts.errorMessage ?? "The lesson change could not be applied.")}
        if let history {
            undoStates.append(history);if undoStates.count>100 {undoStates.removeFirst(undoStates.count-100)};redoStates.removeAll()
        }
        // Failed atomic saves retain local edits in the existing session; they never
        // publish a second source or silently abandon the user's draft.
        guard drafts.save() else {error=drafts.errorMessage;return false}
        error=nil;return true
    }
    private func perform(_ action:()throws->Bool)->Bool {do{return try action()}catch{self.error=error.localizedDescription;return false}}
    private func failure(_ message:String)->LearningContractError {.init(message)}
}
