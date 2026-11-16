import Foundation
import Combine
import KwipMLCore

/// Window-local tabs borrow the lesson's single persisted source owner.
@MainActor final class ModuleCompositionLearningCanvasSession: ObservableObject {
    @Published private(set) var activeCanvas: ModelCanvasAdapter?
    @Published private(set) var paths: [[String]] = [[]]
    @Published private(set) var activePath: [String] = []
    @Published private(set) var statusMessage = ""
    let historyProvider = EditorHistoryProvider()
    let authority: ModuleCompositionLearningCanvasAuthority
    private var adapters: [[String]: ModelCanvasAdapter] = [:]
    private var observation: AnyCancellable?
    private var retired = false
    init(drafts: TensorLearningDraftSession, registry: ComponentRegistry,
         isCurrent: @escaping () -> Bool, isBusy: @escaping () -> Bool) throws {
        authority = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:isCurrent,isBusy:isBusy)
        let root = try ModelCanvasAdapter(owner:authority)
        adapters[[]] = root; activeCanvas = root
        historyProvider.attach(root)
        root.onOpenBlock = { [weak self] path in self?.open(path) }
        observation = authority.objectWillChange.sink { [weak self] _ in self?.objectWillChange.send() }
        if let pending = drafts.draft?.compositionModel?.drafts.first { open(pending.instancePath) }
    }
    func open(_ path:[String]) {
        guard !retired,!authority.isBusy else {return}
        do {
            if !path.isEmpty {
                if case .blockModel(let model)=authority.canvasDocument,let pending=model.drafts.first {
                    guard pending.instancePath == path else {
                        statusMessage="Apply or discard the open Block changes before opening another Block.";return
                    }
                } else {
                    guard authority.beginDraft(path:path,lease:authority.lease()) else {
                        statusMessage=authority.error ?? "This Block could not be opened.";return
                    }
                }
            }
            let adapter:ModelCanvasAdapter
            if let existing=adapters[path] {adapter=existing}
            else {
                adapter=try ModelCanvasAdapter(owner:authority,path:path)
                adapter.onOpenBlock = { [weak self] path in self?.open(path) }
                adapters[path]=adapter;paths.append(path)
            }
            activePath=path;activeCanvas=adapter;historyProvider.attach(adapter);statusMessage=""
        } catch {statusMessage=error.localizedDescription}
    }
    @discardableResult func apply()->Bool {
        guard !retired,!activePath.isEmpty,!authority.isBusy else {return false}
        let result=authority.applyDraft(lease:authority.lease(path:activePath))
        statusMessage=result ? "Block changes applied. Other copies remain independent.":authority.error ?? "Changes could not be applied."
        return result
    }
    @discardableResult func discard()->Bool {
        guard !retired,!activePath.isEmpty,!authority.isBusy else {return false}
        let result=authority.discardDraft(lease:authority.lease(path:activePath))
        statusMessage=result ? "Unapplied Block changes discarded.":authority.error ?? "Changes could not be discarded."
        return result
    }
    func close(_ path:[String]) {
        guard !retired,!path.isEmpty,!authority.isBusy else {return}
        if case .blockModel(let model)=authority.canvasDocument,model.drafts.first?.instancePath == path {
            statusMessage="Apply or discard this Block before closing its tab.";return
        }
        adapters.removeValue(forKey:path)?.invalidate();paths.removeAll {$0 == path}
        if activePath == path {open([])}
    }
    func invalidate() {
        retired=true;historyProvider.detach();authority.invalidate();adapters.values.forEach {$0.invalidate()}
        adapters.removeAll();observation=nil;activeCanvas=nil
    }
}
