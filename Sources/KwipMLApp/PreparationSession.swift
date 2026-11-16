import Foundation
import Combine

/// App-owned session: survives conditional tab views without a WorkspaceStore ownership cycle.
@MainActor final class PreparationSession:ObservableObject {
    @Published var showPreparation=false {
        didSet { if let projectID { subsections[projectID]=showPreparation } }
    }
    @Published private(set) var current:DataFlowWorkspaceStore?
    private weak var workspace:WorkspaceStore?
    private var projectID:String?
    private var subsections:[String:Bool]=[:]
    private var observation:AnyCancellable?
    func attach(_ workspace:WorkspaceStore) {
        guard self.workspace !== workspace else{return}
        close();self.workspace=workspace;projectID=workspace.activeProjectID
        showPreparation=projectID.flatMap{subsections[$0]} ?? false
        observation=workspace.$activeProjectID.sink{[weak self] id in self?.transition(to:id)}
    }
    private func transition(to id:String?) {
        guard id != projectID else{return}
        releaseCurrent()
        projectID=id
        showPreparation=id.flatMap{subsections[$0]} ?? false
    }
    func preparation(for workspace:WorkspaceStore)->DataFlowWorkspaceStore {
        attach(workspace);transition(to:workspace.activeProjectID)
        if let current{return current}
        let created=DataFlowWorkspaceStore(workspace:workspace);current=created;return created
    }
    private func releaseCurrent() {
        current?.closeSourceInspection()
        if current?.isPreparing == true{current?.cancelPreparation()}
        current?.save()
        current=nil
    }
    func close() {
        releaseCurrent();observation=nil;workspace=nil;projectID=nil;showPreparation=false
    }
}
