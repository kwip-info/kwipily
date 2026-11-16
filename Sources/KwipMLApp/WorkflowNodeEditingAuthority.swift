import Foundation
import Combine
import KwipMLCore

/// One retained selected-node editing session. Own commits renew its lease; any
/// external replacement or selection departure permanently retires its callbacks.
@MainActor final class WorkflowNodeEditingAuthority:ObservableObject {
    let owner:WorkflowAuthoringStore
    let address:WorkflowNodeAddress
    private let contextID=UUID()
    var contextIdentity:String {contextID.uuidString}
    private(set) var token:WorkflowEditorLease
    private var active=true,committing=false
    private var observations:[AnyCancellable]=[]
    @Published var message:String?
    init(owner:WorkflowAuthoringStore,address:WorkflowNodeAddress)throws {
        self.owner=owner;self.address=address;token=owner.lease()
        guard owner.isCurrent(token),owner.selection.nodeAddresses == [address],owner.selection.edgeIDs.isEmpty,
              owner.authoringNode(at:address) != nil else {throw DataFlowContractError("Select a node in the active workflow.")}
        observations.append(owner.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()})
        observations.append(owner.$selection.dropFirst().sink {[weak self] next in
            guard let self,next.nodeAddresses != [self.address] || !next.edgeIDs.isEmpty else {return}
            self.retire()
        })
        observations.append(owner.$document.dropFirst().sink {[weak self] _ in
            guard let self,!self.committing else {return}
            if self.active && self.owner.isPublishingLayoutMove {
                self.token = self.owner.lease()
                return
            }
            self.retire()
        })
    }
    var isCurrent:Bool {active && owner.isCurrent(token) && owner.selection.nodeAddresses == [address] && owner.selection.edgeIDs.isEmpty}
    var isBusy:Bool {!isCurrent || owner.isBusy || !owner.document.drafts.isEmpty}
    var node:WorkflowNode? {isCurrent ? owner.authoringNode(at:address) : nil}
    var configuration:[String:JSONValue] {
        guard let node else {return [:]}
        switch node.payload {
        case .dataOperator(_,let config),.modelPrimitive(_,_,let config),.optimizerRecipe(_,_,let config),.schedulerRecipe(_,_,let config),.lossRecipe(_,_,let config),.initializationRecipe(_,_,let config),.gradientPolicyRecipe(_,_,let config),.trainabilityRecipe(_,_,let config),.datasetRecipe(_,_,let config),.batchRecipe(_,_,let config),.trainingData(let config),.runSetup(let config),.estimator(let config),.train(let config),.frozenModel(let config),.evaluateTest(let config),.saveModel(let config),.generateText(let config),.gameSession(let config):return config
        case .modelBlock(_,let overrides):return overrides
        case .modelDefinition:return [:]
        }
    }
    func accepts(_ expected:[String:JSONValue])->Bool {
        guard !isBusy else {return false}
        guard configuration == expected else {message="Configuration changed. Reload this editor before applying its draft.";return false}
        return true
    }
    /// Shared lifetime wrapper for canonical owner transactions, including explicit
    /// schema migration. Failed persistence leaves this context and its tokens alive.
    @discardableResult func applyTransaction(_ transaction:(WorkflowEditorLease)->Bool)->Bool {
        guard !isBusy else {return false}
        committing=true;defer {committing=false}
        guard transaction(token) else {message=owner.error ?? "Change was not applied.";return false}
        token=owner.lease();message="Saved with the workflow.";return true
    }
    @discardableResult func apply(_ command:WorkflowCommand)->Bool {
        applyTransaction {owner.apply(command,lease:$0) != nil}
    }
    @discardableResult func configure(_ config:[String:JSONValue],expected:[String:JSONValue])->Bool {
        accepts(expected) && apply(.configure(address:address,config:config))
    }
    @discardableResult func rename(_ name:String)->Bool {apply(.rename(address:address,name:String(name.prefix(80))))}
    func numericOrigin(key:String)->String {contextIdentity+":"+key}
    func currentNumericOrigin(key:String)->String? {!isBusy ? numericOrigin(key:key) : nil}
    func setNumericPending(origin:String,key:String,pending:Bool) {
        guard isCurrent,origin == numericOrigin(key:key) else {return}
        _=owner.setInspectorEditPending(contextID:contextID,address:address,key:key,pending:pending,lease:token)
    }
    func retire() {
        guard active else {return};active=false
        owner.clearInspectorEdits(contextID:contextID);observations.removeAll();objectWillChange.send()
    }
}
