import Foundation
import KwipMLCore

/// Presentation evidence for actual operation boundaries. Incoming wires denote
/// consumed inputs, not measured transport; this does not claim layer timings.
struct WorkflowCanvasProgress:Equatable {
    enum Stage:String {case capturing,preparing,training,verifying}
    enum Disposition:String {case computed,reused,captured}
    let context:UUID
    private let allowedNodes:Set<WorkflowNodeAddress>
    private let edges:[WorkflowEdge]
    private(set) var sequence=0
    private(set) var stage:Stage?
    private(set) var activeNodeAddresses:Set<WorkflowNodeAddress>=[]
    private(set) var activeEdgeIDs:Set<String>=[]
    private(set) var completed:[WorkflowNodeAddress:Disposition]=[:]
    private var closed=false
    init(context:UUID,allowedNodes:Set<WorkflowNodeAddress>,edges:[WorkflowEdge]) {
        self.context=context;self.allowedNodes=allowedNodes;self.edges=edges
    }
    @discardableResult mutating func start(_ address:WorkflowNodeAddress,stage:Stage,context:UUID,sequence:Int)->Bool {
        guard accepts(address,context:context,sequence:sequence) else {return false}
        self.sequence=sequence;self.stage=stage;activeNodeAddresses=[address]
        activeEdgeIDs=stage == .capturing || stage == .verifying ? []:Set(edges.filter {$0.to.address == address}.map(\.id))
        return true
    }
    @discardableResult mutating func complete(_ address:WorkflowNodeAddress,disposition:Disposition,context:UUID,sequence:Int)->Bool {
        guard accepts(address,context:context,sequence:sequence),activeNodeAddresses.contains(address) else {return false}
        self.sequence=sequence;completed[address]=disposition;activeNodeAddresses=[];activeEdgeIDs=[];stage=nil
        return true
    }
    mutating func clear() {activeNodeAddresses=[];activeEdgeIDs=[];stage=nil;closed=true}
    private func accepts(_ address:WorkflowNodeAddress,context:UUID,sequence:Int)->Bool {
        !closed && context == self.context && sequence>self.sequence && allowedNodes.contains(address)
    }
}

struct WorkflowSourceCaptureProgress:Sendable {
    let address:WorkflowNodeAddress
    let sequence:Int
    let finished:Bool
}

/// Current-owner projection used by the canvas, separate from retained outputs.
enum WorkflowProgressPresentation {
    static func isActive(_ address:WorkflowNodeAddress,progress:WorkflowCanvasProgress?,isCurrent:Bool)->Bool {
        isCurrent && (progress?.activeNodeAddresses.contains(address) ?? false)
    }
}
extension WorkflowCanvasAdapter {
    var currentCanvasProgress:WorkflowCanvasProgress? {
        guard let host=outputProvider as? WorkflowWorkspaceSession,host.canvas === self,!host.isRetired,
              host.projectID == owner.projectID,host.sessionID == owner.sessionID,owner.isCurrent(owner.lease()) else {return nil}
        if let capture=host.currentSourceCaptureProgress {return capture}
        if let preparation=host.predictionPreparation,preparation.isRunning,preparation.document == owner.document {return preparation.canvasProgress}
        if let preparation=host.currentDataPreparation,
           preparation.document == owner.document {return preparation.canvasProgress}
        guard let execution=host.execution,execution.request.isCurrent(in:owner) else {return nil}
        return execution.canvasProgress
    }
    func isCanvasEdgeExecuting(_ displayID:String)->Bool {
        guard let id=projection.edgeIDByDisplayID[displayID] else {return false}
        if currentCanvasProgress?.activeEdgeIDs.contains(id) == true {return true}
        guard let host=outputProvider as? WorkflowWorkspaceSession,host.canvas === self,!host.isRetired,
              let execution=host.execution,execution.request.isCurrent(in:owner),
              let feed=execution.modelOperationProgress else {return false}
        let graph=execution.request.plan.model.graph,origins=execution.request.plan.model.origins
        guard let edge=owner.document.edges.first(where:{$0.id == id}) else {return false}
        return graph.edges.filter {feed.activeIncomingEdgeIDs.contains($0.id)}.contains {expanded in
            guard let source=origins[expanded.source],let target=origins[expanded.target] else {return false}
            return edge.from.address == WorkflowNodeAddress(domain:.model,localID:source.instancePath.first ?? source.localNodeID)
                && edge.to.address == WorkflowNodeAddress(domain:.model,localID:target.instancePath.first ?? target.localNodeID)
        }
    }
}
