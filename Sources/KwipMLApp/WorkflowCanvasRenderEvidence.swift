import Foundation
import KwipMLCore

/// One immutable render's cached evidence. Camera packets never ask every wire
/// to compare the entire captured document again or convert captured values.
struct WorkflowCanvasRenderEvidence {
    var statuses:[String:String]=[:]
    var feedback:[String:CanvasNodeFeedback]=[:]
    var badges:[String:[String:String]]=[:]
    var executingEdges=Set<String>()

    static func badge(port:String,kind:CanvasPortKind,snapshot:WorkflowCapturedDataOutput?)->String {
        guard let snapshot,case .object(let value)=snapshot.ports[port] else {return port+" · not captured"}
        let evidence = !snapshot.matchesCurrentConstruction || !snapshot.matchesCurrentBindings ? "stale capture":"captured"
        if case .object(let preview)=value["preview"],case .integer(let count)=preview["rowCount"] {return port+" · "+evidence+" · \(count) rows"}
        return port+" · "+evidence+" "+kind.rawValue
    }

    @MainActor static func make(adapter:WorkflowCanvasAdapter,active:Bool,validation:ValidationResult)->Self {
        var result=Self()
        let owner=adapter.owner,graph=adapter.graph
        let provider=adapter.outputProvider
        let current=active && owner.isCurrent(owner.lease()) && provider?.projectID == owner.projectID && provider?.sessionID == owner.sessionID
        let addresses=Set(adapter.projection.addressByDisplayID.values.filter {$0.domain != .control})
        let captures=current ? provider?.snapshots(for:addresses) ?? [:]:[:]
        let host=current ? provider as? WorkflowWorkspaceSession:nil
        let ownedHost=host?.canvas === adapter && host?.isRetired == false
        let failure=ownedHost ? host?.attributedPreparationFailure:nil
        let execution=ownedHost ? host?.execution:nil
        let executionCurrent=execution.map {$0.request.isCurrent(in:owner)} ?? false
        let preparation=ownedHost ? host?.predictionPreparation:nil
        let preparationCurrent=preparation?.isRunning == true && preparation?.document == owner.document
        let fit=ownedHost ? host?.backendFit:nil
        let fitCurrent=fit.map {$0.document == owner.document && $0.bindingRevision == host?.bindings.revision} ?? false
        let progress=ownedHost ? adapter.currentCanvasProgress:nil
        for node in graph.nodes {
            let address=adapter.projection.addressByDisplayID[node.id]
            let capture=address.flatMap {captures[$0]}.flatMap {value in
                value.address == address && value.projectID == owner.projectID && value.sessionID == owner.sessionID ? value:nil
            }
            let status:String
            if current,let address {
                if failure?.currentCause(for:address) != nil {status="Failed"}
                else if address.domain == .control {status=provider?.terminalStatus(for:address) ?? "Not evaluated"}
                else if let capture {status = !capture.matchesCurrentConstruction || !capture.matchesCurrentBindings ? "Stale":capture.phase == .completed ? "Captured":"Partial capture"}
                else {status="Not evaluated"}
            } else {status="Not evaluated"}
            result.statuses[node.id]=status
            var executing=false
            if ownedHost,let host,let address {
                executing=host.modelLoader.isBusy && host.modelLoader.activeAddress==address
                if let execution {
                    executing=WorkflowProgressPresentation.isActive(address,progress:execution.canvasProgress,isCurrent:executionCurrent)
                    if executionCurrent,address.domain == .model {executing = executing || execution.modelOperationProgress?.activeNode(in:[]) == address.localID}
                }
                if let capture=host.currentSourceCaptureProgress {executing = executing || WorkflowProgressPresentation.isActive(address,progress:capture,isCurrent:true)}
                if preparationCurrent,let preparation {executing = executing || WorkflowProgressPresentation.isActive(address,progress:preparation.canvasProgress,isCurrent:true)}
                if let fit {executing = executing || CanvasNodeFeedback.isExecuting(phase:fit.phase,matchesTerminal:fit.plan.estimator==address,isCurrent:fitCurrent)}
            }
            let nodeFailure=ownedHost ? address.flatMap {host?.modelLoader.failure($0) ?? failure?.currentCause(for:$0)}:nil
            result.feedback[node.id] = .init(nodeID:node.id,issues:validation.issues+adapter.connectionFeedbackIssues(for:node.id),isExecuting:executing,executionFailure:nodeFailure)
            if let definition=adapter.registry.component(type:node.type) {
                result.badges[node.id]=Dictionary(uniqueKeysWithValues:definition.outputs.map {port in
                    (port,Self.badge(port:port,kind:adapter.canvasPortDescriptor(nodeID:node.id,port:port,direction:.output).kind,snapshot:capture))
                })
            }
        }
        result.executingEdges=Set(adapter.projection.edgeIDByDisplayID.compactMap {display,id in progress?.activeEdgeIDs.contains(id) == true ? display:nil})
        if executionCurrent,let execution,let feed=execution.modelOperationProgress {
            let model=execution.request.plan.model
            // Expanded edges are mapped once, not scanned for every displayed wire.
            let pairs=Set(model.graph.edges.filter {feed.activeIncomingEdgeIDs.contains($0.id)}.compactMap {edge -> String? in
                guard let from=model.origins[edge.source],let to=model.origins[edge.target] else {return nil}
                return (from.instancePath.first ?? from.localNodeID)+"\0"+(to.instancePath.first ?? to.localNodeID)
            })
            for edge in owner.document.edges where edge.from.address.domain == .model && edge.to.address.domain == .model {
                if pairs.contains(edge.from.address.localID+"\0"+edge.to.address.localID),let display=adapter.projection.displayIDByEdgeID[edge.id] {result.executingEdges.insert(display)}
            }
        }
        return result
    }
}
