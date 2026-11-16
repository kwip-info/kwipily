import Foundation
import KwipMLCore

/// Selection scope is independent of inspector visibility or its deferred reconciliation.
/// Invalid retained selection never becomes an unrequested global-results view.
enum WorkflowResultsScope: Equatable {
    case global
    case component(WorkflowNodeAddress)
    case connection(WorkflowEdge)
    case multiple
    case unavailable

    static func resolve(selection: WorkflowSelection, document: WorkflowDocument) -> Self {
        let nodeIDs=document.nodes.map(\.address),edgeIDs=document.edges.map(\.id)
        let nodes=Set(nodeIDs),edges=Set(edgeIDs)
        guard nodes.count == nodeIDs.count,edges.count == edgeIDs.count,
              selection.nodeAddresses.allSatisfy({nodes.contains($0) || (try? WorkflowInlineContainer.parse(in:document,address:$0)) != nil}),selection.edgeIDs.allSatisfy({edges.contains($0) || WorkflowInlineContainerProjection.internalEdge(in:document,displayID:$0) != nil}) else {return .unavailable}
        if selection.nodeAddresses.isEmpty && selection.edgeIDs.isEmpty {return .global}
        // Even a mixed selection containing a broken edge remains unavailable.
        let selectedEdges=(document.edges+document.inlineInternalEdges).filter {selection.edgeIDs.contains($0.id)}
        guard selectedEdges.allSatisfy({(nodes.contains($0.from.address) || (try? WorkflowInlineContainer.parse(in:document,address:$0.from.address)) != nil) && (nodes.contains($0.to.address) || (try? WorkflowInlineContainer.parse(in:document,address:$0.to.address)) != nil)}) else {return .unavailable}
        if selection.edgeIDs.isEmpty,selection.nodeAddresses.count == 1,let address=selection.nodeAddresses.first {return .component(address)}
        if selection.nodeAddresses.isEmpty,selectedEdges.count == 1 {return .connection(selectedEdges[0])}
        return .multiple
    }

    /// Captured dependency membership, not proof that an operation reached this node.
    static func capturedDependencies(document: WorkflowDocument, terminal: WorkflowNodeAddress) -> Set<WorkflowNodeAddress> {
        let addresses=document.nodes.map(\.address),nodes=Set(addresses)
        guard nodes.count == addresses.count,nodes.contains(terminal) else {return []}
        let incoming=Dictionary(grouping:document.edges,by:{$0.to.address})
        var visiting=Set<WorkflowNodeAddress>(),visited=Set<WorkflowNodeAddress>()
        func visit(_ address:WorkflowNodeAddress)->Bool {
            guard nodes.contains(address),!visiting.contains(address) else {return false}
            if visited.contains(address) {return true}
            visiting.insert(address)
            for edge in incoming[address] ?? [] {if !visit(edge.from.address) {return false}}
            visiting.remove(address);visited.insert(address)
            return true
        }
        return visit(terminal) ? visited:[]
    }
}

/// Current-session results only. History retains its separate immutable run ownership.
@MainActor enum WorkflowCurrentResult {
    case preparation(WorkflowDataPreparationSession)
    case training(WorkflowExecutionSession)
    case backend(BackendFitSession)

    var runID: UUID {
        switch self {
        case .preparation(let value): value.runID
        case .training(let value): value.runID
        case .backend(let value): value.runID
        }
    }
    var capturedDocument: WorkflowDocument {
        switch self {
        case .preparation(let value): value.document
        case .training(let value): value.request.document
        case .backend(let value): value.document
        }
    }
    func containsCapturedComponent(_ address: WorkflowNodeAddress) -> Bool {
        switch self {
        case .preparation(let value):
            return address.domain == .data && value.graph.nodes.contains { $0.id == address.localID }
        case .training(let value):
            if address.domain == .data {
                // Captured preparation also includes connected output-sink branches.
                return value.request.plan.dataflow.nodes.contains {$0.id == address.localID}
            }
            return WorkflowResultsScope.capturedDependencies(document:value.request.document,terminal:value.request.plan.train).contains(address)
        case .backend(let value):
            return WorkflowResultsScope.capturedDependencies(document:value.document,terminal:value.plan.estimator).contains(address)
        }
    }
}
