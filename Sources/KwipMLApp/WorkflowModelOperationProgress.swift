import Foundation
import CoreFoundation
import KwipMLCore

/// One actual representative validation-capture forward. This presentation ledger
/// does not describe every training batch, device timing or uncaptured tensor values.
struct WorkflowModelOperationProgress {
    enum ContractError: Error { case invalidCapturedGraph }
    private let graph: GraphDocument
    private let origins: [String: CompositeNodeOrigin]
    private let workerRunID: String
    private var begun = false
    private var closed = false
    private(set) var sequence = 0
    private(set) var activeNodeID: String?
    private(set) var completedNodeIDs: Set<String> = []
    var activeOrigin: CompositeNodeOrigin? { activeNodeID.flatMap { origins[$0] } }
    var activeIncomingEdgeIDs: Set<String> {
        guard let activeNodeID else { return [] }
        return Set(graph.edges.filter { $0.target == activeNodeID }.map(\.id))
    }
    var isCapturing: Bool { begun && !closed }

    init(graph: GraphDocument, origins: [String: CompositeNodeOrigin], workerRunID: String) throws {
        let ids = Set(graph.nodes.map(\.id))
        let mapped = origins.values.map { $0.instancePath + [$0.localNodeID] }
        guard !workerRunID.isEmpty, (1...128).contains(graph.nodes.count), ids.count == graph.nodes.count,
              Set(origins.keys) == ids, Set(mapped).count == mapped.count,
              origins.values.allSatisfy({ origin in
                  origin.instancePath.count <= 8 && (origin.instancePath + [origin.localNodeID]).allSatisfy { !$0.isEmpty && $0.count <= 128 }
              }), graph.edges.count <= 256, Set(graph.edges.map(\.id)).count == graph.edges.count,
              graph.edges.allSatisfy({ ids.contains($0.source) && ids.contains($0.target) }) else {
            throw ContractError.invalidCapturedGraph
        }
        self.graph = graph; self.origins = origins; self.workerRunID = workerRunID
    }

    /// Rejected events cannot mutate sequence or highlights; caller controls refusal reporting.
    @discardableResult mutating func consume(_ event: [String: Any]) -> Bool {
        guard !closed, Self.integer(event["schemaVersion"]) == 1,
              event["runID"] as? String == workerRunID, event["stage"] as? String == "validationCapture",
              let next = Self.integer(event["sequence"]), next == sequence + 1, next <= graph.nodes.count * 2 + 2,
              let name = event["event"] as? String else { return false }
        let common: Set<String> = ["event", "schemaVersion", "runID", "stage", "sequence"]
        switch name {
        case "modelCaptureStarted":
            guard !begun, next == 1, Set(event.keys) == common.union(["operationCount"]),
                  Self.integer(event["operationCount"]) == graph.nodes.count else { return false }
            begun = true
        case "modelNodeStarted":
            guard begun, activeNodeID == nil, Set(event.keys) == common.union(["nodeID"]),
                  let node = event["nodeID"] as? String, origins[node] != nil, !completedNodeIDs.contains(node),
                  graph.edges.filter({ $0.target == node }).allSatisfy({ completedNodeIDs.contains($0.source) }) else { return false }
            activeNodeID = node
        case "modelNodeCompleted":
            guard begun, Set(event.keys) == common.union(["nodeID"]),
                  let node = event["nodeID"] as? String, activeNodeID == node else { return false }
            completedNodeIDs.insert(node); activeNodeID = nil
        case "modelCaptureCompleted":
            guard begun, activeNodeID == nil, Set(event.keys) == common,
                  completedNodeIDs.count == graph.nodes.count, next == graph.nodes.count * 2 + 2 else { return false }
            closed = true
        default: return false
        }
        sequence = next
        return true
    }

    /// Ancestor Blocks aggregate only a descendant in their exact captured instance.
    func activeNode(in instancePath: [String]) -> String? {
        guard let origin = activeOrigin, origin.instancePath.starts(with: instancePath) else { return nil }
        if origin.instancePath.count == instancePath.count { return origin.localNodeID }
        return origin.instancePath[instancePath.count]
    }
    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        return Int(number.stringValue)
    }
    mutating func clear() { activeNodeID = nil; closed = true }
}
