import Foundation
import CryptoKit

/// Reversible-by-current-source display edge identity. It is never a saved workflow edge.
public enum WorkflowInlineContainerProjection {
    public static func internalEdgeID(root: WorkflowNodeAddress, edgeID: String) -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        // Length-qualified fields prevent ambiguous Unicode/separator combinations.
        let value = [root.domain.rawValue, root.localID, edgeID]
        let bytes = (try? encoder.encode(value)) ?? Data()
        return "inline-edge-" + SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
    /// Accept a projected GraphEdge ID or its source lookup token, and resolve only current immediate primitive wires.
    public static func internalEdge(in document: WorkflowDocument, displayID: String) -> (root: WorkflowNodeAddress, edgeID: String)? {
        let token = (try? WorkflowCanvasIDs.edgeID(from: displayID)) ?? displayID
        for root in document.nodes {
            guard let definition = try? WorkflowInlineContainer.definition(in: document, root: root.address) else { continue }
            let primitives = Set(definition.body.nodes.map(\.id))
            for edge in definition.body.edges where primitives.contains(edge.source) && primitives.contains(edge.target) {
                if internalEdgeID(root: root.address, edgeID: edge.id) == token { return (root.address, edge.id) }
            }
        }
        return nil
    }
}

public extension WorkflowCanvasProjection {
    /// Adds immediate members of expanded graph-composition Blocks to a root projection.
    /// Positions are model coordinates supplied by the canvas layout. Root source, root
    /// visual nodes/wires and their typed port indexes are preserved verbatim. No boundary
    /// links, workers, inference, schema edits or source mutations are introduced here.
    func addingInlineMembers(_ document: WorkflowDocument,
                             expandedRoots: Set<WorkflowNodeAddress>,
                             positions: [WorkflowNodeAddress: CGPoint] = [:],
                             modelRegistry: ComponentRegistry,
                             dataRegistry: DataFlowOperatorRegistry) throws -> WorkflowCanvasProjection {
        guard !expandedRoots.isEmpty else { return self }
        guard try updatingPresentation(document) != nil else {
            throw DataFlowContractError("Inline members require the matching root-only source projection.")
        }
        var memberNodes: [WorkflowNode] = [], memberEdges: [WorkflowEdge] = []
        var additionalIssues: [WorkflowCanvasIssue] = []
        let actualAddresses = Set(document.nodes.map(\.address))
        let actualEdgeIDs = Set(document.edges.map(\.id))
        for root in document.nodes where expandedRoots.contains(root.address) {
            guard let definition = try? WorkflowInlineContainer.definition(in: document, root: root.address) else { continue }
            let primitiveIDs = Set(definition.body.nodes.map(\.id))
            for primitive in definition.body.nodes {
                var member = try WorkflowInlineContainer.node(in: document, root: root.address, childID: primitive.id, registry: modelRegistry)
                if WorkflowMixtureContainer.contains(document,root:root.address){member.name=primitive.name}
                guard !actualAddresses.contains(member.address) else {
                    throw DataFlowContractError("An inline member display identity conflicts with a saved workflow node.")
                }
                if let point = positions[member.address] {
                    guard point.x.isFinite, point.y.isFinite else { throw DataFlowContractError("Inline member positions must be finite model coordinates.") }
                    member.x = Double(point.x); member.y = Double(point.y)
                }
                memberNodes.append(member)
            }
            for edge in definition.body.edges where primitiveIDs.contains(edge.source) && primitiveIDs.contains(edge.target) {
                let id = WorkflowInlineContainerProjection.internalEdgeID(root: root.address, edgeID: edge.id)
                guard !actualEdgeIDs.contains(id) else {
                    throw DataFlowContractError("An inline wire display identity conflicts with a saved workflow edge.")
                }
                memberEdges.append(.init(id: id,
                    from: .init(address: WorkflowInlineContainer.address(root: root.address, childID: edge.source), port: edge.sourcePort),
                    to: .init(address: WorkflowInlineContainer.address(root: root.address, childID: edge.target), port: edge.targetPort), kind: .tensor))
            }
            if !definition.body.instances.isEmpty {
                additionalIssues.append(.init(address: root.address, port: nil, message: "Nested Blocks are retained in the source; this inline view shows immediate primitive members only."))
            }
        }
        guard !memberNodes.isEmpty else { return self }
        // A temporary display-only source lets the existing projector resolve the same
        // configured component metadata and port rules. It is never returned as authoring source.
        var synthetic = WorkflowDocument(projectID: document.projectID, revision: document.revision,
            name: document.name, nodes: memberNodes, edges: memberEdges, definitions: document.definitions)
        synthetic.schemaVersion = document.schemaVersion
        let members = try WorkflowCanvasProjector.project(synthetic, modelRegistry: modelRegistry, dataRegistry: dataRegistry)
        let existingNodeIDs = Set(graph.nodes.map(\.id)), existingEdgeIDs = Set(graph.edges.map(\.id))
        guard existingNodeIDs.isDisjoint(with: members.graph.nodes.map(\.id)),
              existingEdgeIDs.isDisjoint(with: members.graph.edges.map(\.id)) else {
            throw DataFlowContractError("Inline members must augment a root-only canvas projection.")
        }
        var mergedGraph = graph
        mergedGraph.nodes += members.graph.nodes; mergedGraph.edges += members.graph.edges
        var components = registry.components
        let componentIDs = Set(components.map(\.id))
        components += members.registry.components.filter { !componentIDs.contains($0.id) }
        return .init(graph: mergedGraph, registry: .init(schemaVersion: registry.schemaVersion, components: components),
            addressByDisplayID: addressByDisplayID.merging(members.addressByDisplayID) { first, _ in first },
            displayIDByAddress: displayIDByAddress.merging(members.displayIDByAddress) { first, _ in first },
            edgeIDByDisplayID: edgeIDByDisplayID.merging(members.edgeIDByDisplayID) { first, _ in first },
            displayIDByEdgeID: displayIDByEdgeID.merging(members.displayIDByEdgeID) { first, _ in first },
            typeByDisplayID: typeByDisplayID.merging(members.typeByDisplayID) { first, _ in first },
            ports: ports + members.ports, issues: issues + members.issues + additionalIssues, sourceDocument: document)
    }
}
