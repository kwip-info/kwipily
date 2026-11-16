import Foundation
import KwipMLCore

/// Presentation metadata only. No capture, source reads or implicit computation.
enum WorkflowSelectionPresentation {
    static func outputDescription(_ snapshot: WorkflowCapturedDataOutput?) -> String {
        guard let snapshot, !snapshot.ports.isEmpty else { return "Not captured" }
        return snapshot.ports.keys.sorted().map { name in
            if case .object(let value) = snapshot.ports[name], case .object(let preview) = value["preview"],
               case .integer(let count) = preview["rowCount"], count >= 0 {
                return "\(name) · \(count) records"
            }
            return name
        }.joined(separator: ", ")
    }
    static func captureState(_ snapshot: WorkflowCapturedDataOutput?) -> String {
        guard let snapshot, !snapshot.ports.isEmpty else { return "Not captured" }
        if !snapshot.matchesCurrentConstruction || !snapshot.matchesCurrentBindings { return "Stale" }
        return snapshot.phase == .completed ? "Current" : "Partial capture"
    }
    static func owned(_ snapshot: WorkflowCapturedDataOutput?, projectID: String, sessionID: UUID,
                      address: WorkflowNodeAddress) -> WorkflowCapturedDataOutput? {
        guard let snapshot, snapshot.projectID == projectID, snapshot.sessionID == sessionID,
              snapshot.address == address else { return nil }
        return snapshot
    }

    static func connections(in document: WorkflowDocument, selection: WorkflowSelection) -> [WorkflowEdge] {
        document.edges.filter { selection.edgeIDs.contains($0.id) || selection.nodeAddresses.contains($0.from.address) || selection.nodeAddresses.contains($0.to.address) }
    }
    static func relation(_ edge: WorkflowEdge, selection: WorkflowSelection) -> String {
        let from = selection.nodeAddresses.contains(edge.from.address), to = selection.nodeAddresses.contains(edge.to.address)
        if from && to { return "Internal" }
        if from { return "Outgoing" }
        if to { return "Incoming" }
        return "Selected"
    }
    static func configurationIssues(_ issues: [ValidationIssue], displayIDs: Set<String>) -> [ValidationIssue] {
        issues.filter { $0.nodeID.map(displayIDs.contains) == true }
    }
    static func diagnostics(_ entries: [WorkflowRunConsole.Entry], selection: WorkflowSelection) -> [WorkflowRunConsole.Entry] {
        entries.filter { $0.level != .info && $0.component.map(selection.nodeAddresses.contains) == true }
    }
}

struct WorkflowSelectionBookmark {
    let projectID: String
    let sessionID: UUID
    let selection: WorkflowSelection
    func canRestore(projectID: String, sessionID: UUID, document: WorkflowDocument) -> Bool {
        self.projectID == projectID && self.sessionID == sessionID && document.projectID == projectID &&
        selection.nodeAddresses.isSubset(of: Set(document.nodes.map(\.address))) &&
        selection.edgeIDs.isSubset(of: Set(document.edges.map(\.id))) &&
        WorkflowResultsScope.resolve(selection: selection, document: document) == .multiple
    }
}
