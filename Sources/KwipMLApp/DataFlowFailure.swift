import Foundation
import KwipMLCore

struct DataFlowFailureDetail: Decodable, Equatable {
    let schemaVersion: Int
    let code: String
    let nodeID: String
    let nodeLabel: String?
    let operatorID: String
    let componentTitle: String
    let inputRowCount: Int?
    let cause: String
    let suggestion: String

    static func decode(_ data: Data, graph: DataFlowDocument, labels: [String: String], titles: [String: String]) throws -> Self {
        guard data.count <= 8192 else { throw DataFlowContractError("Failure detail exceeds its event bound.") }
        let object = try JSONDecoder().decode([String:JSONValue].self,from:data)
        guard Set(object.keys) == ["schemaVersion","code","nodeID","nodeLabel","operatorID","componentTitle","inputRowCount","cause","suggestion"] else {
            throw DataFlowContractError("Failure detail fields differ from the worker contract.")
        }
        let value = try JSONDecoder().decode(Self.self, from: data)
        func bounded(_ text: String, _ limit: Int) -> Bool { !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && text.unicodeScalars.count <= limit }
        guard value.schemaVersion == 1, bounded(value.code, 80), bounded(value.nodeID, 128), bounded(value.operatorID, 200), bounded(value.componentTitle, 200), bounded(value.cause, 500), bounded(value.suggestion, 1000),
              value.nodeLabel.map({ bounded($0, 200) }) ?? true,
              value.inputRowCount.map({ (0...5_000_000).contains($0) }) ?? true,
              let node = graph.nodes.first(where: { $0.id == value.nodeID }), node.operatorID == value.operatorID,
              labels[value.nodeID] == value.nodeLabel, titles[value.operatorID] == value.componentTitle else {
            throw DataFlowContractError("Failure detail does not match the captured preparation request.")
        }
        return value
    }
}

struct DataFlowFailure {
    let detail: DataFlowFailureDetail
    let graph: DataFlowDocument
    let labels: [String: String]
    let sourceBindings: [String: String]
    let sourceDigests: [String: String]
}

extension DataFlowWorkspaceStore {
    var requestNodeLabels: [String: String] {
        Dictionary(uniqueKeysWithValues: project.graph.nodes.compactMap { node in
            guard let name = project.editor[node.id]?.name, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            return (node.id, String(name.unicodeScalars.prefix(200)))
        })
    }
    var failureIsCurrent: Bool {
        guard let failure = preparationFailure, projectID == workspace.activeProjectID,
              failure.graph == project.graph, failure.labels == requestNodeLabels, failure.sourceBindings == project.sourceBindings else { return false }
        return failure.sourceDigests.allSatisfy { workspace.sourceDigest($0.key) == $0.value }
    }
    func selectFailureNode() {
        guard let failure = preparationFailure, project.graph.nodes.contains(where: { $0.id == failure.detail.nodeID && $0.operatorID == failure.detail.operatorID }) else { return }
        selectNode(failure.detail.nodeID)
    }
}
