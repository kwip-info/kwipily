import Foundation
import KwipMLCore

/// View lifetime follows the field contract; mutation authorization follows the owner lease.
struct ModelInspectorFieldIdentity {
    struct ViewID: Hashable {
        let contextKey: String
        let projectID: String
        let sessionID: UUID
        let path: [String]
        let nodeID: String
        let nodeType: String
        let fieldKey: String
        let kind: String
        let minimum: Double?
        let maximum: Double?
        let minimumItems: Int?
        let maximumItems: Int?
        let allowsNull: Bool
    }
    let viewID: ViewID
    let origin: String

    init(contextKey: String, node: GraphNode, field: ComponentField, lease: ModelEditorLease) {
        viewID = ViewID(contextKey: contextKey, projectID: lease.projectID, sessionID: lease.sessionID,
                        path: lease.instancePath, nodeID: node.id, nodeType: node.type, fieldKey: field.key,
                        kind: field.kind.rawValue, minimum: field.min, maximum: field.max, minimumItems: field.minItems, maximumItems: field.maxItems, allowsNull: field.allowsNull)
        // Length prefixes avoid collisions when user-defined IDs themselves contain separators.
        let components = [contextKey, lease.projectID, lease.sessionID.uuidString, String(lease.editRevision),
                          String(lease.instancePath.count)] + lease.instancePath + [node.id, node.type, field.key]
        origin = components.map { String($0.utf8.count) + ":" + $0 }.joined()
    }
}
