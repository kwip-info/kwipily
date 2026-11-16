import Foundation
import KwipMLCore

struct PoolingExecutionWarning {
    let nodeID: String
    let message: String
    static func decode(_ data: Data, graph: GraphDocument) -> Self? {
        guard data.count <= 4096,
              let object = try? JSONDecoder().decode([String:JSONValue].self, from: data),
              Set(object.keys) == ["event", "schemaVersion", "runID", "nodeID", "code", "computeDevice", "returnDevice", "message"],
              object["event"]?.stringValue == "warning", object["schemaVersion"]?.integerValue == 1,
              object["code"]?.stringValue == "pooling.cpuFallback",
              object["computeDevice"]?.stringValue == "cpu", object["returnDevice"]?.stringValue == "mps",
              let runID = object["runID"]?.stringValue, UUID(uuidString: runID) != nil,
              let id = object["nodeID"]?.stringValue,
              let node = graph.nodes.first(where: { $0.id == id }),
              ["adaptive_avg_pool1d", "adaptive_avg_pool2d"].contains(node.type),
              let raw = object["message"]?.stringValue, !raw.isEmpty, raw.utf8.count <= 2048 else { return nil }
        // Use reviewed wording, rather than presenting arbitrary worker-provided text.
        return .init(nodeID: id, message: "Adaptive average · uneven groups use CPU compute; values and input influence return to MPS.")
    }
}
