import Foundation

/// Representative shapes only. Private assessment fixtures vary widths and prefixes.
/// Shared parameters deliberately have no observation/batch axis.
public enum BilinearLearningCanvas {
    public struct Shape: Equatable {
        public enum LeadingAxis: Equatable { case observations, sharedParameter(Int) }
        public let leadingAxis: LeadingAxis
        public let remainingDimensions: [Int]
        public var description: String {
            let first: String
            switch leadingAxis { case .observations: first = "N"; case .sharedParameter(let count): first = String(count) }
            return "[" + ([first] + remainingDimensions.map(String.init)).joined(separator: ", ") + "]"
        }
        fileprivate func transformed(_ node: GraphNode) throws -> Shape {
            let config = try TensorTransformConfiguration(operation: node.type, values: node.config).values
            guard !remainingDimensions.isEmpty else { throw LearningContractError("A shared bias vector has no trailing axes to transform; its output-feature axis is protected.") }
            let layout = try TensorLayout(remainingDimensions)
            func axis(_ key: String) throws -> Int {
                let value = config[key]!.integerValue!, rank = remainingDimensions.count + 1
                let normalized = value < 0 ? value + rank : value
                guard normalized > 0, normalized < rank else { throw LearningContractError("Axis \(value) is outside the trailing dimensions or targets the protected leading axis.") }
                return normalized - 1
            }
            let result: TensorLayout
            switch node.type {
            case "reshape":
                guard case .array(let raw) = config["shape"] else { throw LearningContractError("Missing reshape dimensions.") }
                result = try layout.reshaped(raw.map { $0.integerValue! })
            case "transpose": result = try layout.transposed(axis("dim0"), axis("dim1"))
            default: result = try layout.flattened(from: axis("start_dim"), through: axis("end_dim"))
            }
            return Shape(leadingAxis: leadingAxis, remainingDimensions: result.dimensions)
        }
    }
    public struct Projection: Equatable {
        public var validation = ValidationResult()
        public var shapes: [String: Shape] = [:]
        public func shapeDescription(_ nodeID: String) -> String? { shapes[nodeID]?.description }
    }
    public static func registry(task: BilinearLearningTask) -> ComponentRegistry {
        var registry = TensorLearningCanvas.registry
        registry.components.insert(.init(id: "bilinear", title: "Bilinear", category: "Layers",
            inputs: task == .withBias ? ["a", "b", "weight", "bias"] : ["a", "b", "weight"], outputs: ["out"],
            fields: [.init(key: "bias", title: "Learned bias", kind: .boolean, defaultValue: .bool(task == .withBias))]), at: registry.components.count - 1)
        return registry
    }
    public static func inspect(_ graph: GraphDocument, task: BilinearLearningTask) -> Projection {
        var projection = Projection()
        func issue(_ message: String, _ nodeID: String? = nil) {
            projection.validation.issues.append(.init(id: "bilinear.lesson.\(projection.validation.issues.count)", message: message, nodeID: nodeID))
        }
        do { _ = try BilinearLearningSubmission.validatedGraph(graph, complete: false) }
        catch { issue(error.localizedDescription); return projection }
        do { _ = try BilinearLearningSubmission.validatedGraph(graph, complete: true) }
        catch { issue(error.localizedDescription) }
        for node in graph.nodes where node.type == "bilinear" {
            if node.config["bias"] != .bool(task == .withBias) { issue("The Bilinear bias mode must match this lesson.", node.id) }
        }
        let nodes = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        let incoming = Dictionary(grouping: graph.edges, by: \.target)
        let outgoing = Dictionary(grouping: graph.edges, by: \.source)
        var pending = nodes.mapValues { incoming[$0.id]?.count ?? 0 }
        var queue = graph.nodes.filter { pending[$0.id] == 0 }.map(\.id), cursor = 0
        while cursor < queue.count {
            let id = queue[cursor]; cursor += 1
            guard let node = nodes[id] else { continue }
            projection.validation.orderedNodeIDs.append(id)
            func operand(_ port: String) -> Shape? {
                incoming[id]?.first { $0.targetPort == port }.flatMap { projection.shapes[$0.source] }
            }
            do {
                var shape: Shape?
                if node.type == "input" {
                    switch node.config["fixture"]?.stringValue {
                    case "left": shape = .init(leadingAxis: .observations, remainingDimensions: [2])
                    case "right": shape = .init(leadingAxis: .observations, remainingDimensions: [3])
                    case "weight": shape = .init(leadingAxis: .sharedParameter(2), remainingDimensions: [2, 3])
                    case "bias": shape = .init(leadingAxis: .sharedParameter(2), remainingDimensions: [])
                    default: break
                    }
                } else if node.type == "bilinear" {
                    if let a = operand("a"), let b = operand("b"), let weight = operand("weight"),
                       node.config["bias"] == .bool(false) || operand("bias") != nil {
                        guard a.leadingAxis == .observations, b.leadingAxis == .observations,
                              !a.remainingDimensions.isEmpty, !b.remainingDimensions.isEmpty,
                              a.remainingDimensions.dropLast() == b.remainingDimensions.dropLast(),
                              case .sharedParameter(let outputs) = weight.leadingAxis,
                              weight.remainingDimensions == [a.remainingDimensions.last!, b.remainingDimensions.last!] else {
                            throw LearningContractError("Bilinear needs matching observation prefixes and shared weight [outputs, first features, second features]; no axes broadcast.")
                        }
                        if node.config["bias"] == .bool(true) {
                            guard operand("bias") == Shape(leadingAxis: .sharedParameter(outputs), remainingDimensions: []) else {
                                throw LearningContractError("Bilinear bias must be a shared vector with one entry per output feature.")
                            }
                        }
                        let dimensions = Array(a.remainingDimensions.dropLast()) + [outputs]
                        _ = try TensorLayout(dimensions)
                        shape = .init(leadingAxis: .observations, remainingDimensions: dimensions)
                    }
                } else if let input = operand("in") {
                    shape = node.type == "output" ? input : try input.transformed(node)
                }
                if let shape { projection.shapes[id] = shape }
            } catch { issue(error.localizedDescription, id) }
            for edge in outgoing[id] ?? [] {
                pending[edge.target, default: 0] -= 1
                if pending[edge.target] == 0 { queue.append(edge.target) }
            }
        }
        return projection
    }
}
