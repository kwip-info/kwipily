import Foundation
import KwipMLCore

/// Resolves derivative geometry only from the captured graph and verified ownership.
enum WorkflowDerivativeGeometry {
    private static func fail() -> WorkflowPhaseError {
        WorkflowPhaseError("Derivative geometry differs from the captured model ports or verified native ownership.")
    }
    private static func shape(_ value: JSONValue?) throws -> [Int] {
        guard case .array(let raw) = value, !raw.isEmpty, raw.count <= 8 else { throw fail() }
        return try raw.map {
            guard case .integer(let n) = $0, n > 0, n <= 200_000 else { throw fail() }
            return Int(n)
        }
    }
    private static func elements(_ shape: [Int], maximum: Int, allowEmpty: Bool = false) throws -> Int {
        if allowEmpty && shape == [0] { return 0 }
        var count = 1
        for dimension in shape {
            guard dimension > 0, dimension <= maximum, count <= maximum / dimension else { throw fail() }
            count *= dimension
        }
        return count
    }
    static func resolve(policy: GradientPolicyConfiguration, graph: GraphDocument,
                        registry: ComponentRegistry, context: GraphValidationContext,
                        identity: WorkflowGradientPolicyIdentity, bag: WorkflowBagDerivativeContext? = nil) throws -> [DerivativeRequestGeometry] {
        guard identity.gradientPolicy == policy, Set(graph.nodes.map(\.id)).count == graph.nodes.count else { throw fail() }
        let nodes = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        let validation = GraphValidator.validate(graph, registry: registry, context: context)
        guard validation.isValid else { throw fail() }
        if let bag {
            guard try WorkflowMappingRequirements.bagInputRequirement(graph, registry: registry, context: context) == bag.requirement else { throw fail() }
        }
        struct Owner { let name: String; let shape: [Int]; let enabled: Bool }
        var owners: [String: Owner] = [:], nativeNames = Set<String>(), nativeElements = 0
        for raw in identity.fullOwnership {
            guard case .object(let row) = raw,
                  Set(row.keys) == ["node", "parameter", "name", "shape", "dtype", "requiresGrad"],
                  let node = row["node"]?.stringValue, let parameter = row["parameter"]?.stringValue,
                  let name = row["name"]?.stringValue, nodes[node] != nil,
                  !node.isEmpty, !parameter.isEmpty, !name.isEmpty,
                  !node.contains("\0"), !parameter.contains("\0"), !name.contains("\0"),
                  row["dtype"] == .string("float32"), case .bool(let enabled) = row["requiresGrad"] else { throw fail() }
            let dimensions = try shape(row["shape"]), count = try elements(dimensions, maximum: 200_000)
            let key = node + "\0" + parameter
            guard owners[key] == nil, nativeNames.insert(name).inserted, nativeElements <= 200_000 - count else { throw fail() }
            nativeElements += count; owners[key] = .init(name: name, shape: dimensions, enabled: enabled)
        }
        func tensor(_ node: String, _ port: String) throws -> TypedTensorContract {
            guard nodes[node] != nil,
                  let definition = validation.configuredDefinition(nodeID: node, graph: graph, registry: registry),
                  definition.outputs.contains(port),
                  let contract = validation.outputTensorContracts[.init(node, port)] else { throw fail() }
            return contract
        }
        var results: [DerivativeRequestGeometry] = [], appearances = 0
        for raw in policy.requests {
            guard case .object(let request) = raw, let id = request["id"]?.stringValue,
                  case .integer(let rows) = request["row_count"], rows >= 1, rows <= 32,
                  case .object(let endpoint) = request["output"],
                  let outputNode = endpoint["node"]?.stringValue, let outputPort = endpoint["port"]?.stringValue,
                  case .array(let selected) = request["targets"] else { throw fail() }
            let output = try tensor(outputNode, outputPort)
            let outputShape = [Int(rows)] + output.layout.sample.dimensions
            guard output.dtype == .float32, outputShape.count <= 8 else { throw fail() }
            _ = try elements(outputShape, maximum: 100_000_000)
            if case .object(let upstream) = request["upstream"] {
                guard try shape(upstream["shape"]) == outputShape else { throw fail() }
            }
            let bagCounts = try bag?.selectedCounts(raw)
            var targets: [DerivativeTargetGeometry] = [], names = Set<String>()
            for rawTarget in selected {
                guard case .object(let target) = rawTarget,
                      let kind = target["kind"]?.stringValue, let node = target["node"]?.stringValue else { throw fail() }
                let name: String, dimensions: [Int], dtype: ContractDType, enabled: Bool
                if kind == "input" {
                    guard let source = nodes[node], let port = target["port"]?.stringValue else { throw fail() }
                    name = node + "." + port
                    if source.type == "bag_input" {
                        guard let bag, let bagCounts,
                              let contract = validation.outputBagContracts[.init(node, port)],
                              contract.sourceNodeID == bag.requirement.sourceNodeID,
                              contract.configuration == bag.requirement.configuration,
                              contract.role.rawValue == port else { throw fail() }
                        dimensions = try bag.coordinateShape(node: node, port: port, counts: bagCounts)
                        dtype = contract.dtype; enabled = dtype == .float32
                    } else {
                        guard bag == nil, ["input", "index_input"].contains(source.type), port == "out" else { throw fail() }
                        let input = try tensor(node, port)
                        guard [.float32, .int64].contains(input.dtype) else { throw fail() }
                        dimensions = [Int(rows)] + input.layout.sample.dimensions
                        dtype = input.dtype; enabled = dtype == .float32
                    }
                    guard dimensions.count <= 8, !nativeNames.contains(name) else { throw fail() }
                } else if kind == "parameter" {
                    guard let parameter = target["parameter"]?.stringValue,
                          let owner = owners[node + "\0" + parameter] else { throw fail() }
                    name = owner.name; dimensions = owner.shape; dtype = .float32; enabled = owner.enabled
                } else { throw fail() }
                let count = try elements(dimensions, maximum: 65_536, allowEmpty: kind == "input" && bag != nil && dimensions == [0])
                guard names.insert(name).inserted, appearances <= 65_536 - count else { throw fail() }
                appearances += count
                targets.append(.init(target: rawTarget, name: name, shape: dimensions, dtype: dtype, requiresGrad: enabled))
            }
            results.append(.init(requestID: id, outputShape: outputShape, targets: targets))
        }
        return results
    }
}
