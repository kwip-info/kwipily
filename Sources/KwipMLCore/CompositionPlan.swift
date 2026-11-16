import Foundation
import CryptoKit

/// Immutable, source-derived native registration tree; flat graph edges remain authoritative.
public struct CompositionPlan: Codable, Equatable {
    public struct Endpoint: Codable, Equatable, Hashable {
        public let node: String
        public let port: String
        public init(node: String, port: String) { self.node = node; self.port = port }
    }
    public struct Input: Codable, Equatable {
        public let id: String
        public let targets: [Endpoint]
        public init(id: String, targets: [Endpoint]) { self.id = id; self.targets = targets }
    }
    public struct Output: Codable, Equatable {
        public let id: String
        public let source: Endpoint
        public init(id: String, source: Endpoint) { self.id = id; self.source = source }
    }
    public struct Child: Codable, Equatable {
        public enum Kind: String, Codable { case node, container }
        public let kind: Kind
        public let id: String
        public let key: String
        public init(kind: Kind, id: String, key: String) { self.kind = kind; self.id = id; self.key = key }
    }
    public struct Container: Codable, Equatable {
        public let id: String
        public let kind: CompositionPolicy.Kind
        public let path: [String]
        public let children: [Child]
        public let inputs: [Input]
        public let outputs: [Output]
        public init(id: String, kind: CompositionPolicy.Kind, path: [String], children: [Child], inputs: [Input], outputs: [Output]) {
            self.id = id; self.kind = kind; self.path = path; self.children = children; self.inputs = inputs; self.outputs = outputs
        }
    }
    public let schemaVersion: Int
    public let root: String
    public let containers: [Container]
    public init(schemaVersion: Int = 1, root: String = "root", containers: [Container]) {
        self.schemaVersion = schemaVersion; self.root = root; self.containers = containers
    }
    public static func containerID(path: [String]) -> String {
        if path.isEmpty { return "root" }
        let bytes = path.reduce(into: Data()) { data, part in
            data.append(Data("\(part.utf8.count):".utf8)); data.append(Data(part.utf8))
        }
        return "c-" + SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw CompositeContractError(.invalidExpandedGraph, message) }
    }
    private static func object(_ value: JSONValue, _ keys: Set<String>) throws -> [String: JSONValue] {
        guard case .object(let fields) = value, Set(fields.keys) == keys else { throw CompositeContractError(.invalidExpandedGraph, "Composition plan has unknown or missing fields.") }
        return fields
    }
    private static func text(_ value: JSONValue?) throws -> String {
        guard case .string(let value) = value, !value.isEmpty, value.unicodeScalars.count <= 128 else { throw CompositeContractError(.invalidExpandedGraph, "Composition plan requires bounded nonempty identities.") }
        return value
    }
    private static func array(_ value: JSONValue?) throws -> [JSONValue] {
        guard case .array(let values) = value else { throw CompositeContractError(.invalidExpandedGraph, "Composition plan requires complete arrays.") }
        return values
    }
    public init(from decoder: Decoder) throws {
        let fields = try Self.object(JSONValue(from: decoder), ["schemaVersion", "root", "containers"])
        guard case .integer(1) = fields["schemaVersion"] else { throw CompositeContractError(.unsupportedSchema, "Composition plan requires version 1.") }
        schemaVersion = 1; root = try Self.text(fields["root"])
        func endpoint(_ value: JSONValue) throws -> Endpoint {
            let f = try Self.object(value, ["node", "port"])
            return try Endpoint(node: Self.text(f["node"]), port: Self.text(f["port"]))
        }
        containers = try Self.array(fields["containers"]).map { value in
            let f = try Self.object(value, ["id", "kind", "path", "children", "inputs", "outputs"])
            guard let kind = CompositionPolicy.Kind(rawValue: try Self.text(f["kind"])) else { throw CompositeContractError(.invalidExpandedGraph, "Unknown native container kind.") }
            let children = try Self.array(f["children"]).map { value -> Child in
                let c = try Self.object(value, ["kind", "id", "key"])
                guard let kind = Child.Kind(rawValue: try Self.text(c["kind"])) else { throw CompositeContractError(.invalidExpandedGraph, "Unknown typed child reference.") }
                return try Child(kind: kind, id: Self.text(c["id"]), key: Self.text(c["key"]))
            }
            let inputs = try Self.array(f["inputs"]).map { value -> Input in
                let i = try Self.object(value, ["id", "targets"])
                return try Input(id: Self.text(i["id"]), targets: Self.array(i["targets"]).map(endpoint))
            }
            let outputs = try Self.array(f["outputs"]).map { value -> Output in
                let o = try Self.object(value, ["id", "source"])
                guard let source = o["source"] else { throw CompositeContractError(.invalidExpandedGraph, "Missing composition output source.") }
                return try Output(id: Self.text(o["id"]), source: endpoint(source))
            }
            return try Container(id: Self.text(f["id"]), kind: kind, path: Self.array(f["path"]).map { try Self.text($0) }, children: children, inputs: inputs, outputs: outputs)
        }
        try validateStructure()
    }
    private func validateStructure() throws {
        try Self.require(schemaVersion == 1 && root == "root" && (1...129).contains(containers.count), "Composition plan version, root or container count is invalid.")
        try Self.require(containers.first?.id == root && containers.first?.path == [] && containers.first?.kind == .graph, "Composition plan must start at its root.")
        try Self.require(Set(containers.map(\.id)).count == containers.count, "Container identities must be unique in their typed namespace.")
        let childCount = containers.reduce(0) { $0 + $1.children.count }
        let endpointCount = containers.reduce(0) { count, container in
            count + container.inputs.reduce(0) { $0 + $1.targets.count } + container.outputs.count
        }
        try Self.require(childCount <= 256 && endpointCount <= 4096, "Composition plan exceeds its child or boundary budget.")
        let lookup = Dictionary(uniqueKeysWithValues: containers.map { ($0.id, $0) })
        var visited = Set<String>(), preorder: [String] = []
        func visit(_ id: String) throws {
            guard let c = lookup[id] else { throw CompositeContractError(.invalidExpandedGraph, "Composition child container is missing.") }
            try Self.require(visited.insert(id).inserted, "Container aliases and cycles are refused.")
            try Self.require(c.path.count <= 8 && c.id == Self.containerID(path: c.path) && c.path.allSatisfy { !$0.isEmpty && $0.unicodeScalars.count <= 128 }, "Container identity must derive from its bounded source instance path.")
            try Self.require((1...128).contains(c.children.count), "Native containers require 1–128 children.")
            try Self.require(Set(c.children.map { $0.kind.rawValue + ":" + $0.id }).count == c.children.count, "Typed children must be unique.")
            if c.kind == .namedMap {
                let pseudo = c.children.enumerated().map { CompositionPolicy.Child(localID: String($0.offset), key: $0.element.key) }
                let ports = Dictionary(uniqueKeysWithValues: pseudo.map { ($0.localID, CompositionPolicy.ChildPorts(inputs: [], outputs: ["out"])) })
                try CompositionPolicy(kind: .namedMap, children: pseudo).validate(childPorts: ports, edges: [], inputs: [], outputs: [.init("out", source: .init("0", "out"))])
            } else {
                try Self.require(c.children.enumerated().allSatisfy { $0.element.key == String($0.offset) }, "Ordered native storage keys must be exact numeric indices.")
            }
            try Self.require(c.inputs.count <= 128 && (1...128).contains(c.outputs.count) && Set(c.inputs.map(\.id)).count == c.inputs.count && Set(c.outputs.map(\.id)).count == c.outputs.count, "Container boundary identities must be unique and bounded.")
            for input in c.inputs { try Self.require(!input.id.isEmpty && input.id.unicodeScalars.count <= 128 && !input.targets.isEmpty && input.targets.count <= 256 && Set(input.targets).count == input.targets.count, "Container input targets are missing or duplicated.") }
            preorder.append(id)
            for child in c.children where child.kind == .container {
                guard let nested = lookup[child.id] else { throw CompositeContractError(.invalidExpandedGraph, "Missing typed container reference.") }
                try Self.require(nested.path.count == c.path.count + 1 && Array(nested.path.dropLast()) == c.path, "Nested container path must extend its actual parent.")
                try visit(child.id)
            }
        }
        try visit(root)
        try Self.require(preorder == containers.map(\.id), "Composition containers must be complete, non-aliased preorder.")
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        try Self.require(try encoder.encode(self).count <= 131072, "Composition plan exceeds 131072 UTF8 bytes.")
    }
    public func validate(graph: GraphDocument, registry: ComponentRegistry) throws {
        try validateStructure()
        try Self.require(graph.schemaVersion == 2 && graph.nodes.count <= 128 && graph.edges.count <= 256, "Composition requires a bounded primitive schema2 graph.")
        let sentinels = Set(graph.nodes.filter { ["input", "index_input", "bag_input", "output"].contains($0.type) }.map(\.id))
        try Self.require(Set(graph.nodes.map(\.id)).count == graph.nodes.count, "Primitive identities must be unique.")
        let nodes = Dictionary(uniqueKeysWithValues: graph.nodes.map { ($0.id, $0) })
        let lookup = Dictionary(uniqueKeysWithValues: containers.map { ($0.id, $0) })
        var shapeGraph = graph; shapeGraph.compositionPlan = nil
        let inferred = GraphValidator.validate(shapeGraph, registry: registry, context: .explicitObjective)
        try Self.require(inferred.isValid, "Composition flat graph is not executable: " + inferred.issues.map(\.message).joined(separator: " "))
        let leaves = containers.flatMap(\.children).filter { $0.kind == .node }.map(\.id)
        try Self.require(Set(leaves).count == leaves.count && Set(leaves) == Set(nodes.keys).subtracting(sentinels), "Every compute primitive must be registered once; sentinels cannot be native children.")
        func descendants(_ c: Container) -> Set<String> {
            c.children.reduce(into: Set<String>()) { set, child in
                if child.kind == .node { set.insert(child.id) } else if let nested = lookup[child.id] { set.formUnion(descendants(nested)) }
            }
        }
        func ports(_ id: String) throws -> ChildPortsProxy {
            guard let node = nodes[id], let definition = registry.component(type: node.type) else { throw CompositeContractError(.invalidExpandedGraph, "Unknown compute primitive in composition plan.") }
            let config = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 2, registry: registry)
            let incoming = graph.edges.first { $0.target == id && $0.targetPort == "in" }
            let layout = incoming.flatMap { inferred.outputTensorContracts[.init($0.source, $0.sourcePort)]?.layout }
            return ChildPortsProxy(inputs: definition.configuredInputs(config), outputs: definition.configuredOutputs(config, inputLayout: layout))
        }
        for container in containers {
            let owned = descendants(container)
            func check(_ e: Endpoint, input: Bool) throws {
                let p = try ports(e.node)
                try Self.require(owned.contains(e.node) && (input ? p.inputs : p.outputs).contains(e.port), "Container boundary must reach an owned configured primitive port.")
            }
            let targets = container.inputs.flatMap(\.targets), sources = container.outputs.map(\.source)
            try Self.require(Set(targets).count == targets.count, "A container input cannot have multiple boundary owners.")
            for e in targets { try check(e, input: true) }
            for e in sources { try check(e, input: false) }
            let incoming = graph.edges.filter { !owned.contains($0.source) && owned.contains($0.target) }.map { Endpoint(node: $0.target, port: $0.targetPort) }
            let outgoing = graph.edges.filter { owned.contains($0.source) && !owned.contains($0.target) }.map { Endpoint(node: $0.source, port: $0.sourcePort) }
            try Self.require(Set(incoming).isSubset(of: Set(targets)) && Set(outgoing).isSubset(of: Set(sources)), "Cross-container wires must use resolved declared boundaries.")
            for input in container.inputs {
                let wires = try input.targets.map { target -> GraphEdge in
                    let matches = graph.edges.filter { $0.target == target.node && $0.targetPort == target.port }
                    try Self.require(matches.count == 1 && !owned.contains(matches[0].source), "Each input boundary must resolve one external actual wire per target.")
                    return matches[0]
                }
                if container.id == root, let source = nodes[input.id], source.type == "bag_input" {
                    let configuration = try BagSourceConfiguration(values: source.config)
                    try Self.require(wires.allSatisfy { $0.source == source.id && configuration.outputPorts.contains($0.sourcePort) }, "Root bag input must resolve enabled roles of its actual source.")
                } else {
                    let addresses = Set(wires.map { Endpoint(node: $0.source, port: $0.sourcePort) })
                    try Self.require(addresses.count == 1, "A container input boundary cannot merge distinct external source ports.")
                }
            }
            if container.id == root {
                let inputNodes = graph.nodes.filter { ["input", "index_input", "bag_input"].contains($0.type) }
                let outputNodes = graph.nodes.filter { $0.type == "output" }
                try Self.require(container.inputs.map(\.id) == inputNodes.map(\.id) && container.outputs.map(\.id) == outputNodes.map(\.id), "Root boundaries must preserve actual source and output sentinel identities.")
                for input in container.inputs {
                    let expected = graph.edges.filter { $0.source == input.id }.map { Endpoint(node: $0.target, port: $0.targetPort) }
                    try Self.require(Set(input.targets) == Set(expected), "Root input boundary differs from actual flat wires.")
                }
                for output in container.outputs {
                    let expected = graph.edges.filter { $0.target == output.id }.map { Endpoint(node: $0.source, port: $0.sourcePort) }
                    try Self.require(expected == [output.source], "Root output boundary differs from actual flat wires.")
                }
            }
            if container.kind == .sequential {
                var ins: [[Endpoint]] = [], outs: [Endpoint] = []
                for child in container.children {
                    if child.kind == .node {
                        let p = try ports(child.id)
                        try Self.require(p.inputs.count == 1 && p.outputs.count == 1, "Sequence leaves must have exactly one input and output.")
                        ins.append([.init(node: child.id, port: p.inputs[0])]); outs.append(.init(node: child.id, port: p.outputs[0]))
                    } else {
                        let nested = lookup[child.id]!
                        try Self.require(nested.inputs.count == 1 && nested.inputs[0].targets.count == 1 && nested.outputs.count == 1, "Sequence child containers must expose one unary boundary.")
                        ins.append(nested.inputs[0].targets); outs.append(nested.outputs[0].source)
                    }
                }
                try Self.require(container.inputs.count == 1 && container.inputs[0].targets == ins[0] && container.outputs.count == 1 && container.outputs[0].source == outs[outs.count - 1], "Sequence boundaries must reach its first and last child.")
                for i in 0..<(container.children.count - 1) {
                    try Self.require(graph.edges.contains { $0.source == outs[i].node && $0.sourcePort == outs[i].port && $0.target == ins[i + 1][0].node && $0.targetPort == ins[i + 1][0].port }, "Sequence registration order must follow the actual flat chain.")
                }
            }
        }
    }
    private struct ChildPortsProxy { let inputs: [String]; let outputs: [String] }
}
