import Foundation

/// Explicit native registration order for one authored Block definition.
public struct CompositionPolicy: Codable, Equatable {
    public enum Kind: String, Codable, CaseIterable { case graph, sequential, orderedList, namedMap }
    public struct Child: Codable, Equatable {
        public let localID: String
        public let key: String?
        public init(localID: String, key: String? = nil) { self.localID = localID; self.key = key }
        private enum CodingKeys: String, CodingKey { case localID, key }
        public init(from decoder: Decoder) throws {
            try CompositionPolicy.exact(decoder, required: ["localID"], optional: ["key"])
            let c = try decoder.container(keyedBy: CodingKeys.self)
            localID = try c.decode(String.self, forKey: .localID)
            key = c.contains(.key) ? try c.decode(String.self, forKey: .key) : nil
        }
    }
    public struct ChildPorts: Equatable {
        public let inputs: [String]
        public let outputs: [String]
        public init(inputs: [String], outputs: [String]) { self.inputs = inputs; self.outputs = outputs }
    }
    public let schemaVersion: Int
    public let kind: Kind
    public let children: [Child]
    public init(schemaVersion: Int = 1, kind: Kind, children: [Child]) {
        self.schemaVersion = schemaVersion; self.kind = kind; self.children = children
    }
    private enum CodingKeys: String, CodingKey { case schemaVersion, kind, children }
    private struct WireKey: CodingKey {
        let stringValue: String
        let intValue: Int? = nil
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    private static func exact(_ decoder: Decoder, required: Set<String>, optional: Set<String> = []) throws {
        let fields = Set(try decoder.container(keyedBy: WireKey.self).allKeys.map(\.stringValue))
        try require(required.isSubset(of: fields) && fields.isSubset(of: required.union(optional)), "Composition policy has missing or unknown fields.")
    }
    public init(from decoder: Decoder) throws {
        try Self.exact(decoder, required: ["schemaVersion", "kind", "children"])
        let c = try decoder.container(keyedBy: CodingKeys.self)
        schemaVersion = try c.decode(Int.self, forKey: .schemaVersion)
        kind = try c.decode(Kind.self, forKey: .kind)
        children = try c.decode([Child].self, forKey: .children)
        try validateStructure()
    }
    private static func require(_ condition: Bool, _ message: String) throws {
        if !condition { throw CompositeContractError(.invalidBoundary, message) }
    }
    private static func name(_ value: String) throws {
        try require((1...128).contains(value.unicodeScalars.count), "Composition identities require 1–128 Unicode scalars.")
    }
    private func validateStructure() throws {
        try Self.require(schemaVersion == 1, "Composition policy requires schema version 1.")
        try Self.require((1...128).contains(children.count), "List 1–128 actual Block children.")
        var ids = Set<String>(), keys = Set<String>()
        for child in children {
            try Self.name(child.localID)
            try Self.require(ids.insert(child.localID).inserted, "List each Block child exactly once.")
            if kind == .namedMap {
                guard let key = child.key else { throw CompositeContractError(.invalidBoundary, "Give each named child a unique key.") }
                let scalars = Array(key.unicodeScalars)
                let letter: (Unicode.Scalar) -> Bool = { (65...90).contains($0.value) || (97...122).contains($0.value) || $0.value == 95 }
                try Self.require((1...64).contains(scalars.count) && scalars.first.map(letter) == true && scalars.allSatisfy { letter($0) || (48...57).contains($0.value) }, "Named keys use ASCII letters, digits or underscores and start with a letter or underscore.")
                try Self.require(!Self.nativeReservedKeys.contains(key), "This key collides with a native module attribute. Choose another name.")
                try Self.require(keys.insert(key).inserted, "Named child keys must be unique.")
            } else {
                try Self.require(child.key == nil, "Only a named collection accepts explicit child keys.")
            }
        }
    }
    public func validate(childPorts: [String: ChildPorts], edges: [GraphEdge], inputs: [CompositeInput], outputs: [CompositeOutput]) throws {
        try validateStructure()
        try Self.require((1...128).contains(childPorts.count) && Set(children.map(\.localID)) == Set(childPorts.keys), "Include every actual Block child exactly once; foreign children are refused.")
        for (id, ports) in childPorts {
            try Self.name(id)
            for names in [ports.inputs, ports.outputs] {
                try Self.require(names.count <= 128 && Set(names).count == names.count, "Configured port names must be unique and bounded.")
                for port in names { try Self.name(port) }
            }
        }
        try Self.require(edges.count <= 256 && inputs.count <= 128 && (1...128).contains(outputs.count), "Block connections or boundaries exceed their limits.")
        func endpoint(_ reference: CompositePortReference, input: Bool) throws {
            try Self.name(reference.componentID); try Self.name(reference.portID)
            guard let ports = childPorts[reference.componentID] else { throw CompositeContractError(.invalidBoundary, "Boundary names an unknown Block child.") }
            try Self.require((input ? ports.inputs : ports.outputs).contains(reference.portID), "Use an actual configured child port.")
        }
        var occupied = Set<CompositePortReference>(), edgeIDs = Set<String>()
        for edge in edges {
            for value in [edge.id, edge.source, edge.sourcePort, edge.target, edge.targetPort] { try Self.name(value) }
            try Self.require(edgeIDs.insert(edge.id).inserted, "Connection identities must be unique.")
            try endpoint(.init(edge.source, edge.sourcePort), input: false)
            let target = CompositePortReference(edge.target, edge.targetPort)
            try endpoint(target, input: true)
            try Self.require(occupied.insert(target).inserted, "Each child input accepts one internal or boundary connection.")
        }
        var inputIDs = Set<String>(), outputIDs = Set<String>()
        for input in inputs {
            try Self.name(input.id)
            try Self.require(inputIDs.insert(input.id).inserted && (1...256).contains(input.targets.count), "Boundary inputs need unique identities and bounded targets.")
            for target in input.targets {
                try endpoint(target, input: true)
                try Self.require(occupied.insert(target).inserted, "Each child input accepts one internal or boundary connection.")
            }
        }
        for output in outputs {
            try Self.name(output.id)
            try Self.require(outputIDs.insert(output.id).inserted, "Boundary output identities must be unique.")
            try endpoint(output.source, input: false)
        }
        if kind == .sequential {
            try Self.require(childPorts.values.allSatisfy { $0.inputs.count == 1 && $0.outputs.count == 1 }, "Sequence children require exactly one configured input and output.")
            try Self.require(inputs.count == 1 && inputs[0].targets.count == 1 && outputs.count == 1 && edges.count == children.count - 1, "A sequence exposes one input and output and one complete wired chain.")
            let first = children[0].localID, last = children[children.count - 1].localID
            try Self.require(inputs[0].targets[0] == .init(first, childPorts[first]!.inputs[0]) && outputs[0].source == .init(last, childPorts[last]!.outputs[0]), "Sequence boundaries must reach the first and last listed child.")
            for index in 0..<(children.count - 1) {
                let a = children[index].localID, b = children[index + 1].localID
                try Self.require(edges.contains { $0.source == a && $0.sourcePort == childPorts[a]!.outputs[0] && $0.target == b && $0.targetPort == childPorts[b]!.inputs[0] }, "Sequence order must match every connection in the single chain.")
            }
        }
    }
    // Filled from the reviewed Torch 2.14.1 Module/ModuleDict surface; runtime
    // additionally checks its installed native attributes before registration.
    internal static let nativeReservedKeys: Set<String> = [
        "T_destination",
        "__annotations__",
        "__call__",
        "__class__",
        "__contains__",
        "__delattr__",
        "__delitem__",
        "__dict__",
        "__dir__",
        "__doc__",
        "__eq__",
        "__firstlineno__",
        "__format__",
        "__ge__",
        "__getattr__",
        "__getattribute__",
        "__getitem__",
        "__getstate__",
        "__gt__",
        "__hash__",
        "__init__",
        "__init_subclass__",
        "__iter__",
        "__le__",
        "__len__",
        "__lt__",
        "__module__",
        "__ne__",
        "__new__",
        "__reduce__",
        "__reduce_ex__",
        "__repr__",
        "__setattr__",
        "__setitem__",
        "__setstate__",
        "__sizeof__",
        "__static_attributes__",
        "__str__",
        "__subclasshook__",
        "__weakref__",
        "_apply",
        "_backward_hooks",
        "_backward_pre_hooks",
        "_buffers",
        "_call_impl",
        "_compiled_call_impl",
        "_forward_hooks",
        "_forward_hooks_always_called",
        "_forward_hooks_with_kwargs",
        "_forward_pre_hooks",
        "_forward_pre_hooks_with_kwargs",
        "_get_backward_hooks",
        "_get_backward_pre_hooks",
        "_get_name",
        "_is_full_backward_hook",
        "_load_from_state_dict",
        "_load_state_dict_post_hooks",
        "_load_state_dict_pre_hooks",
        "_maybe_warn_non_full_backward_hook",
        "_modules",
        "_named_members",
        "_non_persistent_buffers_set",
        "_parameters",
        "_register_load_state_dict_pre_hook",
        "_register_state_dict_hook",
        "_replicate_for_data_parallel",
        "_save_to_state_dict",
        "_slow_forward",
        "_state_dict_hooks",
        "_state_dict_pre_hooks",
        "_version",
        "_wrapped_call_impl",
        "add_module",
        "apply",
        "bfloat16",
        "buffers",
        "call_super_init",
        "children",
        "clear",
        "compile",
        "cpu",
        "cuda",
        "double",
        "dump_patches",
        "eval",
        "extra_repr",
        "float",
        "forward",
        "get_buffer",
        "get_extra_state",
        "get_parameter",
        "get_submodule",
        "half",
        "ipu",
        "items",
        "keys",
        "load_state_dict",
        "modules",
        "mtia",
        "named_buffers",
        "named_children",
        "named_modules",
        "named_parameters",
        "parameters",
        "pop",
        "register_backward_hook",
        "register_buffer",
        "register_forward_hook",
        "register_forward_pre_hook",
        "register_full_backward_hook",
        "register_full_backward_pre_hook",
        "register_load_state_dict_post_hook",
        "register_load_state_dict_pre_hook",
        "register_module",
        "register_parameter",
        "register_state_dict_post_hook",
        "register_state_dict_pre_hook",
        "requires_grad_",
        "set_extra_state",
        "set_submodule",
        "share_memory",
        "state_dict",
        "to",
        "to_empty",
        "train",
        "training",
        "type",
        "update",
        "values",
        "xpu",
        "zero_grad"
    ]
}
