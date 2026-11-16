import Foundation

/// Transactional composition edits; callers validate the resulting definition/expanded shapes before Apply.
public enum ModuleCompositionEditing {
    public typealias Validation = (CompositeDefinition) throws -> Void
    private static func failure(_ message: String) -> CompositeContractError { .init(.invalidBoundary, message) }
    private static func membership(_ definition: CompositeDefinition, ports: [String: CompositionPolicy.ChildPorts]) throws -> [String] {
        let ids = definition.body.nodes.map(\.id) + definition.body.instances.map(\.id)
        guard (1...128).contains(ids.count), Set(ids).count == ids.count, Set(ids) == Set(ports.keys) else {
            throw failure("Provide the current configured ports for every actual Block child exactly once.")
        }
        return ids
    }
    private static func checkedPolicy(_ definition: CompositeDefinition, ports: [String: CompositionPolicy.ChildPorts]) throws -> CompositionPolicy {
        _ = try membership(definition, ports: ports)
        guard let policy = definition.composition else { throw failure("Select a container kind before editing its stored children.") }
        try policy.validate(childPorts: ports, edges: definition.body.edges, inputs: definition.inputs, outputs: definition.outputs)
        return policy
    }
    private static func finish(_ definition: CompositeDefinition, ports: [String: CompositionPolicy.ChildPorts], validate: Validation) throws -> CompositeDefinition {
        _ = try checkedPolicy(definition, ports: ports)
        try validate(definition)
        return definition
    }
    private static func keyed(_ ids: [String], surviving: [CompositionPolicy.Child] = []) -> [CompositionPolicy.Child] {
        var used = Set(surviving.compactMap(\.key)), next = 1
        let previous = Dictionary(uniqueKeysWithValues: surviving.map { ($0.localID, $0) })
        return ids.map { id in
            if let child = previous[id], child.key != nil { return child }
            while used.contains("child_\(next)") || CompositionPolicy.nativeReservedKeys.contains("child_\(next)") { next += 1 }
            let key = "child_\(next)"; used.insert(key); next += 1
            return .init(localID: id, key: key)
        }
    }
    private static func chain(_ definition: CompositeDefinition, ports: [String: CompositionPolicy.ChildPorts]) throws -> [String] {
        guard definition.inputs.count == 1, definition.inputs[0].targets.count == 1,
              definition.outputs.count == 1, definition.body.edges.count == ports.count - 1,
              ports.values.allSatisfy({ $0.inputs.count == 1 && $0.outputs.count == 1 }) else {
            throw failure("A sequence needs one complete existing unary chain, with one exposed input and output.")
        }
        var order: [String] = [], current = definition.inputs[0].targets[0].componentID
        while !order.contains(current) {
            guard let child = ports[current] else { throw failure("The sequence references an unavailable child.") }
            order.append(current)
            let edges = definition.body.edges.filter { $0.source == current }
            if edges.isEmpty { break }
            guard edges.count == 1, let edge = edges.first, edge.sourcePort == child.outputs[0],
                  let target = ports[edge.target], edge.targetPort == target.inputs[0] else {
                throw failure("Repair branches, extra ports or invalid connections before selecting Sequence.")
            }
            current = edge.target
        }
        guard order.count == ports.count, Set(order) == Set(ports.keys) else {
            throw failure("The existing wires must visit every child exactly once.")
        }
        return order
    }
    public static func selecting(kind: CompositionPolicy.Kind, in definition: CompositeDefinition,
        childPorts: [String: CompositionPolicy.ChildPorts], validate: Validation) throws -> CompositeDefinition {
        let ids = try membership(definition, ports: childPorts)
        if definition.composition != nil { _ = try checkedPolicy(definition, ports: childPorts) }
        let order: [String]
        if kind == .sequential { order = try chain(definition, ports: childPorts) }
        else { order = definition.composition?.children.map(\.localID) ?? ids }
        let children: [CompositionPolicy.Child] = kind == .namedMap
            ? keyed(order, surviving: definition.composition?.kind == .namedMap ? definition.composition!.children : [])
            : order.map { .init(localID: $0) }
        var result = definition; result.composition = .init(kind: kind, children: children)
        return try finish(result, ports: childPorts, validate: validate)
    }
    public static func refreshingMembership(in definition: CompositeDefinition,
        childPorts: [String: CompositionPolicy.ChildPorts], validate: Validation) throws -> CompositeDefinition {
        let ids = try membership(definition, ports: childPorts)
        guard let policy = definition.composition else { throw failure("Select a container kind before refreshing membership.") }
        // Decode validates structural/key bounds without pretending the stale child list is current.
        _ = try JSONDecoder().decode(CompositionPolicy.self, from: JSONEncoder().encode(policy))
        let survivors = policy.children.filter { childPorts[$0.localID] != nil }
        let survivorIDs = Set(survivors.map(\.localID))
        let order = survivors.map(\.localID) + ids.filter { !survivorIDs.contains($0) }
        let children = policy.kind == .namedMap ? keyed(order, surviving: survivors) : order.map { CompositionPolicy.Child(localID: $0) }
        var result = definition; result.composition = .init(kind: policy.kind, children: children)
        return try finish(result, ports: childPorts, validate: validate)
    }
    public static func movingChild(localID: String, to index: Int, in definition: CompositeDefinition,
        childPorts: [String: CompositionPolicy.ChildPorts], validate: Validation) throws -> CompositeDefinition {
        let policy = try checkedPolicy(definition, ports: childPorts)
        guard let oldIndex = policy.children.firstIndex(where: { $0.localID == localID }), policy.children.indices.contains(index) else {
            throw failure("Choose an existing child and a position within the current container.")
        }
        var children = policy.children; let child = children.remove(at: oldIndex); children.insert(child, at: index)
        var result = definition; result.composition = .init(kind: policy.kind, children: children)
        if policy.kind == .sequential {
            // Wire identity follows its old chain position, not the body's storage ordering.
            for position in 0..<(children.count - 1) {
                let oldSource = policy.children[position].localID, oldTarget = policy.children[position + 1].localID
                guard let edgeIndex = definition.body.edges.firstIndex(where: { $0.source == oldSource && $0.target == oldTarget }) else {
                    throw failure("The captured sequence wire is no longer available.")
                }
                let source = children[position].localID, target = children[position + 1].localID
                result.body.edges[edgeIndex].source = source; result.body.edges[edgeIndex].sourcePort = childPorts[source]!.outputs[0]
                result.body.edges[edgeIndex].target = target; result.body.edges[edgeIndex].targetPort = childPorts[target]!.inputs[0]
            }
            let first = children[0].localID, last = children[children.count - 1].localID
            result.inputs[0].targets = [.init(first, childPorts[first]!.inputs[0])]
            result.outputs[0].source = .init(last, childPorts[last]!.outputs[0])
        }
        return try finish(result, ports: childPorts, validate: validate)
    }
    public static func renamingKey(localID: String, key: String, in definition: CompositeDefinition,
        childPorts: [String: CompositionPolicy.ChildPorts], validate: Validation) throws -> CompositeDefinition {
        let policy = try checkedPolicy(definition, ports: childPorts)
        guard policy.kind == .namedMap, let index = policy.children.firstIndex(where: { $0.localID == localID }) else {
            throw failure("Choose an existing child in a named collection before changing its storage key.")
        }
        var children = policy.children; children[index] = .init(localID: localID, key: key)
        var result = definition; result.composition = .init(kind: .namedMap, children: children)
        return try finish(result, ports: childPorts, validate: validate)
    }
}
