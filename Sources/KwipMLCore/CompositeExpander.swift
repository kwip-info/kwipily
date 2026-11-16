import Foundation
import CryptoKit

public enum CompositeExpander {
    public static func expand(_ project:CompositeProjectDocument,registry:ComponentRegistry,context:GraphValidationContext = .bundledDataset)throws->CompositeExpansion {
        let worker=CompositeExpansionWorker(project:project,registry:registry,context:context)
        return try worker.expand()
    }
}
public enum CompositeProjectEdits {
    /// Existing pins are immutable through this authoring API. Identical re-add is idempotent.
    public static func addingRevision(_ definition:CompositeDefinition,to project:CompositeProjectDocument)throws->CompositeProjectDocument {
        if let existing=project.definitions.first(where:{$0.reference == definition.reference}) {
            guard existing == definition else {throw CompositeContractError(.immutableRevision,"A pinned definition revision cannot be replaced. Create a new version explicitly.")}
            return project
        }
        var copy=project;copy.definitions.append(definition);return copy
    }
    public static func pinningInstance(_ id:String,to reference:CompositeDefinitionReference,in project:CompositeProjectDocument)throws->CompositeProjectDocument {
        guard project.definitions.contains(where:{$0.reference == reference}) else {throw CompositeContractError(.missingDefinition,"The requested definition revision is not in this project.")}
        guard let index=project.root.instances.firstIndex(where:{$0.id == id}) else {throw CompositeContractError(.missingDefinition,"The root instance to upgrade does not exist. Nested instances belong to an immutable definition revision.")}
        var copy=project;copy.root.instances[index].definition=reference;return copy
    }
}

private struct CompositeScopeResolution {
    var inputs:[String:[CompositePortReference]]=[:]
    var outputs:[String:CompositePortReference]=[:]
}
private final class CompositeExpansionWorker {
    let project:CompositeProjectDocument
    let registry:ComponentRegistry
    let context:GraphValidationContext
    var definitions:[CompositeDefinitionReference:CompositeDefinition]=[:]
    var nodes:[GraphNode]=[]
    var edges:[GraphEdge]=[]
    var origins:[String:CompositeNodeOrigin]=[:]
    var consumed:[CompositeDefinitionReference]=[]
    var planEnabled = false
    var plannedContainers: [CompositionPlan.Container?] = []
    var policyChecks: [(CompositionPolicy, CompositeGraph, CompositeDefinition, [String: String])] = []
    var nodeIDs=Set<String>(),edgeIDs=Set<String>(),occupiedInputs=Set<CompositePortReference>()
    init(project:CompositeProjectDocument,registry:ComponentRegistry,context:GraphValidationContext) {self.project=project;self.registry=registry;self.context=context}
    func expand()throws->CompositeExpansion {
        guard [1,2,3].contains(project.schemaVersion) else {throw CompositeContractError(.unsupportedSchema,"Composite envelope schema 1, 2 or 3 is required; existing GraphDocument files are not silently reinterpreted.")}
        // Legacy flattening cannot represent native container order or ownership.
        // Policy-bearing sources require the separately admitted composition plan.
        guard project.schemaVersion == 3 || project.definitions.allSatisfy({ $0.composition == nil }) else {
            throw CompositeContractError(.unsupportedSchema,"Native container policies require the composition execution contract. This legacy source cannot be flattened without losing its ownership and order.")
        }
        guard project.definitions.count <= 64 else {throw CompositeContractError(.resourceLimit,"Composite library limit is 64 pinned revisions.")}
        for definition in project.definitions {
            guard !definition.reference.id.isEmpty,definition.reference.version.major >= 0,definition.reference.version.minor >= 0,definition.reference.version.patch >= 0,definitions[definition.reference] == nil else {throw CompositeContractError(.duplicateIdentity,"Definition references must be nonempty, versioned and unique.")}
            definitions[definition.reference]=definition
        }
        var visited=Set<CompositeDefinitionReference>(),active=Set<CompositeDefinitionReference>()
        func visit(_ reference:CompositeDefinitionReference)throws {
            guard let definition=definitions[reference] else {throw CompositeContractError(.missingDefinition,"Missing pinned definition \(reference.id).")}
            if active.contains(reference) {throw CompositeContractError(.recursion,"Recursive composite definitions are not supported; loops require an explicit state/control contract.")}
            guard visited.insert(reference).inserted else{return}
            active.insert(reference)
            for instance in definition.body.instances {try visit(instance.definition)}
            active.remove(reference)
        }
        for definition in project.definitions {try visit(definition.reference)}
        var reachable = Set<CompositeDefinitionReference>()
        func select(_ reference: CompositeDefinitionReference) throws {
            guard let definition = definitions[reference] else { throw CompositeContractError(.missingDefinition, "Missing reachable Block definition.") }
            guard reachable.insert(reference).inserted else { return }
            if definition.composition != nil { planEnabled = true }
            for instance in definition.body.instances { try select(instance.definition) }
        }
        for instance in project.root.instances { try select(instance.definition) }
        _ = try expandScope(project.root,path:[],definition:nil,x:0,y:0)
        var orderedContainers: [CompositionPlan.Container] = []
        if planEnabled {
            let complete = plannedContainers.compactMap { $0 }
            guard complete.count == plannedContainers.count, Set(complete.map(\.id)).count == complete.count else { throw CompositeContractError(.invalidExpandedGraph, "Composition container expansion is incomplete or duplicated.") }
            let lookup = Dictionary(uniqueKeysWithValues: complete.map { ($0.id, $0) })
            var seen = Set<String>()
            func appendContainer(_ id: String) throws {
                guard let container = lookup[id], seen.insert(id).inserted else { throw CompositeContractError(.invalidExpandedGraph, "Composition container has missing or aliased ownership.") }
                orderedContainers.append(container)
                for child in container.children where child.kind == .container { try appendContainer(child.id) }
            }
            try appendContainer("root")
            guard seen.count == complete.count else { throw CompositeContractError(.invalidExpandedGraph, "Composition has an unreachable native container.") }
        }
        let plan: CompositionPlan? = planEnabled ? CompositionPlan(containers: orderedContainers) : nil
        var graph=GraphDocument(name:project.name,nodes:nodes,edges:edges,execution:project.execution,compositionPlan:plan)
        graph.namedInputs=project.namedInputs

        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        if project.schemaVersion == 3 {
            guard try encoder.encode(project).count <= ModelProjectCodec.maximumBytes else { throw CompositeContractError(.resourceLimit, "Composite source exceeds 2 MB.") }
        }
        let validation=GraphValidator.validate(graph,registry:registry,context:context)
        guard validation.isValid else {
            let messages=validation.issues.map {issue->String in
                let location=issue.nodeID.flatMap {origins[$0]}.map {($0.instancePath+[$0.localNodeID]).joined(separator:"/")} ?? "root"
                return "\(location): \(issue.message)"
            }
            throw CompositeContractError(.invalidExpandedGraph,messages.joined(separator:" "),validationIssues:validation.issues,nodeOrigins:origins)
        }
        for (policy, body, definition, primitiveIDs) in policyChecks {
            var ports: [String: CompositionPolicy.ChildPorts] = [:]
            for node in body.nodes {
                guard let flatID = primitiveIDs[node.id], let flat = nodes.first(where: { $0.id == flatID }), let declaration = registry.component(type: flat.type) else { throw CompositeContractError(.invalidPrimitive, "Missing configured composition primitive.") }
                let incoming = edges.first { $0.target == flatID && $0.targetPort == "in" }
                let layout = incoming.flatMap { validation.outputTensorContracts[.init($0.source, $0.sourcePort)]?.layout }
                ports[node.id] = .init(inputs: declaration.configuredInputs(flat.config), outputs: declaration.configuredOutputs(flat.config, inputLayout: layout))
            }
            for instance in body.instances { let child = definitions[instance.definition]!; ports[instance.id] = .init(inputs: child.inputs.map(\.id), outputs: child.outputs.map(\.id)) }
            try policy.validate(childPorts: ports, edges: body.edges, inputs: definition.inputs, outputs: definition.outputs)
        }
        if let plan { try plan.validate(graph: graph, registry: registry) }
        return CompositeExpansion(graph:graph,nodeOrigins:origins,consumedDefinitions:consumed)
    }
    private func namespaced(_ id:String,path:[String],kind:String)throws->String {
        guard !id.isEmpty else {throw CompositeContractError(.duplicateIdentity,"Component and connection IDs must be nonempty.")}
        if path.isEmpty {return id}
        let data=try JSONEncoder().encode([kind]+path+[id])
        return kind+"-"+SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined()
    }
    private func expandScope(_ graph:CompositeGraph,path:[String],definition:CompositeDefinition?,x:Double,y:Double)throws->CompositeScopeResolution {
        guard path.count <= 8,graph.nodes.count+graph.instances.count <= 128,graph.edges.count <= 256 else {throw CompositeContractError(.resourceLimit,"Composite expansion exceeds depth 8 or the bounded graph limits.")}
        let ids=graph.nodes.map(\.id)+graph.instances.map(\.id)
        guard Set(ids).count == ids.count,!ids.contains(where:{$0.isEmpty}),Set(graph.edges.map(\.id)).count == graph.edges.count else {throw CompositeContractError(.duplicateIdentity,"Local component and wire identities must be unique.")}
        let containerIndex: Int? = planEnabled ? plannedContainers.count : nil
        if planEnabled { plannedContainers.append(nil) }
        var primitiveIDs:[String:String]=[:],nested:[String:CompositeScopeResolution]=[:]
        for node in graph.nodes {
            guard node.x.isFinite,node.y.isFinite else {throw CompositeContractError(.invalidPrimitive,"Component position must be finite.")}
            if definition != nil && (["input","bag_input","output"].contains(node.type) || (planEnabled && node.type == "index_input")) {throw CompositeContractError(.invalidPrimitive,"Composite internals use explicit interface bindings; data Input and task Output remain root nodes.")}
            let config:[String:JSONValue]
            do {config=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.configurationVersion ?? 1,registry:registry)}
            catch {throw CompositeContractError(.invalidParameter,error.localizedDescription)}
            let flatID=try namespaced(node.id,path:path,kind:"n")
            guard nodeIDs.insert(flatID).inserted else {throw CompositeContractError(.duplicateIdentity,"Flattened component identity collision.")}
            var flat=node;flat.config=config;flat.id=flatID;flat.x += x;flat.y += y
            guard flat.x.isFinite,flat.y.isFinite else {throw CompositeContractError(.invalidPrimitive,"Expanded component position is not finite.")}
            primitiveIDs[node.id]=flatID;nodes.append(flat)
            origins[flatID]=CompositeNodeOrigin(instancePath:path,localNodeID:node.id,definition:definition?.reference)
            if nodes.count > 128 {throw CompositeContractError(.resourceLimit,"Expanded model exceeds 128 primitive nodes.")}
        }
        for instance in graph.instances {
            guard instance.x.isFinite,instance.y.isFinite,let definition=definitions[instance.definition] else {throw CompositeContractError(.missingDefinition,"Instance \(instance.id) requires an available exact definition revision and finite position.")}
            if !consumed.contains(instance.definition) {consumed.append(instance.definition)}
            let body=try CompositeConfigurationResolver.resolve(definition,overrides:instance.overrides,definitions:Array(definitions.values),registry:registry)
            nested[instance.id]=try expandScope(body,path:path+[instance.id],definition:definition,x:x+instance.x,y:y+instance.y)
        }
        func input(_ reference:CompositePortReference)throws->[CompositePortReference] {
            if let id=primitiveIDs[reference.componentID],let node=graph.nodes.first(where:{$0.id == reference.componentID}),registry.component(type:node.type).map({ planEnabled ? $0.configuredInputs(nodes.first(where: { $0.id == id })!.config).contains(reference.portID) : $0.inputs.contains(reference.portID) }) == true {return [CompositePortReference(id,reference.portID)]}
            if let resolved=nested[reference.componentID]?.inputs[reference.portID] {return resolved}
            throw CompositeContractError(.invalidBoundary,"Missing declared input \(reference.componentID).\(reference.portID).")
        }
        func output(_ reference:CompositePortReference)throws->CompositePortReference {
            if let id=primitiveIDs[reference.componentID],let node=graph.nodes.first(where:{$0.id == reference.componentID}),registry.component(type:node.type).map({ planEnabled ? $0.configuredOutputs(nodes.first(where: { $0.id == id })!.config).contains(reference.portID) : $0.outputs.contains(reference.portID) }) == true {return CompositePortReference(id,reference.portID)}
            if let resolved=nested[reference.componentID]?.outputs[reference.portID] {return resolved}
            throw CompositeContractError(.invalidBoundary,"Missing declared output \(reference.componentID).\(reference.portID).")
        }
        var scope=CompositeScopeResolution(),bound=Set<CompositePortReference>()
        if let definition {
            guard Set(definition.inputs.map(\.id)).count == definition.inputs.count,Set(definition.outputs.map(\.id)).count == definition.outputs.count else {throw CompositeContractError(.invalidBoundary,"External port IDs must be unique within each direction.")}
            for port in definition.inputs {
                guard !port.id.isEmpty,!port.targets.isEmpty else {throw CompositeContractError(.invalidBoundary,"Each exposed input needs a name and explicit internal target(s).")}
                var targets:[CompositePortReference]=[]
                for target in port.targets {
                    for leaf in try input(target) {
                        guard bound.insert(leaf).inserted,!occupiedInputs.contains(leaf) else {throw CompositeContractError(.invalidBoundary,"An internal input cannot be bound by multiple interfaces or an existing internal wire.")}
                        targets.append(leaf)
                    }
                }
                scope.inputs[port.id]=targets
            }
            for port in definition.outputs {
                guard !port.id.isEmpty else {throw CompositeContractError(.invalidBoundary,"Exposed outputs require names.")}
                scope.outputs[port.id]=try output(port.source)
            }
        }
        for edge in graph.edges {
            let source=try output(CompositePortReference(edge.source,edge.sourcePort))
            let targets=try input(CompositePortReference(edge.target,edge.targetPort))
            for (index,target) in targets.enumerated() {
                guard !bound.contains(target),occupiedInputs.insert(target).inserted else {throw CompositeContractError(.invalidBoundary,"Input \(edge.target).\(edge.targetPort) has an ambiguous interface/internal connection or more than one source.")}
                // First fan-out wire retains its scoped original identity; extra wires are namespaced.
                let id=try namespaced(edge.id,path:index == 0 ? path : path+["$fanout",String(index)],kind:"e")
                guard edgeIDs.insert(id).inserted else {throw CompositeContractError(.duplicateIdentity,"Flattened wire identity collision.")}
                edges.append(GraphEdge(id:id,source:source.componentID,sourcePort:source.portID,target:target.componentID,targetPort:target.portID))
                if edges.count > 256 {throw CompositeContractError(.resourceLimit,"Expanded model exceeds 256 connections.")}
            }
        }
        if let containerIndex {
            let policy = definition?.composition
            if let policy, let definition { policyChecks.append((policy, graph, definition, primitiveIDs)) }
            let ordered = policy?.children ?? (graph.nodes.filter { !["input", "index_input", "bag_input", "output"].contains($0.type) }.map { CompositionPolicy.Child(localID: $0.id) } + graph.instances.map { CompositionPolicy.Child(localID: $0.id) })
            let children = try ordered.enumerated().map { index, child -> CompositionPlan.Child in
                let key: String
                if policy?.kind == .namedMap {
                    guard let named = child.key else { throw CompositeContractError(.invalidBoundary, "Named collection child requires a key.") }
                    key = named
                } else { key = String(index) }
                if let id = primitiveIDs[child.localID] { return .init(kind: .node, id: id, key: key) }
                guard nested[child.localID] != nil else { throw CompositeContractError(.invalidBoundary, "Unknown native container child.") }
                return .init(kind: .container, id: CompositionPlan.containerID(path: path + [child.localID]), key: key)
            }
            let planInputs: [CompositionPlan.Input]
            let planOutputs: [CompositionPlan.Output]
            if let definition {
                planInputs = definition.inputs.map { port in .init(id: port.id, targets: scope.inputs[port.id]!.map { .init(node: $0.componentID, port: $0.portID) }) }
                planOutputs = definition.outputs.map { port in let e = scope.outputs[port.id]!; return .init(id: port.id, source: .init(node: e.componentID, port: e.portID)) }
            } else {
                planInputs = graph.nodes.filter { ["input", "index_input", "bag_input"].contains($0.type) }.map { node in
                    .init(id: node.id, targets: edges.filter { $0.source == node.id }.map { .init(node: $0.target, port: $0.targetPort) })
                }
                planOutputs = try graph.nodes.filter { $0.type == "output" }.map { node in
                    let incoming = edges.filter { $0.target == node.id }
                    guard incoming.count == 1 else { throw CompositeContractError(.invalidBoundary, "Root output requires one actual source.") }
                    return .init(id: node.id, source: .init(node: incoming[0].source, port: incoming[0].sourcePort))
                }
            }
            plannedContainers[containerIndex] = .init(id: CompositionPlan.containerID(path: path), kind: policy?.kind ?? .graph, path: path, children: children, inputs: planInputs, outputs: planOutputs)
        }
        return scope
    }
}
