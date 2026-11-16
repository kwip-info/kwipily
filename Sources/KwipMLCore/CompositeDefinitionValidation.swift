import Foundation

/// Local construction validity; enclosing tensor widths are checked by execution expansion.
public enum CompositeDefinitionValidation {
    public static func validate(_ definition:CompositeDefinition,definitions:[CompositeDefinition],registry:ComponentRegistry)throws {
        guard definitions.count<=64 else {throw CompositeContractError(.resourceLimit,"At most 64 Block definitions are supported.")}
        var lookup:[CompositeDefinitionReference:CompositeDefinition]=[:]
        for item in definitions {guard lookup[item.reference] == nil else {throw CompositeContractError(.duplicateIdentity,"Duplicate definition reference.")};lookup[item.reference]=item}
        lookup[definition.reference]=definition
        var completed=Set<Visit>()
        try validate(definition,lookup:lookup,registry:registry,active:[],depth:0,completed:&completed)
    }
    public static func canExpose(_ field:ComponentField,to target:ComponentField)->Bool {
        guard field.isValidDefinition,target.isValidDefinition,field.kind == target.kind, !field.allowsNull || target.allowsNull else {return false}
        switch field.kind {
        case .choice:
            return Set(field.choices!.map(\.value)).isSubset(of: Set(target.choices!.map(\.value)))
        case .boolean:return true
        case .text:
            guard let lower=field.minItems,let upper=field.maxItems,let targetLower=target.minItems,let targetUpper=target.maxItems else {return false}
            return lower>=targetLower && upper<=targetUpper
        case .integerList, .numberList:
            guard let lower=field.minItems,let upper=field.maxItems,let targetLower=target.minItems,let targetUpper=target.maxItems,
                  lower>=targetLower,upper<=targetUpper,let min=field.min,let max=field.max,let targetMin=target.min,let targetMax=target.max else {return false}
            return min>=targetMin && max<=targetMax
        case .integer, .number:guard let min=field.min,let max=field.max,let targetMin=target.min,let targetMax=target.max else {return false};return min>=targetMin && max<=targetMax
        }
    }
    private struct Visit:Hashable {let reference:CompositeDefinitionReference;let depth:Int}
    private struct Ports {let inputs:[String];let outputs:[String];let fields:[ComponentField]}
    private static func validate(_ definition:CompositeDefinition,lookup:[CompositeDefinitionReference:CompositeDefinition],registry:ComponentRegistry,active:Set<CompositeDefinitionReference>,depth:Int,completed:inout Set<Visit>)throws {
        let version=definition.reference.version
        guard named(definition.reference.id),[version.major,version.minor,version.patch].allSatisfy({(0...Int(Int32.max)).contains($0)}) else {
            throw failure("Block definition references require a bounded identity and nonnegative version components.")
        }
        guard definition.body.configurationVersion == nil || [1,2].contains(definition.body.configurationVersion!) else {
            throw failure("Unsupported Block configuration version.")
        }
        guard depth<8,!active.contains(definition.reference) else {throw CompositeContractError(.recursion,"Block definitions contain recursion or exceed depth 8.")}
        let visit=Visit(reference:definition.reference,depth:depth)
        if completed.contains(visit) {return}
        let graph=definition.body,ids=graph.nodes.map(\.id)+graph.instances.map(\.id)
        guard ids.count<=128,graph.edges.count<=256,Set(ids).count == ids.count,ids.allSatisfy(named),Set(graph.edges.map(\.id)).count == graph.edges.count else {throw failure("Block body has duplicate identities or exceeds bounds.")}
        var ports:[String:Ports]=[:]
        for node in graph.nodes {
            guard !["input","output"].contains(node.type),node.x.isFinite,node.y.isFinite,let primitive=registry.component(type:node.type) else {throw failure("Block bodies require known primitives and explicit boundary ports.")}
            _ = try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.configurationVersion ?? 1,registry:registry)
            ports[node.id]=Ports(inputs:primitive.inputs,outputs:primitive.outputs,fields:primitive.fields)
        }
        let ancestry=active.union([definition.reference])
        for instance in graph.instances {
            guard instance.x.isFinite,instance.y.isFinite,let nested=lookup[instance.definition] else {throw failure("Nested Block definition is missing or has invalid placement.")}
            try validate(nested,lookup:lookup,registry:registry,active:ancestry,depth:depth+1,completed:&completed)
            let fields=nested.parameters.map(\.field)
            guard Set(instance.overrides.keys).isSubset(of:Set(fields.map(\.key))),fields.allSatisfy({$0.accepts(instance.overrides[$0.key] ?? $0.defaultValue)}) else {throw failure("Nested Block overrides do not match its exposed parameters.")}
            ports[instance.id]=Ports(inputs:nested.inputs.map(\.id),outputs:nested.outputs.map(\.id),fields:fields)
        }
        if let policy = definition.composition {
            var configured: [String: CompositionPolicy.ChildPorts] = [:]
            for node in graph.nodes {
                guard let declaration = registry.component(type: node.type) else { throw failure("Unknown composition primitive.") }
                let config = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.configurationVersion ?? 1, registry: registry)
                configured[node.id] = .init(inputs: declaration.configuredInputs(config), outputs: declaration.configuredOutputs(config))
                ports[node.id] = Ports(inputs: declaration.configuredInputs(config), outputs: declaration.configuredOutputs(config), fields: declaration.fields)
            }
            for instance in graph.instances { let child = lookup[instance.definition]!; configured[instance.id] = .init(inputs: child.inputs.map(\.id), outputs: child.outputs.map(\.id)) }
            if graph.nodes.contains(where: { ["split", "chunk"].contains($0.type) }) {
                // These output counts depend on the enclosing input layout. Check
                // authored membership now; expansion proves exact configured ports.
                _ = try JSONDecoder().decode(CompositionPolicy.self, from: JSONEncoder().encode(policy))
                guard Set(policy.children.map(\.localID)) == Set(configured.keys) else { throw failure("Composition must list each actual child exactly once.") }
            } else {
                try policy.validate(childPorts: configured, edges: graph.edges, inputs: definition.inputs, outputs: definition.outputs)
            }
        }
        func hasPort(_ port:CompositePortReference,output:Bool)->Bool {
            guard let component=ports[port.componentID] else {return false}
            return (output ? component.outputs:component.inputs).contains(port.portID)
        }
        var occupied=Set<CompositePortReference>()
        for edge in graph.edges {
            let from=CompositePortReference(edge.source,edge.sourcePort),to=CompositePortReference(edge.target,edge.targetPort)
            guard named(edge.id),hasPort(from,output:true),hasPort(to,output:false),occupied.insert(to).inserted else {throw failure("An internal wire has a missing or multiply connected endpoint.")}
        }
        guard definition.inputs.count<=128,definition.outputs.count<=128,definition.parameters.count<=128,
              Set(definition.inputs.map(\.id)).count == definition.inputs.count,Set(definition.outputs.map(\.id)).count == definition.outputs.count,
              Set(definition.parameters.map{$0.field.key}).count == definition.parameters.count,!definition.outputs.isEmpty else {throw failure("Block boundary names must be unique and expose at least one output.")}
        for input in definition.inputs {
            guard named(input.id),!input.targets.isEmpty,input.targets.count<=256 else {throw failure("Every boundary input needs a name and targets.")}
            for target in input.targets {guard hasPort(target,output:false),occupied.insert(target).inserted else {throw failure("A boundary input has an absent or already connected target.")}}
        }
        for output in definition.outputs {guard named(output.id),hasPort(output.source,output:true) else {throw failure("A boundary output has no valid source.")}}
        for (id,component) in ports {for input in component.inputs {guard occupied.contains(.init(id,input)) else {throw failure("Connect required input \(id).\(input) internally or expose it on the Block boundary.")}}}
        var remaining=Set(ids),done=Set<String>()
        while !remaining.isEmpty {
            let ready=remaining.filter{id in graph.edges.filter{$0.target == id}.allSatisfy{done.contains($0.source)}}
            guard !ready.isEmpty else {throw CompositeContractError(.recursion,"Block internal wires contain a cycle.")}
            remaining.subtract(ready);done.formUnion(ready)
        }
        var contributing=Set(definition.outputs.map{$0.source.componentID}),queue=Array(contributing)
        while let id=queue.popLast() {for edge in graph.edges where edge.target == id {if contributing.insert(edge.source).inserted {queue.append(edge.source)}}}
        guard contributing == Set(ids) else {throw failure("Every internal component must contribute to a declared Block output.")}
        _ = try CompositeConfigurationResolver.resolve(definition,definitions:Array(lookup.values),registry:registry)
        completed.insert(visit)
    }
    private static func named(_ text:String)->Bool {!text.isEmpty && text.unicodeScalars.count<=128}
    private static func failure(_ message:String)->CompositeContractError {CompositeContractError(.invalidBoundary,message)}
}
