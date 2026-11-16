import KwipMLCore

/// Derived from the exact implemented registry bridge, never node labels or cached previews.
private func primitivePort(_ node:GraphNode,port:String,direction:PortDirection,registry:ComponentRegistry)->CanvasPortDescriptor {
    if direction == .output, let definition = registry.component(type:node.type),
       !definition.configuredOutputs(node.config).contains(port) {
        return .unknown(direction:direction,reason:"Output disabled by component settings")
    }
    guard let declarations=try? BuiltInCapabilityBridge.declarations(registry:registry),
          let contract=declarations.first(where:{$0.id == node.type})?.ports.first(where:{$0.id == port && $0.direction == direction}) else {return .unknown(direction:direction)}
    return .contract(contract,configuration:node.config)
}

extension WorkspaceStore {
    func canvasPortDescriptor(nodeID:String,port:String,direction:PortDirection)->CanvasPortDescriptor {
        guard let node=graph.nodes.first(where:{$0.id == nodeID}) else {return .unknown(direction:direction)}
        if TensorRoutingConfiguration.componentIDs.contains(node.type),
           let definition=validation.configuredDefinition(nodeID:nodeID,graph:graph,registry:registry),
           !(direction == .input ? definition.inputs:definition.outputs).contains(port) {
            return .unknown(direction:direction,reason:"Port unavailable with this configuration and source shape")
        }
        let inferred=validation.tensorPortDescriptor(nodeID:nodeID,port:port,direction:direction,graph:graph,registry:registry)
        if inferred.kind == .tensor,inferred.declared != ["Shape unresolved"] {return inferred}
        return primitivePort(node,port:port,direction:direction,registry:registry)
    }
}
extension DataFlowWorkspaceStore {
    func canvasConfiguration(nodeID:String)->[String:JSONValue] {project.graph.nodes.first(where:{$0.id == nodeID})?.config ?? [:]}
    func canvasPortDescriptor(nodeID:String,port:String,direction:PortDirection)->CanvasPortDescriptor {
        guard let node=project.graph.nodes.first(where:{$0.id == nodeID}),let definition=operators?.operatorDefinition(id:node.operatorID),let contract=(direction == .input ? definition.inputs:definition.outputs).first(where:{$0.id == port}) else {return .unknown(direction:direction)}
        return .data(contract,direction:direction)
    }
}
extension CompositeWorkspaceStore {
    private var effectiveCanvasBody: CompositeGraph? { try? configuredBody() }
    func canvasConfiguration(nodeID:String)->[String:JSONValue] {
        guard let body=effectiveCanvasBody else { return [:] }
        if let node=body.nodes.first(where:{$0.id == nodeID}) {return node.config}
        guard let instance=body.instances.first(where:{$0.id == nodeID}),let definition=project.definitions.first(where:{$0.reference == instance.definition}) else {return [:]}
        var values=definition.parameters.reduce(into:[String:JSONValue]()) {$0[$1.field.key]=$1.field.defaultValue}
        values.merge(instance.overrides) {_,override in override}
        return values
    }
    func canvasPortDescriptor(nodeID:String,port:String,direction:PortDirection)->CanvasPortDescriptor {
        guard let body=effectiveCanvasBody else { return .unknown(direction:direction,reason:projectionError ?? "Invalid Block configuration") }
        return compositePort(body: body, nodeID: nodeID, port: port, direction: direction, definitions: project.definitions, registry: primitiveRegistry)
    }
}

/// Resolves configured declarations through nested boundaries without claiming measured shapes.
private func compositePort(body: CompositeGraph, nodeID: String, port: String, direction: PortDirection,
                           definitions: [CompositeDefinition], registry: ComponentRegistry) -> CanvasPortDescriptor {
        func resolve(body:CompositeGraph,nodeID:String,port:String,visited:Set<CompositeDefinitionReference>)->CanvasPortDescriptor {
            if let primitive=body.nodes.first(where:{$0.id == nodeID}) {
                var value=primitivePort(primitive,port:port,direction:direction,registry:registry)
                // A general model boundary can serve any explicit objective. The
                // legacy classification declaration's fixed width is not its shape.
                if primitive.type == "output",port == "in",direction == .input,value.kind == .tensor {
                    value.declared=value.declared.map {$0.hasPrefix("[") ? "Shape follows connected input":$0}
                }
                return value
            }
            guard let instance=body.instances.first(where:{$0.id == nodeID}),!visited.contains(instance.definition),visited.count<16,
                  let definition=definitions.first(where:{$0.reference == instance.definition}) else {return .unknown(direction:direction,reason:"Composite boundary unresolved")}
            guard let inner=try? CompositeConfigurationResolver.resolve(definition, overrides: instance.overrides, definitions: definitions, registry: registry) else { return .unknown(direction:direction,reason:"Invalid Block configuration") }
            let references:[CompositePortReference]
            if direction == .input {references=definition.inputs.first(where:{$0.id == port})?.targets ?? []}
            else {references=definition.outputs.first(where:{$0.id == port}).map {[$0.source]} ?? []}
            let values=references.map {resolve(body:inner,nodeID:$0.componentID,port:$0.portID,visited:visited.union([instance.definition]))}
            guard let first=values.first,first.kind != .unknown,values.allSatisfy({$0 == first}) else {return .unknown(direction:direction,reason:"Composite boundary has unresolved or differing declarations")}
            return first
        }
    return resolve(body: body, nodeID: nodeID, port: port, visited: [])
}

extension ModelCanvasAdapter {
    func canvasPortDescriptor(nodeID: String, port: String, direction: PortDirection) -> CanvasPortDescriptor {
        guard let scope else { return .unknown(direction: direction, reason: "Block scope unavailable") }
        do {
            let body = try activeDefinition.map {
                try CompositeConfigurationResolver.resolve($0, definitions: definitions, registry: primitiveRegistry)
            } ?? scope
            return compositePort(body: body, nodeID: nodeID, port: port, direction: direction,
                                 definitions: definitions, registry: primitiveRegistry)
        } catch {
            return .unknown(direction: direction, reason: "Invalid Block configuration: " + error.localizedDescription)
        }
    }
}
