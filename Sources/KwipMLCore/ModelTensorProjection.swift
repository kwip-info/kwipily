import Foundation

/// Immutable inferred port presentation for a captured model. Construct once per
/// authoring revision; paint/hover queries never revalidate or expand the graph.
public struct ModelTensorProjection {
    private struct Address:Hashable {let path:[String];let node:String;let port:String;let input:Bool}
    private let descriptors:[Address:CanvasPortDescriptor]
    private let layouts:[Address:BatchedTensorLayout]
    private let inputTypes:[Address:ContractDType]
    public init(snapshot:ModelExecutionSnapshot,registry:ComponentRegistry,context:GraphValidationContext = .bundledDataset) {
        let validation=GraphValidator.validate(snapshot.graph,registry:registry,context:context)
        var values:[Address:CanvasPortDescriptor]=[:]
        var typed:[Address:BatchedTensorLayout]=[:]
        var types:[Address:ContractDType]=[:]
        for node in snapshot.graph.nodes {
            guard let origin=snapshot.origins[node.id],let definition=registry.component(type:node.type) else {continue}
            for (direction,ports) in [(PortDirection.input,definition.inputs),(.output,definition.outputs)] {
                for port in ports {
                    let key=Address(path:origin.instancePath,node:origin.localNodeID,port:port,input:direction == .input)
                    let descriptor=validation.tensorPortDescriptor(nodeID:node.id,port:port,direction:direction,graph:snapshot.graph,registry:registry)
                    // Never choose one of conflicting expanded origins silently.
                    if values[key] != nil {values[key] = .unknown(direction:direction,reason:"Ambiguous model origin");typed.removeValue(forKey:key);types.removeValue(forKey:key)}
                    else {
                        values[key]=descriptor
                        if direction == .input {
                            let incoming=snapshot.graph.edges.filter {$0.target==node.id && $0.targetPort==port}
                            if incoming.count==1,let edge=incoming.first {
                                typed[key]=validation.outputTensorContracts[.init(edge.source,edge.sourcePort)]?.layout
                                types[key]=validation.outputTensorContracts[.init(edge.source,edge.sourcePort)]?.dtype
                            }
                        }
                    }
                }
            }
        }
        if let source=snapshot.provenance?.sourceCompositeDocument {
            func visit(_ body:CompositeGraph,path:[String],active:Set<CompositeDefinitionReference>) {
                guard path.count<16 else {return}
                for instance in body.instances {
                    guard !active.contains(instance.definition),let definition=source.definitions.first(where:{$0.reference == instance.definition}) else {continue}
                    let innerPath=path+[instance.id]
                    visit(definition.body,path:innerPath,active:active.union([instance.definition]))
                    func resolve(_ references:[CompositePortReference],direction:PortDirection)->CanvasPortDescriptor {
                        let candidates=references.compactMap {reference in
                            values[Address(path:innerPath,node:reference.componentID,port:reference.portID,input:direction == .input)]
                        }
                        guard !candidates.isEmpty,candidates.count == references.count,let first=candidates.first,
                              first.kind == .tensor,candidates.allSatisfy({$0 == first}) else {
                            return .unknown(direction:direction,reason:"Block boundary shape is unresolved or inconsistent")
                        }
                        return first
                    }
                    for input in definition.inputs {
                        values[Address(path:path,node:instance.id,port:input.id,input:true)]=resolve(input.targets,direction:.input)
                    }
                    for output in definition.outputs {
                        values[Address(path:path,node:instance.id,port:output.id,input:false)]=resolve([output.source],direction:.output)
                    }
                }
            }
            visit(source.root,path:[],active:[])
        }
        descriptors=values;layouts=typed;inputTypes=types
    }
    public func containsInput(instancePath:[String],nodeID:String,port:String)->Bool {
        descriptors[Address(path:instancePath,node:nodeID,port:port,input:true)] != nil
    }
    public func inputLayout(instancePath:[String],nodeID:String,port:String)->BatchedTensorLayout? {
        layouts[Address(path:instancePath,node:nodeID,port:port,input:true)]
    }
    public func inputContract(instancePath:[String],nodeID:String,port:String)->TypedTensorContract? {
        let key=Address(path:instancePath,node:nodeID,port:port,input:true)
        guard let layout=layouts[key],let dtype=inputTypes[key] else {return nil}
        return .init(layout:layout,dtype:dtype)
    }
    public func port(instancePath:[String],nodeID:String,port:String,direction:PortDirection)->CanvasPortDescriptor {
        descriptors[Address(path:instancePath,node:nodeID,port:port,input:direction == .input)] ?? .unknown(direction:direction,reason:"Model origin or port unavailable")
    }
}
