import Foundation

public struct GraphSelection:Equatable,Codable {
    public var nodeIDs:Set<String>;public var edgeIDs:Set<String>
    public init(nodeIDs:Set<String>=[],edgeIDs:Set<String>=[]) {self.nodeIDs=nodeIDs;self.edgeIDs=edgeIDs}
}
public struct GraphEditorState:Equatable {
    public var document:GraphDocument;public var selection:GraphSelection
    public init(document:GraphDocument,selection:GraphSelection=GraphSelection()) {self.document=document;self.selection=selection}
}
public struct GraphPortAddress:Hashable {
    public var nodeID:String;public var portID:String
    public init(_ nodeID:String,_ portID:String) {self.nodeID=nodeID;self.portID=portID}
}
public struct GraphOffset:Equatable {
    public var x:Double;public var y:Double
    public init(x:Double=30,y:Double=30) {self.x=x;self.y=y}
}
public enum GraphSelectionMode {case replace,add,remove,toggle}
public enum GraphCopySemantics {case independent,sharedParameters}
public enum GraphCommand {
    case select(GraphSelection,mode:GraphSelectionMode)
    case setActivation(nodeID:String,function:ActivationConfiguration.Function,configuration:[String:JSONValue]? = nil)
    case setPooling(nodeID:String,variant:PoolingConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setConvolution(nodeID:String,variant:ConvolutionConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setBilinear(nodeID:String,configuration:[String:JSONValue])
    case setEmbedding(nodeID:String,configuration:[String:JSONValue])
    case setAttention(nodeID:String,configuration:[String:JSONValue])
    case setEmbeddingBag(nodeID:String,configuration:[String:JSONValue])
    case setRecurrent(nodeID:String,method:RecurrentSequenceConfiguration.Method,configuration:[String:JSONValue]? = nil)
    case setTensorOperation(nodeID:String,type:String,configuration:[String:JSONValue])
    case setReshape(nodeID:String,variant:ReshapeAuthoringConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setLinear(nodeID:String,variant:LinearConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case setNormalization(nodeID:String,variant:NormalizationConfiguration.Variant,configuration:[String:JSONValue]? = nil)
    case move(nodeIDs:Set<String>,offset:GraphOffset)
    case duplicate(nodeIDs:Set<String>,offset:GraphOffset,semantics:GraphCopySemantics)
    case delete(GraphSelection)
    case connect(source:GraphPortAddress,target:GraphPortAddress)
    case reconnect(edgeID:String,source:GraphPortAddress,target:GraphPortAddress)
    case insertOnWire(edgeID:String,node:GraphNode,inputPort:String,outputPort:String)
}
public struct GraphCopyMap:Equatable {
    public var nodes:[String:String];public var edges:[String:String]
    public init(nodes:[String:String]=[:],edges:[String:String]=[:]) {self.nodes=nodes;self.edges=edges}
}
public struct GraphCommandDelta:Equatable {
    public var insertedNodeIDs:Set<String>;public var removedNodeIDs:Set<String>;public var updatedNodeIDs:Set<String>;public var insertedEdgeIDs:Set<String>;public var removedEdgeIDs:Set<String>;public var updatedEdgeIDs:Set<String>
    public init(insertedNodeIDs:Set<String>=[],removedNodeIDs:Set<String>=[],updatedNodeIDs:Set<String>=[],insertedEdgeIDs:Set<String>=[],removedEdgeIDs:Set<String>=[],updatedEdgeIDs:Set<String>=[]) {self.insertedNodeIDs=insertedNodeIDs;self.removedNodeIDs=removedNodeIDs;self.updatedNodeIDs=updatedNodeIDs;self.insertedEdgeIDs=insertedEdgeIDs;self.removedEdgeIDs=removedEdgeIDs;self.updatedEdgeIDs=updatedEdgeIDs}
}
public struct GraphCommandResult:Equatable {
    public var before:GraphEditorState;public var after:GraphEditorState;public var delta:GraphCommandDelta;public var copies:GraphCopyMap
    public var changesDocument:Bool {before.document != after.document}
    public init(before:GraphEditorState,after:GraphEditorState,delta:GraphCommandDelta=GraphCommandDelta(),copies:GraphCopyMap=GraphCopyMap()) {self.before=before;self.after=after;self.delta=delta;self.copies=copies}
}
public struct GraphCommandError:Error,Equatable,LocalizedError {
    public enum Code:String {case invalidDocument,invalidSelection,unknownComponent,missingNode,missingEdge,missingPort,occupiedPort,cycle,incompatibleShape,unsupportedSharing,invalidConfiguration,invalidPosition,identityCollision,resourceLimit}
    public let code:Code;public let message:String
    public init(_ code:Code,_ message:String) {self.code=code;self.message=message}
    public var errorDescription:String? {message}
}

/// Pure commands: the caller installs `after` only after success, and records the
/// complete before/after values as one undo transaction. No draft is run here.
public enum GraphCommandEngine {
    /// Custom domains may replace shape semantics; endpoint, occupancy and cycle checks always run.
    public static func apply(_ command:GraphCommand,to state:GraphEditorState,registry:ComponentRegistry,connectionSemantics:((GraphEdge,GraphDocument)throws->Void)?=nil,makeID:()->String={UUID().uuidString})throws->GraphCommandResult {
        try checkIdentity(state.document)
        var document=state.document,selection=state.selection,copies=GraphCopyMap()
        var reserved=Set(document.nodes.map(\.id)+document.edges.map(\.id))
        func newIdentity()throws->String {
            let id=makeID()
            guard validNewIdentity(id),reserved.insert(id).inserted else {throw GraphCommandError(.identityCollision,"Generated identity is invalid or already belongs to this document.")}
            return id
        }
        func checkSelection(_ value:GraphSelection)throws {
            guard value.nodeIDs.isSubset(of:Set(document.nodes.map(\.id))),value.edgeIDs.isSubset(of:Set(document.edges.map(\.id))) else {throw GraphCommandError(.invalidSelection,"Selection refers to a missing node or connection.")}
        }
        func moved(_ node:GraphNode,_ offset:GraphOffset)throws->GraphNode {
            let x=node.x+offset.x,y=node.y+offset.y
            guard offset.x.isFinite,offset.y.isFinite,x.isFinite,y.isFinite else {throw GraphCommandError(.invalidPosition,"Positions and movement offsets must be finite.")}
            var copy=node;copy.x=x;copy.y=y;return copy
        }
        switch command {
        case .setAttention(let id,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected Attention component no longer exists.")}
            let inputs=Set(document.edges.filter {$0.target==id}.map(\.targetPort))
            let outputs=Set(document.edges.filter {$0.source==id}.map(\.sourcePort))
            let changed=try AttentionConfiguration.applying(configuration,to:document.nodes[position],connectedInputPorts:inputs,connectedOutputPorts:outputs,registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setEmbeddingBag(let id,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected Embedding bag layer no longer exists.")}
            let inputs=Set(document.edges.filter {$0.target==id}.map(\.targetPort))
            let changed=try EmbeddingBagConfiguration.applying(configuration,to:document.nodes[position],connectedInputPorts:inputs,registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setEmbedding(let id,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected Embedding layer no longer exists.")}
            let changed=try EmbeddingConfiguration.applying(configuration,to:document.nodes[position],registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setBilinear(let id,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected Bilinear layer no longer exists.")}
            let changed=try BilinearConfiguration.applying(configuration,to:document.nodes[position],registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setRecurrent(let id,let method,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected recurrent no longer exists.")}
            let inputs=Set(document.edges.filter {$0.target==id}.map(\.targetPort))
            let outputs=Set(document.edges.filter {$0.source==id}.map(\.sourcePort))
            let changed=try RecurrentAuthoringConfiguration.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:method,configuration:configuration,connectedInputPorts:inputs,connectedOutputPorts:outputs,registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setTensorOperation(let id,let type,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected tensor operation no longer exists.")}
            let contracts=TensorOperationAuthoring.incomingContracts(nodeID:id,graph:document,validation:GraphValidator.validate(document,registry:registry))
            let inputs=contracts.mapValues(\.layout)
            let changed=try TensorOperationAuthoring.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:type,configuration:configuration,
                connectedInputs:Set(document.edges.filter {$0.target==id}.map(\.targetPort)),connectedOutputs:Set(document.edges.filter {$0.source==id}.map(\.sourcePort)),inputs:inputs,registry:registry,inputContracts:contracts)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry);document.nodes.insert(changed,at:position)
            }
        case .setReshape(let id,let variant,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected transform no longer exists.")}
            let changed=try ReshapeAuthoringConfiguration.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:variant,configuration:configuration,registry:registry)
            if let input=ReshapeAuthoringConfiguration.inputLayout(nodeID:id,graph:document,validation:GraphValidator.validate(document,registry:registry)) {
                _=try ReshapeAuthoringConfiguration.outputLayout(variant:variant,configuration:changed.config,input:input)
            }
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setLinear(let id,let variant,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected linear no longer exists.")}
            let outputs=Set(document.edges.filter {$0.source==id}.map(\.sourcePort))
            let changed=try LinearConfiguration.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setNormalization(let id,let variant,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected normalization no longer exists.")}
            let outputs=Set(document.edges.filter {$0.source==id}.map(\.sourcePort))
            let changed=try NormalizationConfiguration.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setConvolution(let id,let variant,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected convolution no longer exists.")}
            let outputs=Set(document.edges.filter {$0.source==id}.map(\.sourcePort))
            let changed=try ConvolutionConfiguration.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setPooling(let id,let variant,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected pooling component no longer exists.")}
            let outputs=Set(document.edges.filter {$0.source==id}.map(\.sourcePort))
            let changed=try PoolingConfiguration.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:variant,configuration:configuration,connectedOutputPorts:outputs,registry:registry)
            if changed != document.nodes[position] {
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .setActivation(let id,let function,let configuration):
            guard let position=document.nodes.firstIndex(where:{$0.id==id}) else {throw GraphCommandError(.missingNode,"The selected activation no longer exists.")}
            let changed=try ActivationConfiguration.switching(document.nodes[position],configurationVersion:document.schemaVersion,to:function,configuration:configuration,registry:registry)
            if changed != document.nodes[position] {
                // The replacement has already passed its target contract. Migrate
                // other retained nodes without requiring the malformed old target
                // to validate, then restore it at the same source position.
                var retained=document;retained.nodes.remove(at:position)
                document=try GraphMigration.toCurrent(retained,registry:registry)
                document.nodes.insert(changed,at:position)
            }
        case .select(let value,let mode):
            try checkSelection(value)
            switch mode {
            case .replace:selection=value
            case .add:selection.nodeIDs.formUnion(value.nodeIDs);selection.edgeIDs.formUnion(value.edgeIDs)
            case .remove:selection.nodeIDs.subtract(value.nodeIDs);selection.edgeIDs.subtract(value.edgeIDs)
            case .toggle:selection.nodeIDs.formSymmetricDifference(value.nodeIDs);selection.edgeIDs.formSymmetricDifference(value.edgeIDs)
            }
        case .move(let ids,let offset):
            try checkSelection(GraphSelection(nodeIDs:ids))
            for index in document.nodes.indices where ids.contains(document.nodes[index].id) {document.nodes[index]=try moved(document.nodes[index],offset)}
        case .duplicate(let ids,let offset,let semantics):
            guard semantics == .independent else {throw GraphCommandError(.unsupportedSharing,"Graph document v1 cannot represent shared parameter ownership. Independent copies are supported; sharing requires an explicit future document contract.")}
            try checkSelection(GraphSelection(nodeIDs:ids))
            for node in state.document.nodes where ids.contains(node.id) {
                var copy=try moved(node,offset);copy.id=try newIdentity();copies.nodes[node.id]=copy.id
                document.nodes.append(copy)
            }
            for edge in state.document.edges where ids.contains(edge.source) && ids.contains(edge.target) {
                var copy=edge;copy.id=try newIdentity();copy.source=copies.nodes[edge.source]!;copy.target=copies.nodes[edge.target]!
                copies.edges[edge.id]=copy.id;document.edges.append(copy)
            }
            selection=GraphSelection(nodeIDs:Set(copies.nodes.values),edgeIDs:Set(copies.edges.values))
        case .delete(let value):
            try checkSelection(value)
            document.nodes.removeAll {value.nodeIDs.contains($0.id)}
            document.edges.removeAll {value.edgeIDs.contains($0.id) || value.nodeIDs.contains($0.source) || value.nodeIDs.contains($0.target)}
        case .connect(let source,let target):
            let edge=GraphEdge(id:try newIdentity(),source:source.nodeID,sourcePort:source.portID,target:target.nodeID,targetPort:target.portID)
            try checkConnection(edge,in:document,registry:registry,semantics:connectionSemantics)
            document.edges.append(edge);selection=GraphSelection(edgeIDs:[edge.id])
        case .reconnect(let edgeID,let source,let target):
            guard let index=document.edges.firstIndex(where:{$0.id == edgeID}) else {throw GraphCommandError(.missingEdge,"The connection to reconnect no longer exists.")}
            let edge=GraphEdge(id:edgeID,source:source.nodeID,sourcePort:source.portID,target:target.nodeID,targetPort:target.portID)
            var without=document;without.edges.remove(at:index)
            try checkConnection(edge,in:without,registry:registry,semantics:connectionSemantics)
            document.edges[index]=edge;selection=GraphSelection(edgeIDs:[edgeID])
        case .insertOnWire(let edgeID,let node,let inputPort,let outputPort):
            guard let index=document.edges.firstIndex(where:{$0.id == edgeID}) else {throw GraphCommandError(.missingEdge,"The connection to split no longer exists.")}
            guard validNewIdentity(node.id),reserved.insert(node.id).inserted else {throw GraphCommandError(.identityCollision,"The inserted block needs a new valid identity.")}
            try checkInsertedNode(node,registry:registry)
            let original=document.edges[index]
            document.edges.remove(at:index);document.nodes.append(node)
            let upstream=GraphEdge(id:original.id,source:original.source,sourcePort:original.sourcePort,target:node.id,targetPort:inputPort)
            let downstream=GraphEdge(id:try newIdentity(),source:node.id,sourcePort:outputPort,target:original.target,targetPort:original.targetPort)
            try checkConnection(upstream,in:document,registry:registry,semantics:connectionSemantics)
            document.edges.insert(upstream,at:index)
            try checkConnection(downstream,in:document,registry:registry,semantics:connectionSemantics)
            document.edges.append(downstream);selection=GraphSelection(nodeIDs:[node.id])
        }
        if document != state.document && (document.nodes.count > 128 || document.edges.count > 256) {throw GraphCommandError(.resourceLimit,"This POC supports at most 128 nodes and 256 connections.")}
        selection.nodeIDs.formIntersection(Set(document.nodes.map(\.id)))
        selection.edgeIDs.formIntersection(Set(document.edges.map(\.id)))
        let after=GraphEditorState(document:document,selection:selection)
        return GraphCommandResult(before:state,after:after,delta:delta(from:state.document,to:document),copies:copies)
    }
    private static func validNewIdentity(_ id:String)->Bool {!id.isEmpty && !id.contains(".") && id.unicodeScalars.count <= 128}
    private static func checkIdentity(_ document:GraphDocument)throws {
        guard [1,2].contains(document.schemaVersion),Set(document.nodes.map(\.id)).count == document.nodes.count,Set(document.edges.map(\.id)).count == document.edges.count,!document.nodes.contains(where:{$0.id.isEmpty}),!document.edges.contains(where:{$0.id.isEmpty}) else {throw GraphCommandError(.invalidDocument,"Commands require graph schema 1 or 2 and unique nonempty identities. Repair the document before editing.")}
    }
    private static func checkInsertedNode(_ node:GraphNode,registry:ComponentRegistry)throws {
        guard registry.component(type:node.type) != nil else {throw GraphCommandError(.unknownComponent,"The inserted component has no registered implementation.")}
        guard node.x.isFinite,node.y.isFinite else {throw GraphCommandError(.invalidPosition,"The inserted block needs a finite position.")}
        do { _ = try GraphMigration.effectiveConfiguration(of:node,schemaVersion:2,registry:registry) }
        catch {throw GraphCommandError(.invalidConfiguration,error.localizedDescription)}
    }
    private static func checkConnection(_ edge:GraphEdge,in document:GraphDocument,registry:ComponentRegistry,semantics:((GraphEdge,GraphDocument)throws->Void)?)throws {
        guard let source=document.nodes.first(where:{$0.id == edge.source}),let target=document.nodes.first(where:{$0.id == edge.target}) else {throw GraphCommandError(.missingNode,"Both connection endpoints must exist.")}
        guard let sourceType=registry.component(type:source.type),let targetType=registry.component(type:target.type) else {throw GraphCommandError(.unknownComponent,"Both components must have registered port contracts.")}
        let validation=GraphValidator.validate(document,registry:registry)
        let sourceDefinition=validation.configuredDefinition(nodeID:source.id,graph:document,registry:registry) ?? sourceType
        guard sourceDefinition.configuredOutputs(source.config).contains(edge.sourcePort),targetType.configuredInputs(target.config).contains(edge.targetPort) else {throw GraphCommandError(.missingPort,"Connect a declared output port to a declared input port.")}
        guard !document.edges.contains(where:{$0.target == edge.target && $0.targetPort == edge.targetPort}) else {throw GraphCommandError(.occupiedPort,"This input already has a connection. Reconnect that wire explicitly instead of replacing it silently.")}
        var visited=Set<String>(),pending=[edge.target]
        while let id=pending.popLast() {
            if id == edge.source {throw GraphCommandError(.cycle,"This connection would create a cycle. Stateful loops need an explicit future control construct.")}
            guard visited.insert(id).inserted else{continue}
            pending += document.edges.filter {$0.source == id}.map(\.target)
        }
        if let semantics {try semantics(edge,document);return}
        let incomingContract=validation.outputTensorContracts[.init(edge.source,edge.sourcePort)]
        guard let incoming=incomingContract?.layout else{return} // Incomplete drafts keep unresolved shapes visible.
        do {
            if target.type == "linear",let expected=target.config["in_features"]?.integerValue,let output=target.config["out_features"]?.integerValue {
                _ = try incoming.linear(inputFeatures:expected,outputFeatures:output)
            }
            if target.type == "bilinear" {
                let config = try BilinearConfiguration(values: target.config)
                guard incomingContract?.dtype == .float32 else {
                    throw TensorLayout.Failure(message: "Bilinear requires float32 features on both inputs.")
                }
                let expected = edge.targetPort == "a" ? config.input1Features : config.input2Features
                guard incoming.sample.dimensions.last == expected else {
                    throw TensorLayout.Failure(message: "Bilinear input \(edge.targetPort) expects \(expected) features on its final sample axis.")
                }
                if let other = document.edges.first(where: { $0.target == target.id && $0.targetPort != edge.targetPort }),
                   let otherContract = validation.outputTensorContracts[.init(other.source,other.sourcePort)] {
                    guard otherContract.dtype == .float32 else {
                        throw TensorLayout.Failure(message: "Bilinear requires float32 features on both inputs.")
                    }
                    _ = try config.outputLayout(first: edge.targetPort == "a" ? incoming : otherContract.layout,
                                                second: edge.targetPort == "b" ? incoming : otherContract.layout)
                }
            }
            if AttentionConfiguration.componentIDs.contains(target.type) {
                let c=try AttentionConfiguration(componentID:target.type,values:target.config)
                let expected:ContractDType=c.inputDTypes[edge.targetPort]!
                guard incomingContract?.dtype == expected else {throw GraphConfigurationError("This attention port requires \(expected.rawValue) arrays.")}
                let shape=incoming.sample.dimensions
                let padding=edge.targetPort.contains("padding")
                if expected == .int64 {
                    guard shape.count == (padding ? 1:2) else {throw GraphConfigurationError("Mask directions must match the query and source sequence; padding has one source direction.")}
                } else {
                    let ranks=target.type == "attention_mask" ? [2,3]:target.type == "scaled_dot_product_attention" ? [3]:[2]
                    guard ranks.contains(shape.count),shape.allSatisfy({(1...4096).contains($0)}) else {throw GraphConfigurationError("Connect an observation-first sequence with supported directions.")}
                    if target.type == "multihead_attention" {
                        let width=c.values[edge.targetPort == "query" ? "embed_dim":edge.targetPort == "key" ? "kdim":"vdim"]?.integerValue ?? c.values["embed_dim"]!.integerValue!
                        guard shape.last == width else {throw GraphConfigurationError("Connect the declared value count for this attention source.")}
                    } else if let width=c.values["d_model"]?.integerValue {
                        guard shape.last == width else {throw GraphConfigurationError("Connect the declared model value count.")}
                    }
                }
                let candidates=c.configuredInputs.compactMap {port -> TypedTensorContract? in
                    if port == edge.targetPort {return incomingContract}
                    guard let wire=document.edges.first(where:{$0.target == target.id && $0.targetPort == port}) else {return nil}
                    return validation.outputTensorContracts[.init(wire.source,wire.sourcePort)]
                }
                if candidates.count == c.configuredInputs.count {_ = try c.resolved(layouts:candidates.map(\.layout))}
            }
            if RecurrentConfiguration.componentIDs.contains(target.type) {
                let config=try RecurrentConfiguration(componentID:target.type,values:target.config)
                guard incomingContract?.dtype == .float32 else {throw GraphConfigurationError("Memory inputs and states need float32 values.")}
                if edge.targetPort == "in" {
                    guard incoming.sample.dimensions.count == (config.isCell ? 1:2),incoming.sample.dimensions.last == config.inputSize,config.isCell || (1...4096).contains(incoming.sample.dimensions[0]) else {throw GraphConfigurationError("Connect the declared feature width and supported ordered sequence.")}
                } else {
                    let expected=config.isCell ? [config.hiddenSize]:[config.layerCount*config.directionCount,edge.targetPort == "cell" ? config.hiddenSize:config.stateWidth]
                    guard incoming.sample.dimensions == expected else {throw GraphConfigurationError("Connect observation-first state with the declared layers, directions and width.")}
                }
            }
            if TensorRoutingConfiguration.componentIDs.contains(target.type) {
                let values=try GraphMigration.effectiveConfiguration(of:target,schemaVersion:document.schemaVersion,registry:registry)
                let routing=try TensorRoutingConfiguration(componentID:target.type,values:values)
                let expected:ContractDType=target.type == "gather" && edge.targetPort == "indices" ? .int64:.float32
                guard incomingContract?.dtype == expected else {throw TensorLayout.Failure(message:"This routing input requires \(expected.rawValue) arrays.")}
                let candidates=routing.configuredInputs.compactMap {port -> TypedTensorContract? in
                    if port == edge.targetPort {return incomingContract}
                    guard let wire=document.edges.first(where:{$0.target == target.id && $0.targetPort == port}) else {return nil}
                    return validation.outputTensorContracts[.init(wire.source,wire.sourcePort)]
                }
                if candidates.count == routing.configuredInputs.count {_ = try routing.resolved(inputs:candidates)}
            }
            if target.type == "output",incoming.sample.dimensions != [2] {
                throw TensorLayout.Failure(message:"Classification Output requires [N, 2]; the source supplies [\(incoming.displayDimensions.joined(separator:", "))].")
            }
            if ["reshape","transpose","flatten"].contains(target.type) {
                _ = try TensorTransformConfiguration(operation:target.type,values:target.config).outputLayout(for:incoming)
            }
            if ["add","concat"].contains(target.type),let other=document.edges.first(where:{$0.target == target.id && $0.targetPort != edge.targetPort}),let otherShape=validation.outputTensorContracts[.init(other.source,other.sourcePort)]?.layout {
                if target.type == "add" {_ = try incoming.adding(otherShape)}
                else {_ = try incoming.concatenated(with:otherShape,axis:-1)}
            }
        } catch {throw GraphCommandError(.incompatibleShape,error.localizedDescription)}
    }
    private static func delta(from before:GraphDocument,to after:GraphDocument)->GraphCommandDelta {
        let oldNodes=Dictionary(uniqueKeysWithValues:before.nodes.map {($0.id,$0)}),newNodes=Dictionary(uniqueKeysWithValues:after.nodes.map {($0.id,$0)})
        let oldEdges=Dictionary(uniqueKeysWithValues:before.edges.map {($0.id,$0)}),newEdges=Dictionary(uniqueKeysWithValues:after.edges.map {($0.id,$0)})
        let a=Set(oldNodes.keys),b=Set(newNodes.keys),c=Set(oldEdges.keys),d=Set(newEdges.keys)
        return GraphCommandDelta(insertedNodeIDs:b.subtracting(a),removedNodeIDs:a.subtracting(b),updatedNodeIDs:Set(a.intersection(b).filter {oldNodes[$0] != newNodes[$0]}),insertedEdgeIDs:d.subtracting(c),removedEdgeIDs:c.subtracting(d),updatedEdgeIDs:Set(c.intersection(d).filter {oldEdges[$0] != newEdges[$0]}))
    }
}
