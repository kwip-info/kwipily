import Foundation

/// Current tensor editing surfaces retain the original executable primitive identities.
public enum TensorOperationAuthoring {
    public enum Family:String,CaseIterable {case axes,parts,gather,einsum,matmul,unfold,fold,concatenation,addition,stacking
        public var title:String {switch self {case .stacking:return "Stack";case .addition:return "Add";case .concatenation:return "Concatenate";case .axes:return "Transpose";case .parts:return "Split tensor";case .gather:return "Gather";case .einsum:return "Einsum";case .matmul:return "Matrix multiply";case .unfold:return "Unfold";case .fold:return "Fold"}}
        public var types:[String] {switch self {case .stacking:return ["stack"];case .addition:return ["add"];case .concatenation:return ["concat"];case .axes:return ["transpose","permute"];case .parts:return ["split","chunk"];default:return [rawValue]}}
    }
    public static let componentIDs=Set(Family.allCases.flatMap(\.types))
    public static func family(_ type:String)->Family? {Family.allCases.first {$0.types.contains(type)}}
    public static func methodTitle(_ type:String)->String {switch type {case "transpose":return "Swap axes";case "permute":return "Reorder axes";case "split":return "Split sizes";case "chunk":return "Request chunks";default:return family(type)?.title ?? type}}
    public static func canonical(_ type:String,_ values:[String:JSONValue])throws->[String:JSONValue] {
        switch type {
        case "transpose":return try TensorTransformConfiguration(operation:type,values:values).values
        case "add","concat","stack","permute":return try TensorCompositionConfiguration(componentID:type,values:values).values
        case "split","chunk","gather","einsum":return try TensorRoutingConfiguration(componentID:type,values:values).values
        case "unfold","fold":return try SpatialPatchConfiguration(componentID:type,values:values).values
        case "matmul":guard values.isEmpty else {throw GraphConfigurationError("Matrix multiply has no configuration fields. Prepare operands explicitly upstream.")};return [:]
        default:throw GraphConfigurationError("Choose a supported tensor operation.")
        }
    }
    public static func defaults(_ type:String,registry:ComponentRegistry)throws->[String:JSONValue] {
        guard let definition=registry.component(type:type) else {throw GraphConfigurationError("This tensor method is unavailable in this editor.")}
        return try canonical(type,Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)}))
    }
    public static func inputPorts(_ type:String,_ values:[String:JSONValue])throws->[String] {
        if TensorRoutingConfiguration.componentIDs.contains(type) {return try TensorRoutingConfiguration(componentID:type,values:values).configuredInputs}
        return type == "matmul" || type == "concat" || type == "add" || type == "stack" ? ["a","b"]:["in"]
    }
    public static func outputs(type:String,values:[String:JSONValue],inputs:[String:BatchedTensorLayout])throws->[String:BatchedTensorLayout] {
        let values=try canonical(type,values),ports=try inputPorts(type,values)
        guard ports.allSatisfy({inputs[$0] != nil}) else {throw GraphConfigurationError("Connect each configured input to resolve the planned shape.")}
        let layouts=ports.map {inputs[$0]!}
        switch type {
        case "transpose":return ["out":try TensorTransformConfiguration(operation:type,values:values).outputLayout(for:layouts[0])]
        case "add","concat","stack","permute":return ["out":try TensorCompositionConfiguration(componentID:type,values:values).resolved(layouts:layouts)]
        case "split","chunk","gather","einsum":return try TensorRoutingConfiguration(componentID:type,values:values).resolved(layouts:layouts)
        case "unfold","fold":return ["out":try SpatialPatchConfiguration(componentID:type,values:values).resolved(layout:layouts[0])]
        default:return ["out":try layouts[0].matrixMultiplied(by:layouts[1])]
        }
    }
    public static func incomingContracts(nodeID:String,graph:GraphDocument,validation:ValidationResult)->[String:TypedTensorContract] {
        let grouped=Dictionary(grouping:graph.edges.filter {$0.target==nodeID},by:\.targetPort)
        return grouped.compactMapValues {edges in
            guard edges.count==1,let edge=edges.first else {return nil}
            return validation.outputTensorContracts[.init(edge.source,edge.sourcePort)] ?? validation.tensorContracts[edge.source]
        }
    }
    /// Only semantic edits use this inference path; presentation uses cached projections.
    public static func incomingContracts(address:WorkflowNodeAddress,document:WorkflowDocument,registry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)->[String:TypedTensorContract] {
        let path:[String],localID:String
        if let child=try? WorkflowInlineContainer.parse(in:document,address:address) {path=[child.root.localID];localID=child.childID}
        else {path=[];localID=address.localID}
        var candidates:[String:[TypedTensorContract?]]=[:]
        for definition in document.nodes {
            guard WorkflowModelBoundary.isVisibleDefinition(definition,in:document),
                  let snapshot=try? WorkflowCompiler.modelSnapshot(document,definition:definition.address,modelRegistry:registry,dataRegistry:dataRegistry) else {continue}
            let projection=ModelTensorProjection(snapshot:snapshot,registry:registry)
            for port in ["in","source","indices","a","b"] where projection.containsInput(instancePath:path,nodeID:localID,port:port) {
                candidates[port,default:[]].append(projection.inputContract(instancePath:path,nodeID:localID,port:port))
            }
        }
        return candidates.compactMapValues {values in guard let first=values.first ?? nil,values.allSatisfy({$0==first}) else {return nil};return first}
    }
    public static func incoming(nodeID:String,graph:GraphDocument,validation:ValidationResult)->[String:BatchedTensorLayout] {
        incomingContracts(nodeID:nodeID,graph:graph,validation:validation).mapValues(\.layout)
    }
    public static func incoming(address:WorkflowNodeAddress,document:WorkflowDocument,registry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)->[String:BatchedTensorLayout] {
        incomingContracts(address:address,document:document,registry:registry,dataRegistry:dataRegistry).mapValues(\.layout)
    }
    public static func validateKnownTypes(type:String,values:[String:JSONValue],inputs:[String:TypedTensorContract])throws {
        guard ["add","stack"].contains(type),let a=inputs["a"],let b=inputs["b"] else {return}
        _=try TensorCompositionConfiguration(componentID:type,values:values).resolved(inputs:[a,b])
    }
    public static func switching(_ node:GraphNode,configurationVersion:Int,to type:String,configuration:[String:JSONValue],connectedInputs:Set<String>,connectedOutputs:Set<String>,inputs:[String:BatchedTensorLayout],registry:ComponentRegistry,inputContracts:[String:TypedTensorContract]=[:])throws->GraphNode {
        guard let current=family(node.type),current==family(type),registry.component(type:type) != nil else {throw GraphConfigurationError("Choose a method in the same available tensor family.")}
        guard node.parameterSettings?.isEmpty != false else {throw GraphConfigurationError("Tensor operations have no learned parameters. Repair unexpected parameter policies before changing this component.")}
        let values=try canonical(type,configuration)
        let old=try canonical(node.type,GraphMigration.effectiveConfiguration(of:node,schemaVersion:configurationVersion,registry:registry))
        if node.type==type,old==values {return node}
        try validateKnownTypes(type:type,values:values,inputs:inputContracts)
        let required=try inputPorts(type,values)
        guard connectedInputs.isSubset(of:Set(required)) else {throw GraphConfigurationError("Disconnect the input removed by this arrangement before applying it.")}
        let available:Set<String>
        if required.allSatisfy({inputs[$0] != nil}) {available=Set(try outputs(type:type,values:values,inputs:inputs).keys)}
        else if current == .parts {
            guard connectedOutputs.isEmpty else {throw GraphConfigurationError("Resolve the input shape before changing connected parts; their availability cannot yet be verified.")}
            available=Set((0..<TensorRoutingConfiguration.maximumOutputs).map {"part\($0)"})
        } else {available=["out"]}
        guard connectedOutputs.isSubset(of:available) else {throw GraphConfigurationError("Disconnect the part outputs removed by these settings before applying them.")}
        var next=node;next.type=type;next.config=values;return next
    }
    public static func switching(_ node:WorkflowNode,to type:String,configuration:[String:JSONValue],connectedInputs:Set<String>,connectedOutputs:Set<String>,inputs:[String:BatchedTensorLayout],registry:ComponentRegistry,inputContracts:[String:TypedTensorContract]=[:])throws->WorkflowNode {
        guard case .modelPrimitive(let original,let version,let values)=node.payload else {throw GraphConfigurationError("Choose an existing tensor primitive.")}
        let source=GraphNode(id:node.address.localID,type:original,name:node.name,x:node.x,y:node.y,config:values,parameterSettings:node.parameterSettings)
        let changed=try switching(source,configurationVersion:version,to:type,configuration:configuration,connectedInputs:connectedInputs,connectedOutputs:connectedOutputs,inputs:inputs,registry:registry,inputContracts:inputContracts)
        guard changed != source else {return node}
        var next=node;next.payload = .modelPrimitive(type:type,configurationVersion:original==type ? version:2,config:changed.config);return next
    }
}
