import Foundation

public enum BuiltInCapabilityBridgeError:LocalizedError {
    case incompatibleRegistry(String)
    public var errorDescription:String? {switch self {case .incompatibleRegistry(let message):return message}}
}

/// Additive projection of the implemented catalog; never rewrites a user's graph.
public enum BuiltInCapabilityBridge {
    public static let packageID="kwip.builtin.pytorch"
    /// Static implemented declarations; these do not assert a discovered or available runtime.
    public static func declarations(registry:ComponentRegistry)throws->[CapabilityDescriptor] {
        guard registry.schemaVersion == 2 else {throw BuiltInCapabilityBridgeError.incompatibleRegistry("Registry schema 2 is required.")}
        let types:Set<String>=["input","linear","relu","tanh","gelu","softmax","concat","add","output","reshape","transpose","flatten"]
        let registered = Set(registry.components.map(\.id))
        guard types.isSubset(of: registered), registered.isSubset(of: types.union(["mixture_experts","layer_norm","index_input","embedding","matmul","bilinear","lazy_linear","conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d","depthwise_conv2d"]).union(BundledEmbeddingComponents.declarations.keys).union(BundledSpatialPatchComponents.declarations.keys).union(BundledTensorRoutingComponents.declarations.keys).union(BundledTensorCompositionComponents.declarations.keys).union(BundledPoolingComponents.declarations.keys).union(BundledActivationComponents.declarations.keys).union(BundledNormalizationComponents.declarations.keys).union(BundledOptimizerComponents.declarations.keys).union(BundledSchedulerComponents.declarations.keys).union(BundledRecurrentComponents.declarations.keys).union(BundledAttentionComponents.declarations.keys).union(BundledLossComponents.declarations.keys).union(BundledInitializationComponents.declarations.keys).union(BundledGradientControlComponents.declarations.keys).union(BundledDataLoadingComponents.declarations.keys)), registry.components.count == registered.count else {throw BuiltInCapabilityBridgeError.incompatibleRegistry("The bridge requires the reviewed implemented primitives. New operators need an explicit adapter contract.")}
        let capabilities=try registry.components.map { definition->CapabilityDescriptor in
            if let declaration=BundledDataLoadingComponents.declarations[definition.id] {return try declaration(definition)}
            if let declaration=BundledGradientControlComponents.declarations[definition.id] {return try declaration(definition)}
            if let declaration=BundledInitializationComponents.declarations[definition.id] {return try declaration(definition)}
            if let declaration=BundledLossComponents.declarations[definition.id] {return try declaration(definition)}
            if let declaration=BundledAttentionComponents.declarations[definition.id] {return try declaration(definition)}
            if let declaration = BundledSpatialPatchComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledTensorRoutingComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledTensorCompositionComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledTensorLayoutComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledLayerComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledActivationComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledRecurrentComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledSchedulerComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledOptimizerComponents.declarations[definition.id] { return try declaration(definition) }
            if let declaration = BundledNormalizationComponents.declarations[definition.id] { return try declaration(definition) }
            if definition.id == "matmul" {return try matrixDeclaration(definition)}
            if definition.id == "conv1d" {return try conv1dDeclaration(definition)}
            if definition.id == "conv2d" {return try conv2dDeclaration(definition)}
            if let declaration = BundledPoolingComponents.declarations[definition.id] { return try declaration(definition) }
            if definition.id == "depthwise_conv2d" {return try depthwiseConv2dDeclaration(definition)}
            if definition.id == "conv_transpose3d" {return try convTranspose3dDeclaration(definition)}
            if definition.id == "conv_transpose2d" {return try convTranspose2dDeclaration(definition)}
            if definition.id == "conv_transpose1d" {return try convTranspose1dDeclaration(definition)}
            if definition.id == "conv3d" {return try conv3dDeclaration(definition)}
            if definition.id == "lazy_linear" {return try lazyLinearDeclaration(definition)}
            if definition.id == "mixture_experts" {return try mixtureDeclaration(definition)}
            if definition.id == "bilinear" {return try bilinearDeclaration(definition)}
            if let declaration=BundledEmbeddingComponents.declarations[definition.id] {return try declaration(definition)}
            let expectedInputs:[String]=definition.id == "input" ? [] : ["concat","add"].contains(definition.id) ? ["a","b"] : ["in"]
            let expectedOutputs=definition.id == "output" ? [] : ["out"]
            let expectedFields:Set<String>=definition.id == "input" ? ["features"] : definition.id == "linear" ? ["in_features","out_features","bias"] : definition.id == "gelu" ? ["approximate_tanh"] : definition.id == "layer_norm" ? Set(LayerNormComponent.definition.fields.map(\.key)) : definition.id == "softmax" ? ["dim"] : definition.id == "reshape" ? ["shape"] : definition.id == "transpose" ? ["dim0","dim1"] : definition.id == "flatten" ? ["start_dim","end_dim"] : []
            guard definition.inputs == expectedInputs,definition.outputs == expectedOutputs,Set(definition.fields.map(\.key)) == expectedFields,definition.fields.count == expectedFields.count,definition.fields.allSatisfy({$0.isValidDefinition && (definition.id == "layer_norm" ? LayerNormComponent.definition.fields.contains($0) : ["reshape","transpose","flatten"].contains(definition.id) ? TensorLearningCanvas.registry.component(type:definition.id)?.fields.contains($0) == true : (definition.id == "softmax" ? $0.kind == .integer && $0.min == -8 && $0.max == 7 && $0.defaultValue == .integer(-1) : $0.key == "bias" ? $0.kind == .boolean && $0.defaultValue == .bool(true) : $0.key == "approximate_tanh" ? $0.kind == .boolean && $0.defaultValue == .bool(false) : $0.kind == .integer && $0.min == 1 && $0.max == 512))}) else {throw BuiltInCapabilityBridgeError.incompatibleRegistry("\(definition.id) differs from its implemented v1 port/configuration contract.")}
            func width(_ port:String,_ direction:PortDirection)->DimensionContract {
                switch definition.id {
                case "input":return .parameter("features")
                case "linear":return .parameter(direction == .input ? "in_features" : "out_features")
                case "output":return .fixed(2)
                case "concat":return .symbol(direction == .output ? "mergedFeatures" : port+"Features")
                default:return .symbol("features")
                }
            }
            func port(_ id:String,_ direction:PortDirection)->PortContract {
                let tensor:TensorContract
                if ["input","output"].contains(definition.id) {
                    tensor=TensorContract([.symbol("batch"),width(id,direction)])
                } else if definition.id == "linear" {
                    tensor=TensorContract([width(id,direction)],leadingAxes:.init(minimum:1,maximum:7))
                } else {
                    tensor=TensorContract([],leadingAxes:.init(minimum:2,maximum:8))
                }
                return PortContract(id,direction:direction,schema:.tensor(tensor),phases:[.forward],maximumConnections:direction == .input ? 1 : nil)
            }
            let ports=definition.inputs.map {port($0,.input)}+definition.outputs.map {port($0,.output)}
            let state:[StateDeclaration]=["linear", "layer_norm"].contains(definition.id) ? [StateDeclaration("parameters",kind:.parameters)] : []
            let concept:[String]
            switch definition.id {
            case "reshape","transpose","flatten":concept=["linear-algebra.tensor-coordinates","linear-algebra.permutation-gradients"]
            case "linear":concept=["linear-algebra.affine-map","linear-algebra.matrix-dimensions"]
            case "concat":concept=["linear-algebra.feature-concatenation"]
            case "add":concept=["linear-algebra.elementwise-addition"]
            case "relu","tanh":concept=["neural-networks.elementwise-nonlinearity"]
            case "layer_norm":concept=["linear-algebra.normalization-axes","linear-algebra.population-variance","neural-networks.learned-scale-and-bias"]
            case "softmax":concept=["linear-algebra.normalization-axes","linear-algebra.softmax-jacobian"]
            case "gelu":concept=["neural-networks.elementwise-nonlinearity","neural-networks.gelu"]
            default:concept=["linear-algebra.batch-and-feature-axes"]
            }
            return CapabilityDescriptor(id:definition.id,version:definition.id == "linear" ? ContractVersion(1,2,0) : ["relu","tanh","gelu","concat","add"].contains(definition.id) ? ContractVersion(1,1,0) : ContractVersion(),packageID:packageID,backend:.pytorch,ports:ports,fields:definition.fields,lifecycle:LifecycleContract(phases:[.initialize,.forward,.export],state:state),devices:[.cpu,.mps],availability:.existingPOCAdapter,adapterID:"kwip.runner.graph-v2.\(definition.id)",learning:LearningContract(conceptIDs:concept,lessonIDs:definition.id == "layer_norm" ? LayerNormLearningTask.allCases.map {"assessment."+$0.rawValue} : definition.id == "softmax" ? ["assessment.softmax-last-axis","assessment.softmax-first-sample-axis"] : ["reshape","transpose","flatten"].contains(definition.id) ? ["assessment.tensor-"+definition.id] : definition.id == "gelu" ? [ModelConstructionLessonIdentity.geluExact.assessmentID,ModelConstructionLessonIdentity.geluTanh.assessmentID] : ["kwip.course.parallel-classifier.v1"],exampleIDs:["reshape","transpose","flatten"].contains(definition.id) ? ["kwip.example.tensor-layout-classifier.v1"] : definition.id == "gelu" ? ["kwip.example.gelu-exact-classifier.v1","kwip.example.gelu-tanh-classifier.v1"] : definition.id == "softmax" ? ["kwip.example.softmax-classifier.v1"] : definition.id == "layer_norm" ? ["kwip.example.layer-norm-rows-classifier.v1","kwip.example.layer-norm-sample-classifier.v1"] : ["kwip.example.parallel-classifier.v1"]),shapeRuleID:"kwip.shape.\(definition.id).v1")
        }
        return capabilities
    }
    public static func package(registry:ComponentRegistry,runtimeVersion:String)throws->CapabilityPackageDescriptor {
        guard !runtimeVersion.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {throw BuiltInCapabilityBridgeError.incompatibleRegistry("An exact PyTorch runtime pin is required.")}
        let capabilities=try declarations(registry:registry)
        let result=CapabilityPackageDescriptor(id:packageID,backendVersions:["pytorch":runtimeVersion],capabilities:capabilities)
        let validation=CapabilityContractValidator.validate(result)
        guard validation.isValid else {throw BuiltInCapabilityBridgeError.incompatibleRegistry(validation.issues.map(\.message).joined(separator:" "))}
        return result
    }
    public static func validate(graph:GraphDocument,registry:ComponentRegistry,runtimeVersion:String,context:GraphValidationContext = .bundledDataset)->ContractValidation {
        do {
            let package=try package(registry:registry,runtimeVersion:runtimeVersion)
            let legacy=GraphValidator.validate(graph,registry:registry,context:context)
            var issues=legacy.issues.map {ContractIssue("legacy.graph",path:$0.nodeID ?? "graph",message:$0.message)}
            let descriptors=Dictionary(uniqueKeysWithValues:package.capabilities.map {($0.id,$0)})
            // Duplicate node identity is already an error; never trap on a malformed draft.
            var nodes:[String:GraphNode]=[:];for node in graph.nodes {nodes[node.id]=node}
            func specialized(_ descriptor:CapabilityDescriptor,node:GraphNode)->CapabilityDescriptor {
                var result=descriptor
                for index in result.ports.indices {
                    let port=result.ports[index]
                    let address:GraphPortAddress?
                    if port.direction == .output {address = .init(node.id,port.id)}
                    else {address = graph.edges.first {$0.target == node.id && $0.targetPort == port.id}.map { .init($0.source,$0.sourcePort) }}
                    if let address,let packet=legacy.outputBagContracts[address],var tensor=port.schema.tensor {
                        // Typed packet validation has already established role,
                        // coherent source and configured layout. Resolve physical
                        // coordinates without inventing a dense sample axis.
                        tensor.leadingAxes=nil
                        if packet.role == .offsets {tensor.dimensions=[.symbol("bagBoundaries")]}
                        else if packet.configuration.layout == "packed" {tensor.dimensions=[.symbol("bagItems")]}
                        else {tensor.dimensions=[.symbol("bags"),.fixed(packet.configuration.maximumItemsPerBag)]}
                        result.ports[index].schema = .tensor(tensor)
                        continue
                    }
                    guard let address,let contract=legacy.outputTensorContracts[address],var tensor=port.schema.tensor else {continue}
                    // Only polymorphic layout/merge inputs adopt the supplied dtype.
                    // Fixed consumers retain their requirement, so invalid casts remain errors.
                    if port.direction == .output || ["reshape","transpose","flatten","concat","add","stack","permute"].contains(node.type) {
                        tensor.dtype=contract.dtype
                    }
                    tensor.leadingAxes=nil
                    tensor.dimensions=[.symbol("batch")]+contract.layout.sample.dimensions.map { .fixed($0) }
                    result.ports[index].schema = .tensor(tensor)
                }
                return result
            }
            for edge in graph.edges {
                guard let source=nodes[edge.source],let target=nodes[edge.target],let a=descriptors[source.type],let b=descriptors[target.type] else{continue}
                let result=CapabilityContractValidator.connection(from:ContractEndpoint(specialized(a,node:source),instanceID:source.id,portID:edge.sourcePort,configuration:source.config),to:ContractEndpoint(specialized(b,node:target),instanceID:target.id,portID:edge.targetPort,configuration:target.config),phase:.forward)
                issues += result.issues.map {ContractIssue($0.code,path:edge.id+"."+$0.path,message:$0.message,severity:$0.severity)}
            }
            return ContractValidation(issues)
        } catch {return ContractValidation([ContractIssue("bridge.registry",path:"registry",message:error.localizedDescription)])}
    }
}
