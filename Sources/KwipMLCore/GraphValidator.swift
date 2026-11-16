import Foundation

/// Selects the input boundary without relaxing graph shape/type or binary-output checks.
/// Prepared workflow architecture is validated before its ordered column mapping and
/// actual retained preparation schema are checked; this is not source-data approval.
public enum GraphValidationContext: Equatable {
    case bundledDataset
    case preparedWorkflow
    case policyValue
    case frozenPolicy
    case explicitObjective
}

public enum GraphValidator {
    public static func validate(_ graph: GraphDocument, registry: ComponentRegistry, context: GraphValidationContext = .bundledDataset) -> ValidationResult {
        var result = ValidationResult()
        func issue(_ message: String, _ nodeID: String? = nil) {
            result.issues.append(ValidationIssue(id: "issue-\(result.issues.count)", message: message, nodeID: nodeID))
        }
        if ![1,2].contains(graph.schemaVersion) { issue("Unsupported graph version \(graph.schemaVersion). Versions 1 and 2 are supported.") }
        if !(2...128).contains(graph.nodes.count) { issue("Graph requires 2–128 nodes.") }
        if graph.edges.count > 256 { issue("Graph supports at most 256 connections.") }
        var nodes: [String: GraphNode] = [:]
        for node in graph.nodes {
            if node.id.isEmpty { issue("Every component needs a nonempty identity.", node.id) }
            if node.id.contains(".") || node.id.unicodeScalars.count > 128 { issue("Component IDs must be without dots and at most 128 characters.", node.id) }
            if nodes[node.id] != nil { issue("Duplicate component identity: \(node.id).", node.id) }
            nodes[node.id] = node
            if !node.x.isFinite || !node.y.isFinite { issue("Component position must be finite.", node.id) }
            guard registry.component(type: node.type) != nil else { issue("Unknown component type: \(node.type).", node.id); continue }
            do { _ = try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry) }
            catch { issue(error.localizedDescription,node.id) }
        }
        let inputs = graph.nodes.filter { ["input","index_input","bag_input"].contains($0.type) }
        let outputs = graph.nodes.filter { $0.type == "output" }
        if let names=graph.namedInputs {
            if graph.schemaVersion != 2 || !(1...8).contains(names.count) || Set(names.values).count != names.count || Set(names.values) != Set(inputs.map(\.id)) || names.keys.contains(where:{$0.range(of:"^[A-Za-z][A-Za-z0-9_]{0,63}$",options:.regularExpression)==nil}) || inputs.contains(where:{$0.type=="bag_input"}) {issue("Named inputs must declare 1–8 distinct tensor boundaries with unique semantic names.")}
            if context == .bundledDataset {issue("Named inputs require prepared table mapping.")}
        } else if inputs.count != 1 { issue("This flow needs exactly one Input component.") }
        if outputs.count != 1 { issue("This flow needs exactly one Model Output component.") }
        var edgeIDs = Set<String>()
        var incoming: [String: [GraphEdge]] = [:]
        var outgoing: [String: [GraphEdge]] = [:]
        for edge in graph.edges {
            if edge.id.isEmpty || !edgeIDs.insert(edge.id).inserted { issue("Connection identities must be nonempty and unique.") }
            guard let source = nodes[edge.source], let target = nodes[edge.target] else { issue("Connection refers to a missing component."); continue }
            guard registry.component(type: source.type)?.outputs.contains(edge.sourcePort) == true else { issue("Source port \(edge.sourcePort) does not exist.", source.id); continue }
            guard registry.component(type: target.type)?.configuredInputs(target.config).contains(edge.targetPort) == true else { issue("Target port \(edge.targetPort) does not exist.", target.id); continue }
            if source.type == "bag_input",registry.component(type:source.type)?.configuredOutputs(source.config).contains(edge.sourcePort) != true {
                issue("Output \(edge.sourcePort) is unavailable for this bag arrangement. Reconnect this saved wire to an enabled output.",source.id)
            }
            incoming[target.id, default: []].append(edge)
            outgoing[source.id, default: []].append(edge)
        }
        for node in graph.nodes {
            guard let definition = registry.component(type: node.type) else { continue }
            for port in definition.configuredInputs(node.config) {
                let count = incoming[node.id, default: []].filter { $0.targetPort == port }.count
                if count != 1 { issue(count == 0 ? "Connect input \(port) on \(node.name.isEmpty ? definition.title : node.name)." : "Input \(port) accepts one connection; remove the extra wire.", node.id) }
            }
            for edge in incoming[node.id,default:[]] where !definition.configuredInputs(node.config).contains(edge.targetPort) {
                issue("Input \(edge.targetPort) is unavailable for this arrangement. Reconnect this saved wire to an enabled input.",node.id)
            }
        }
        var indegree = nodes.mapValues { incoming[$0.id, default: []].count }
        var queue = graph.nodes.filter { indegree[$0.id] == 0 }.map(\.id)
        var visited = Set<String>()
        while !queue.isEmpty {
            let id = queue.removeFirst()
            guard visited.insert(id).inserted else { continue }
            result.orderedNodeIDs.append(id)
            for edge in outgoing[id, default: []] {
                indegree[edge.target, default: 0] -= 1
                if indegree[edge.target] == 0 { queue.append(edge.target) }
            }
        }
        if visited.count != nodes.count { issue("The graph contains a cycle. Remove the backward connection.") }
        // Every node must participate in the selected input-to-output computation.
        func reachable(_ start: String, forward: Bool) -> Set<String> {
            var seen = Set<String>(); var pending = [start]
            while let id = pending.popLast() {
                guard seen.insert(id).inserted else { continue }
                pending += (forward ? outgoing[id, default: []].map(\.target) : incoming[id, default: []].map(\.source))
            }
            return seen
        }
        if !inputs.isEmpty, outputs.count == 1 {
            let active = inputs.reduce(into:Set<String>()) {$0.formUnion(reachable($1.id,forward:true))}.intersection(reachable(outputs[0].id, forward: false))
            for node in graph.nodes where !active.contains(node.id) { issue("Connect this component to the complete Input → Model Output flow, or remove it.", node.id) }
        }
        func dimension(_ node: GraphNode, _ key: String) -> Int? {
            guard let value = node.config[key]?.integerValue, (1...512).contains(value) else { return nil }
            return value
        }
        var parameters = 0, trainableParameters = 0
        for id in result.orderedNodeIDs {
            guard let node = nodes[id] else { continue }
            func contract(_ port:String)->TypedTensorContract? {
                let matches=incoming[id,default:[]].filter {$0.targetPort == port}
                guard matches.count == 1 else {return nil}
                let edge = matches[0]
                guard let value = result.outputTensorContracts[.init(edge.source, edge.sourcePort)] else {
                    issue("Input \(port) has no resolved tensor contract at \(edge.source).\(edge.sourcePort).", id)
                    return nil
                }
                return value
            }
            func layout(_ port:String)->BatchedTensorLayout? {contract(port)?.layout}
            do {
                if node.type != "embedding_bag",incoming[id,default:[]].contains(where:{nodes[$0.source]?.type == "bag_input"}) {
                    throw GraphConfigurationError("Bag coordinates require the matching EmbeddingBag input ports. They are not ordinary tensor samples; reduce the bag before using this component.")
                }
                if (["linear","lazy_linear","conv1d","conv2d","conv3d","conv_transpose1d","conv_transpose2d","conv_transpose3d","depthwise_conv2d","avg_pool1d","avg_pool2d","max_pool1d","max_pool2d","max_pool3d","relu","tanh","gelu","sigmoid","silu","leaky_relu","elu","softmax","layer_norm","rms_norm"].contains(node.type) || AdaptivePoolingConfiguration.componentIDs.contains(node.type) || SpatialPatchConfiguration.componentIDs.contains(node.type)),let input=contract("in"),input.dtype != .float32 {
                    throw GraphConfigurationError("This component requires float32 features; integer indices need an embedding or explicit conversion.")
                }
                switch node.type {
                case "avg_pool2d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let resolved = try AvgPool2dConfiguration(values: values).resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = .init(layout: resolved.outputLayout, dtype: .float32)
                    }
                case "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool2d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let resolved = try AdaptivePoolingConfiguration(componentID: node.type, values: values).resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = .init(layout: resolved.outputLayout, dtype: .float32)
                        if let indices = resolved.indicesLayout { result.outputTensorContracts[.init(id,"indices")] = .init(layout: indices, dtype: .int64) }
                    }
                case "avg_pool1d":
                    if let input=layout("in") {
                        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        let resolved=try AvgPool1dConfiguration(values:values).resolved(input:input)
                        result.outputTensorContracts[.init(id,"out")] = .init(layout:resolved.outputLayout,dtype:.float32)
                    }
                case "max_pool1d":
                    if let input=layout("in") {
                        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        let resolved=try MaxPool1dConfiguration(values:values).resolved(input:input)
                        result.outputTensorContracts[.init(id,"out")] = .init(layout:resolved.outputLayout,dtype:.float32)
                        if let indices=resolved.indicesLayout {
                            result.outputTensorContracts[.init(id,"indices")] = .init(layout:indices,dtype:.int64)
                        }
                    }
                case "max_pool2d":
                    if let input=layout("in") {
                        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        let resolved=try MaxPool2dConfiguration(values:values).resolved(input:input)
                        result.outputTensorContracts[.init(id,"out")] = .init(layout:resolved.outputLayout,dtype:.float32)
                        if let indices=resolved.indicesLayout {
                            result.outputTensorContracts[.init(id,"indices")] = .init(layout:indices,dtype:.int64)
                        }
                    }
                case "max_pool3d":
                    if let input=layout("in") {
                        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        let resolved=try MaxPool3dConfiguration(values:values).resolved(input:input)
                        result.outputTensorContracts[.init(id,"out")] = .init(layout:resolved.outputLayout,dtype:.float32)
                        if let indices=resolved.indicesLayout {
                            result.outputTensorContracts[.init(id,"indices")] = .init(layout:indices,dtype:.int64)
                        }
                    }
                case "bag_input":
                    let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                    let configuration=try BagSourceConfiguration(values:values)
                    for port in configuration.outputPorts {
                        guard let role=BagPortContract.Role(rawValue:port) else {continue}
                        result.outputBagContracts[.init(id,port)] = .init(sourceNodeID:id,configuration:configuration,role:role)
                    }
                case "embedding_bag":
                    let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                    let configuration=try EmbeddingBagConfiguration(values:values)
                    var coordinates:[BagPortContract]=[]
                    for port in configuration.inputPorts {
                        let edges=incoming[id,default:[]].filter {$0.targetPort == port}
                        guard edges.count == 1,let edge=edges.first,
                              let coordinate=result.outputBagContracts[.init(edge.source,edge.sourcePort)],
                              coordinate.role.rawValue == port else {
                            throw GraphConfigurationError("Connect EmbeddingBag \(port) to the matching coordinate output of one Bag input.")
                        }
                        coordinates.append(coordinate)
                    }
                    guard let first=coordinates.first,
                          coordinates.allSatisfy({$0.sourceNodeID == first.sourceNodeID && $0.configuration == first.configuration}),
                          first.configuration.layout == configuration.layout,
                          first.configuration.usesPerSampleWeights == configuration.usesPerSampleWeights,
                          first.configuration.includeLastOffset == configuration.includeLastOffset else {
                        throw GraphConfigurationError("EmbeddingBag coordinates must come from the same Bag input with matching layout, item-weight and terminal-offset controls.")
                    }
                    result.outputTensorContracts[.init(id,"out")]=try .init(sample:[configuration.embedding.embeddingDimension],dtype:.float32)
                    parameters += configuration.parameterCount
                    if !configuration.embedding.frozen {trainableParameters += configuration.parameterCount}
                case "index_input":
                    result.outputTensorContracts[.init(id,"out")]=try IndexInputConfiguration(values:node.config).tensor
                case "embedding":
                    if let input=contract("in") {
                        let config=try EmbeddingConfiguration(values:node.config)
                        guard !config.sparse || !config.scaleGradientByFrequency else {throw GraphConfigurationError("Sparse embedding gradients do not support frequency scaling.")}
                        result.outputTensorContracts[.init(id,"out")]=try input.embedding(config,weightType:.float32)
                        parameters += config.parameterCount
                        if !config.frozen { trainableParameters += config.parameterCount }
                    }
                case "input":
                    if node.config["shape"] != nil {
                        let contract=try FloatingInputConfiguration(node.config).tensor
                        result.outputTensorContracts[.init(id,"out")]=contract
                        if context == .bundledDataset && contract.layout.sample.dimensions != [2] {issue("The bundled dataset has two scalar features.",id)}
                    } else if let features=dimension(node,"features") {
                        result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:try BatchedTensorLayout(sample:[features]),dtype:.float32)
                        if context == .bundledDataset && features != 2 {issue("The bundled dataset has 2 features. Set Input features to 2.",id)}
                    }
                case "linear":
                    if let input=layout("in"),let expected=dimension(node,"in_features"),let output=dimension(node,"out_features") {
                        let actual=input.sample.dimensions.last!
                        if actual != expected {issue("Incoming width is \(actual), but this Linear expects \(expected). Set Input features to \(actual) or change the preceding component.",id)}
                        else {
                            result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:try input.linear(inputFeatures:expected,outputFeatures:output),dtype:.float32)
                            let count = (actual + ((node.config["bias"]?.boolValue ?? (graph.schemaVersion==1)) ? 1:0))*output
                            parameters += count; trainableParameters += count
                        }
                    }
                case "depthwise_conv2d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try DepthwiseConv2dConfiguration(values: values)
                        let resolved = try config.resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "conv_transpose3d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try ConvTranspose3dConfiguration(values: values)
                        let resolved = try config.resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "conv_transpose2d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try ConvTranspose2dConfiguration(values: values)
                        let resolved = try config.resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "conv_transpose1d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try ConvTranspose1dConfiguration(values: values)
                        let resolved = try config.resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "conv3d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try Conv3dConfiguration(values: values)
                        let resolved = try config.resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "conv2d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try Conv2dConfiguration(values: values)
                        let resolved = try config.resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "conv1d":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try Conv1dConfiguration(values: values)
                        let resolved = try config.resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "lazy_linear":
                    if let input = layout("in") {
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let resolved = try LazyLinearConfiguration(values: values).resolved(input: input)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: resolved.outputLayout, dtype: .float32)
                        parameters += resolved.parameterCount; trainableParameters += resolved.parameterCount
                    }
                case "mixture_experts":
                    if let source=contract("in") {
                        guard source.dtype == .float32 else {throw GraphConfigurationError("MoE features require float32.")}
                        let configuration=try MixtureExpertsConfiguration(values:node.config,registry:registry)
                        if node.config["mask"] == true,let mask=contract("mask") {
                            let expected=Array(source.layout.sample.dimensions.dropLast())
                            guard mask.dtype == .int64,mask.layout.sample.dimensions == (expected.isEmpty ? [1]:expected) else {throw GraphConfigurationError("MoE valid-unit mask needs int64 zero/one values matching leading unit axes.")}
                        }
                        result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:try configuration.outputLayout(source.layout),dtype:.float32)
                        parameters+=configuration.parameterCount;trainableParameters+=configuration.trainableCount
                    }
                case "bilinear":
                    if let first = contract("a"), let second = contract("b") {
                        guard first.dtype == .float32, second.dtype == .float32 else {
                            throw GraphConfigurationError("Bilinear requires float32 features on both inputs; integer indices need an embedding or explicit conversion.")
                        }
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let config = try BilinearConfiguration(values: values)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: try config.outputLayout(first: first.layout, second: second.layout), dtype: .float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "reshape","transpose","flatten":
                    if let input=layout("in") {
                        let config=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:try TensorTransformConfiguration(operation:node.type,values:config).outputLayout(for:input),dtype:contract("in")!.dtype)
                    }
                case "rms_norm":
                    if let input=layout("in") {
                        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        let config=try RMSNormConfiguration(values:values)
                        result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:try config.outputLayout(for:input),dtype:.float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "layer_norm":
                    if let input=layout("in") {
                        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        let config=try LayerNormConfiguration(values:values)
                        result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:try config.outputLayout(for:input),dtype:.float32)
                        parameters += config.parameterCount; trainableParameters += config.parameterCount
                    }
                case "softmax":
                    if let input=layout("in") {
                        let config=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:try SoftmaxConfiguration(values:config).outputLayout(for:input),dtype:.float32)
                    }
                case "prelu","log_softmax","dropout":
                    if let input=contract("in") {
                        let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        let configuration=try RemainingActivationConfiguration(componentID:node.type,values:values)
                        result.outputTensorContracts[.init(id,"out")]=try configuration.resolved(input:input)
                        let count=try configuration.parameterCount(for:input.layout)
                        parameters += count; trainableParameters += count
                    }
                case "relu","tanh","gelu","sigmoid","silu","leaky_relu","elu","output":
                    if node.type == "output",node.config["task"] == .string("gamePolicyValue") {
                        guard context == .policyValue || context == .frozenPolicy else {throw GraphConfigurationError("Use Game policy/value with Self-play, not a supervised objective.")}
                        if let policy=contract("policy"),let value=contract("value") {
                            guard policy.dtype == .float32,value.dtype == .float32,policy.layout.sample.dimensions.count==1,(1...256).contains(policy.layout.sample.dimensions[0]),value.layout.sample.dimensions == [1] else {throw GraphConfigurationError("Policy/value requires float32 [actions] logits and [1] acting-player value.")}
                            result.outputTensorContracts[.init(id,"policy")]=policy;result.outputTensorContracts[.init(id,"value")]=value
                        }
                    } else if let input=layout("in") {
                        result.outputTensorContracts[.init(id,"out")]=TypedTensorContract(layout:input,dtype:contract("in")!.dtype)
                        if node.type == "output" {
                            try ModelOutputConfiguration(node.config).check(sample: input.sample.dimensions, dtype: contract("in")!.dtype.rawValue)
                        }
                        if node.type == "output",context == .frozenPolicy {
                            guard contract("in")?.dtype == .float32,input.sample.dimensions.count==1,(1...256).contains(input.sample.dimensions[0]) else {throw GraphConfigurationError("A game policy requires one bounded float32 action-logit vector.")}
                        }
                        if node.type == "output",context == .explicitObjective {
                            guard contract("in")?.dtype == .float32,input.sample.dimensions.count>=1,input.sample.dimensions.count<=7,input.sample.dimensions.allSatisfy({(1...4096).contains($0)}) else {throw GraphConfigurationError("An explicit objective requires bounded float32 prediction arrays with 1–7 sample directions.")}
                        }
                        if node.type == "output",context != .explicitObjective,context != .frozenPolicy,input.sample.dimensions != [2] {
                            issue("This classification flow needs 2 output features; received \(input.sample.dimensions.last!).",id)
                        }
                    }
                case "concat", "add", "stack", "permute":
                    let ports = node.type == "permute" ? ["in"] : ["a", "b"]
                    let inputs = ports.compactMap { contract($0) }
                    if inputs.count == ports.count {
                        let values = try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                        result.outputTensorContracts[.init(id,"out")] = try TensorCompositionConfiguration(componentID:node.type,values:values).resolved(inputs:inputs)
                    }
                case "multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention","attention_mask","rotary_position":
                    let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                    let c=try AttentionConfiguration(componentID:node.type,values:values)
                    let inputs=c.configuredInputs.compactMap {contract($0)}
                    if inputs.count == c.configuredInputs.count {
                        for (port,input) in zip(c.configuredInputs,inputs) {
                            let expected:ContractDType=c.inputDTypes[port]!
                            guard input.dtype == expected else {throw GraphConfigurationError("Attention input \(port) needs \(expected.rawValue) arrays; masks use whole numbers, 1 blocked and 0 allowed.")}
                        }
                        let resolved=try c.resolved(layouts:inputs.map(\.layout))
                        for (port,shape) in resolved.outputs {result.outputTensorContracts[.init(id,port)] = .init(layout:shape,dtype:resolved.outputDTypes[port]!)}
                        parameters += c.parameterCount;trainableParameters += c.parameterCount
                    }
                case "rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell":
                    let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                    let c=try RecurrentConfiguration(componentID:node.type,values:values)
                    let inputs=c.configuredInputs.compactMap {contract($0)}
                    if inputs.count == c.configuredInputs.count {
                        guard inputs.allSatisfy({$0.dtype == .float32}) else {throw GraphConfigurationError("Memory inputs and states require floating values. Convert whole numbers explicitly first.")}
                        let resolved=try c.resolved(layouts:inputs.map(\.layout))
                        for (port,shape) in resolved.outputs {result.outputTensorContracts[.init(id,port)] = .init(layout:shape,dtype:.float32)}
                        parameters += c.parameterCount;trainableParameters += c.parameterCount
                    }
                case "batch_norm1d", "batch_norm2d", "instance_norm1d", "instance_norm2d", "group_norm":
                    if let input = contract("in") {
                        guard input.dtype == .float32 else { throw GraphConfigurationError("Channel normalization needs floating values. Convert whole numbers explicitly first.") }
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let configuration = try StatefulNormalizationConfiguration(componentID: node.type, values: values)
                        result.outputTensorContracts[.init(id, "out")] = TypedTensorContract(layout: try configuration.resolved(layout: input.layout), dtype: .float32)
                        parameters += configuration.parameterCount; trainableParameters += configuration.parameterCount
                    }
                case "unfold", "fold":
                    if let input = contract("in") {
                        guard input.dtype == .float32 else { throw GraphConfigurationError("Patch blocks need floating values. Convert whole-number positions explicitly before connecting them.") }
                        let values = try GraphMigration.effectiveConfiguration(of: node, schemaVersion: graph.schemaVersion, registry: registry)
                        let configuration = try SpatialPatchConfiguration(componentID: node.type, values: values)
                        result.outputTensorContracts[.init(id,"out")] = TypedTensorContract(layout: try configuration.resolved(layout: input.layout), dtype: .float32)
                    }
                case "split","chunk","gather","einsum":
                    let values=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:graph.schemaVersion,registry:registry)
                    let configuration=try TensorRoutingConfiguration(componentID:node.type,values:values)
                    let valuesByPort=configuration.configuredInputs.compactMap {contract($0)}
                    if valuesByPort.count == configuration.configuredInputs.count {
                        let resolved=try configuration.resolved(inputs:valuesByPort)
                        for (port,value) in resolved {result.outputTensorContracts[.init(id,port)]=value}
                        for edge in outgoing[id,default:[]] where resolved[edge.sourcePort] == nil {
                            issue("Output \(edge.sourcePort) is unavailable for this arrangement and incoming shape. Reconnect this saved wire to an available part.",id)
                        }
                    }
                case "matmul":
                    if let a=contract("a"),let b=contract("b") {result.outputTensorContracts[.init(id,"out")]=try a.matrixMultiplied(by:b)}
                default:break
                }
                if let shape=result.outputTensorContracts[.init(id,"out")]?.layout {
                    if shape.sample.dimensions.last!>4096 {
                        issue("Intermediate width exceeds the POC limit of 4096.",id)
                        result.outputTensorContracts.removeValue(forKey:.init(id,"out"))
                    } else if shape.sample.dimensions.count == 1 {
                        // Legacy consumers receive widths only for an actual [N, features] tensor.
                        result.widths[id]=shape.sample.dimensions[0]
                    }
                }
            } catch {issue(error.localizedDescription,id)}
        }
        if trainableParameters == 0 && context != .frozenPolicy { issue("Add a component with trainable parameters before running; frozen weights and nonaffine normalization do not provide trainable parameters.") }
        if parameters > 2_000_000 { issue("Model exceeds the POC limit of 2 million parameters.") }
        return result
    }
}
