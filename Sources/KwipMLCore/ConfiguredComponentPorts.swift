import Foundation

extension ComponentDefinition {
    /// Configured ports never rewrite saved edges; unavailable wires remain repairable.
    public func configuredInputs(_ configuration:[String:JSONValue]) -> [String] {
        if id == "kwip.data.preprocess" {return configuration["mode"] == .string("table") ? ["table"]:["training","validation","test"]}
        if id == "mixture_experts" {return configuration["mask"] == true ? ["in","mask"]:["in"]}
        if id == "output",configuration["task"] == .string("gamePolicyValue") {return ["policy","value"]}
        if id == "embedding_bag" {return (try? EmbeddingBagConfiguration(values:configuration).inputPorts) ?? []}
        if AttentionConfiguration.componentIDs.contains(id) {return (try? AttentionConfiguration(componentID:id,values:configuration).configuredInputs) ?? inputs}
        if RecurrentConfiguration.componentIDs.contains(id) {return (try? RecurrentConfiguration(componentID:id,values:configuration).configuredInputs) ?? inputs}
        guard TensorRoutingConfiguration.componentIDs.contains(id) else {return inputs}
        return (try? TensorRoutingConfiguration(componentID:id,values:configuration).configuredInputs) ?? inputs
    }
    public func configuredOutputs(_ configuration:[String:JSONValue],inputLayout:BatchedTensorLayout?=nil) -> [String] {
        if id == "kwip.data.preprocess" {return configuration["mode"] == .string("table") ? ["table"]:["training","validation","test"]}
        if id == "bag_input" {return (try? BagSourceConfiguration(values:configuration).outputPorts) ?? []}
        if AttentionConfiguration.componentIDs.contains(id) {return (try? AttentionConfiguration(componentID:id,values:configuration).outputPorts) ?? []}
        if id == "split" || id == "chunk" {
            guard let routing=try? TensorRoutingConfiguration(componentID:id,values:configuration) else {return []}
            guard let inputLayout else {return outputs}
            return (try? routing.outputPorts(layouts:[inputLayout])) ?? []
        }
        if let bundle = BundledPoolingComponents.entries[id] {
            return outputs.filter { port in
                guard let field = bundle.outputConditions[port] else { return true }
                return configuration[field] == .bool(true)
            }
        }
        return outputs
    }
}
