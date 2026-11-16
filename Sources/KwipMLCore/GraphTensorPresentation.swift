import Foundation

extension BatchedTensorLayout {
    public var displayShape:String {"["+displayDimensions.joined(separator:", ")+"]"}
}

extension ValidationResult {
    /// Inferred declarations, never measured evidence. Input ports describe their source tensor.
    public func tensorPortDescriptor(nodeID:String,port:String,direction:PortDirection,graph:GraphDocument,registry:ComponentRegistry)->CanvasPortDescriptor {
        guard let node=graph.nodes.first(where:{$0.id == nodeID}),let definition=configuredDefinition(nodeID:nodeID,graph:graph,registry:registry),
              (direction == .input ? definition.inputs:definition.outputs).contains(port) else {
            return .unknown(direction:direction)
        }
        if direction == .output, !definition.configuredOutputs(node.config).contains(port) {
            return .unknown(direction:direction,reason:"Output disabled by component settings")
        }
        let contract:TypedTensorContract?
        let bagContract:BagPortContract?
        if direction == .output {
            contract=outputTensorContracts[.init(nodeID,port)]
            bagContract=outputBagContracts[.init(nodeID,port)]
        } else {
            let incoming=graph.edges.filter {$0.target == nodeID && $0.targetPort == port}
            contract=incoming.count == 1 ? outputTensorContracts[.init(incoming[0].source,incoming[0].sourcePort)]:nil
            bagContract=incoming.count == 1 ? outputBagContracts[.init(incoming[0].source,incoming[0].sourcePort)]:nil
        }
        if let bagContract {return bagContract.canvasDescriptor(direction:direction)}
        return .init(kind:.tensor,direction:direction,declared:contract.map {[$0.layout.displayShape,$0.dtype.rawValue]} ?? ["Shape unresolved"])
    }
}

extension ValidationResult {
    /// Instance presentation uses the declared vocabulary until routing resolves it.
    public func configuredDefinition(nodeID:String,graph:GraphDocument,registry:ComponentRegistry)->ComponentDefinition? {
        guard let node=graph.nodes.first(where:{$0.id == nodeID}) else {return nil}
        guard var definition=registry.component(type:node.type) else {return nil}
        guard TensorRoutingConfiguration.componentIDs.contains(node.type) || RecurrentConfiguration.componentIDs.contains(node.type) || AttentionConfiguration.componentIDs.contains(node.type) || ["bag_input","embedding_bag"].contains(node.type) else {return definition}
        definition.inputs=definition.configuredInputs(node.config)
        let incoming=graph.edges.filter {$0.target == nodeID && $0.targetPort == "in"}
        let inputLayout=incoming.count == 1 ? outputTensorContracts[.init(incoming[0].source,incoming[0].sourcePort)]?.layout:nil
        definition.outputs=definition.configuredOutputs(node.config,inputLayout:inputLayout)
        return definition
    }
}


private extension BagPortContract {
    /// Physical declarations from the validated source coordinate domain, never
    /// a fabricated BatchedTensorLayout or a measured item/group count.
    func canvasDescriptor(direction:PortDirection)->CanvasPortDescriptor {
        let fixed=configuration.layout == "fixed"
        let itemShape=fixed ? "[bags, \(configuration.maximumItemsPerBag)]":"[items]"
        let geometry=fixed ? "Each row is one observation group":"No observation axis in the item array; offsets keep groups separate"
        let roleLabel:String
        var declared:[String]
        switch role {
        case .indices:
            roleLabel="Category IDs"
            declared=[itemShape,dtype.rawValue,geometry,"IDs select table rows; integer gradient unavailable"]
        case .offsets:
            roleLabel="Group boundaries"
            declared=[configuration.includeLastOffset ? "[bags + 1]":"[bags]",dtype.rawValue,
                      "Offsets mark group starts; equal offsets preserve empty groups",
                      configuration.includeLastOffset ? "Final offset equals the number of items":"Final group continues to the end of the item array",
                      "Integer gradient unavailable"]
        case .perSampleWeights:
            roleLabel="Item influence weights"
            declared=[itemShape,dtype.rawValue,geometry,"Real weights stay paired with their item IDs",
                      "Weights support a gradient with sum; unused source cells contribute zero"]
        }
        declared.append("Maximum \(configuration.maximumItemsPerBag) items per group")
        return .init(kind:.tensor,direction:direction,role:roleLabel,declared:declared)
    }
}
