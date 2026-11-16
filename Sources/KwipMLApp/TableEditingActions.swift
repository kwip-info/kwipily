import KwipMLCore

extension DataFlowWorkspaceStore {
    @discardableResult func applyTableConfiguration(_ config: [String: JSONValue], nodeID: String, expected: [String: JSONValue]) -> Bool {
        guard !isBusy, projectID == workspace.activeProjectID, selectedNodeID == nodeID,
              let node = project.graph.nodes.first(where: { $0.id == nodeID }), node.config == expected else { statusMessage = "Configuration changed. Reload this editor before applying its draft."; return false }
        do {
            if node.operatorID==SequenceWindowsConfiguration.operatorID {
                try SequenceWindowsConfiguration.validateConnectedPorts(config,inputs:Set(project.graph.edges.filter {$0.to.node==nodeID}.map { $0.to.port }),outputs:Set(project.graph.edges.filter {$0.from.node==nodeID}.map { $0.from.port }))
            }
            let schema = availableColumnSchema(for: nodeID,verifySources:false)
            let checked:[String:JSONValue]
            if node.operatorID == "kwip.data.union" {
                let ports=try UnionConfiguration(config).inputs
                checked=try UnionTableConfiguration.validate(config,schemas:Dictionary(uniqueKeysWithValues:ports.map {($0,availableColumnSchema(for:nodeID,verifySources:false,inputPort:$0))}),connected:Set(project.graph.edges.filter {$0.to.node==nodeID}.map { $0.to.port }))
            } else {checked = try DataTableConfiguration.validate(config,operatorID:node.operatorID,schema:schema,leftSchema:availableColumnSchema(for:nodeID,verifySources:false,inputPort:"left"),rightSchema:availableColumnSchema(for:nodeID,verifySources:false,inputPort:"right"),definition:operators?.operatorDefinition(id:node.operatorID))}
            var candidate = project.graph
            guard let index = candidate.nodes.firstIndex(where: { $0.id == nodeID }) else { return false }
            candidate.nodes[index].config = checked
            _ = try DataFlowCodec.encodeGraph(candidate)
            configure(checked, nodeID: nodeID)
            guard project.graph.nodes.first(where: { $0.id == nodeID })?.config == checked else { return false }
            statusMessage = "Configuration applied. Prepare data to inspect actual output rows."
            return true
        } catch { statusMessage = error.localizedDescription; return false }
    }
}
