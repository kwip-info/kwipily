import KwipMLCore

extension ModelCanvasAdapter {
    func poolingBoundFields(nodeID:String)->Set<String> {
        let exposed=activeDefinition?.parameters.flatMap(\.targets).filter {$0.componentID==nodeID}.map(\.field) ?? []
        let derived=activeDefinition?.integerProducts?.flatMap(\.targets).filter {$0.componentID==nodeID}.map(\.field) ?? []
        return Set(exposed+derived)
    }
    @discardableResult func setPooling(_ variant:PoolingConfiguration.Variant,configuration:[String:JSONValue]?=nil,
        nodeID:String,lease:ModelEditorLease,expectedType:String,expectedSourceConfig:[String:JSONValue])->Bool {
        guard !isBusy,selectedNodeIDs == [nodeID],owner.lease(path:path)==lease,
              let node=graph.nodes.first(where:{$0.id==nodeID}),scope?.nodes.contains(where:{$0.id==nodeID}) == true,
              node.type==expectedType,node.config==expectedSourceConfig,PoolingConfiguration.Variant(rawValue:node.type) != nil else {return false}
        let bound=poolingBoundFields(nodeID:nodeID)
        guard bound.isEmpty || variant.rawValue==node.type else {statusMessage="Change the exposed or derived binding before replacing this pooling variant.";return false}
        // The canonical draft owner compares supplied bound values with its resolved
        // binding values and preserves exact raw omissions after the command. Do not
        // authenticate bindings against this presentation projection's fallback config.
        let command=GraphCommand.setPooling(nodeID:nodeID,variant:variant,configuration:configuration)
        let result:GraphCommandResult?
        if path.isEmpty {
            guard owner.allowsRootCanvas else {return false}
            result=owner.applyRootCommand(command,selection:.init(nodeIDs:selectedNodeIDs),lease:lease)
        } else {result=owner.applyDraftCommand(command,selection:.init(nodeIDs:selectedNodeIDs),lease:lease)}
        guard result != nil else {statusMessage=owner.error ?? "Pooling settings were not applied.";return false}
        statusMessage=owner.isFileBacked ? "":owner.editStatus
        return true
    }
}
extension WorkspaceStore {
    @discardableResult func setPooling(_ variant:PoolingConfiguration.Variant,configuration:[String:JSONValue]?=nil,
        nodeID:String,expectedType:String,expectedSourceConfig:[String:JSONValue],expectedProjectID:String?,
        allowedVariants:[PoolingConfiguration.Variant]=PoolingConfiguration.Variant.allCases)->Bool {
        guard modelAuthoring==nil,workflowSession==nil,!isBusy,activeProjectID==expectedProjectID,
              selectedNodeIDs == [nodeID],selectedNodeID==nodeID,allowedVariants.contains(variant),
              let node=graph.nodes.first(where:{$0.id==nodeID}),node.type==expectedType,node.config==expectedSourceConfig,
              PoolingConfiguration.Variant(rawValue:node.type) != nil else {return false}
        return applyCommand(.setPooling(nodeID:nodeID,variant:variant,configuration:configuration))
    }
}
