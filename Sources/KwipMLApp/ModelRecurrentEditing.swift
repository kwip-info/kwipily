import KwipMLCore

extension ModelCanvasAdapter {
    func recurrentBoundFields(nodeID:String)->Set<String> {
        let exposed=activeDefinition?.parameters.flatMap(\.targets).filter {$0.componentID==nodeID}.map(\.field) ?? []
        let derived=activeDefinition?.integerProducts?.flatMap(\.targets).filter {$0.componentID==nodeID}.map(\.field) ?? []
        return Set(exposed+derived)
    }
    @discardableResult func setRecurrent(_ method:RecurrentSequenceConfiguration.Method,configuration:[String:JSONValue]?=nil,
        nodeID:String,lease:ModelEditorLease,expectedType:String,expectedSourceConfig:[String:JSONValue])->Bool {
        guard !isBusy,selectedNodeIDs == [nodeID],owner.lease(path:path)==lease,
              let node=graph.nodes.first(where:{$0.id==nodeID}),scope?.nodes.contains(where:{$0.id==nodeID}) == true,
              node.type==expectedType,node.config==expectedSourceConfig,RecurrentAuthoringConfiguration.method(for:node.type) != nil else {return false}
        let bound=recurrentBoundFields(nodeID:nodeID)
        guard bound.isEmpty || RecurrentAuthoringConfiguration.primitiveID(method,isCell:node.type.hasSuffix("_cell"))==node.type else {statusMessage="Change the exposed or derived binding before replacing this recurrent method.";return false}
        // The canonical draft owner compares supplied bound values with its resolved
        // binding values and preserves exact raw omissions after the command. Do not
        // authenticate bindings against this presentation projection's fallback config.
        let command=GraphCommand.setRecurrent(nodeID:nodeID,method:method,configuration:configuration)
        let result:GraphCommandResult?
        if path.isEmpty {
            guard owner.allowsRootCanvas else {return false}
            result=owner.applyRootCommand(command,selection:.init(nodeIDs:selectedNodeIDs),lease:lease)
        } else {result=owner.applyDraftCommand(command,selection:.init(nodeIDs:selectedNodeIDs),lease:lease)}
        guard result != nil else {statusMessage=owner.error ?? "Recurrent settings were not applied.";return false}
        statusMessage=owner.isFileBacked ? "":owner.editStatus
        return true
    }
}
extension WorkspaceStore {
    @discardableResult func setRecurrent(_ method:RecurrentSequenceConfiguration.Method,configuration:[String:JSONValue]?=nil,
        nodeID:String,expectedType:String,expectedSourceConfig:[String:JSONValue],expectedProjectID:String?,
        allowedMethods:[RecurrentSequenceConfiguration.Method]=RecurrentSequenceConfiguration.Method.allCases)->Bool {
        guard modelAuthoring==nil,workflowSession==nil,!isBusy,activeProjectID==expectedProjectID,
              selectedNodeIDs == [nodeID],selectedNodeID==nodeID,allowedMethods.contains(method),
              let node=graph.nodes.first(where:{$0.id==nodeID}),node.type==expectedType,node.config==expectedSourceConfig,
              RecurrentAuthoringConfiguration.method(for:node.type) != nil else {return false}
        return applyCommand(.setRecurrent(nodeID:nodeID,method:method,configuration:configuration))
    }
}
