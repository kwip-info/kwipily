import KwipMLCore

extension ModelCanvasAdapter {
    func embeddingBagBoundFields(nodeID:String)->Set<String> {
        let exposed=activeDefinition?.parameters.flatMap(\.targets).filter {$0.componentID==nodeID}.map(\.field) ?? []
        let derived=activeDefinition?.integerProducts?.flatMap(\.targets).filter {$0.componentID==nodeID}.map(\.field) ?? []
        return Set(exposed+derived)
    }
    @discardableResult func setEmbeddingBag(configuration:[String:JSONValue],
        nodeID:String,lease:ModelEditorLease,expectedSourceConfig:[String:JSONValue])->Bool {
        guard !isBusy,selectedNodeIDs == [nodeID],owner.lease(path:path)==lease,
              let node=graph.nodes.first(where:{$0.id==nodeID}),scope?.nodes.contains(where:{$0.id==nodeID}) == true,
              node.type=="embedding_bag",node.config==expectedSourceConfig else {return false}
        let command=GraphCommand.setEmbeddingBag(nodeID:nodeID,configuration:configuration)
        let result:GraphCommandResult?
        if path.isEmpty {
            guard owner.allowsRootCanvas else {return false}
            result=owner.applyRootCommand(command,selection:.init(nodeIDs:selectedNodeIDs),lease:lease)
        } else {result=owner.applyDraftCommand(command,selection:.init(nodeIDs:selectedNodeIDs),lease:lease)}
        guard result != nil else {statusMessage=owner.error ?? "Embedding bag settings were not applied.";return false}
        statusMessage=owner.isFileBacked ? "":owner.editStatus
        return true
    }
}
extension WorkspaceStore {
    @discardableResult func setEmbeddingBag(configuration:[String:JSONValue],
        nodeID:String,expectedSourceConfig:[String:JSONValue],expectedProjectID:String?)->Bool {
        guard modelAuthoring==nil,workflowSession==nil,!isBusy,activeProjectID==expectedProjectID,
              selectedNodeIDs == [nodeID],selectedNodeID==nodeID,
              let node=graph.nodes.first(where:{$0.id==nodeID}),node.type=="embedding_bag",node.config==expectedSourceConfig else {return false}
        return applyCommand(.setEmbeddingBag(nodeID:nodeID,configuration:configuration))
    }
}
