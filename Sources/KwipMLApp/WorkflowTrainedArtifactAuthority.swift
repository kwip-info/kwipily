import Foundation
import KwipMLCore

/// A Train output refers to one verified completed run, never an arbitrary latest
/// history item. Presentation edits do not alter its scientific identity.
enum WorkflowTrainedArtifactAuthority {
    static func scientificIdentity(_ plan:WorkflowExecutionPlan)throws->String {
        let bytes=try WorkflowArchiveCodec.encode(WorkflowArchivePlan(plan,compilerContractVersion:15))
        guard var object=try JSONSerialization.jsonObject(with:bytes) as? [String:Any] else {throw failure("Invalid scientific plan.")}
        // These are presentation/source reconstruction fields. The semantic model
        // hash and origins retain topology, parameters and parameter ownership;
        // every recipe, source endpoint and executable setting remains included.
        for key in ["revision","modelGraph","modelProvenance","modelAuthoringRevision"] {object.removeValue(forKey:key)}
        // Inspection-only side branches do not change the learned model's meaning.
        // Keep legacy identities reproducible when compiler16 records extra Views.
        if var data=object["dataflow"] as? [String:Any],let nodes=data["nodes"] as? [[String:Any]],let edges=data["edges"] as? [[String:Any]] {
            var required=Set([plan.training.node,plan.validation.node])
            required.formUnion(nodes.filter {$0["operatorID"] as? String == "kwip.data.outputDelimited"}.compactMap {$0["id"] as? String})
            var changed=true
            while changed {
                changed=false
                for edge in edges {
                    if let from=edge["from"] as? [String:Any],let to=edge["to"] as? [String:Any],let source=from["node"] as? String,let target=to["node"] as? String,required.contains(target),required.insert(source).inserted {changed=true}
                }
            }
            data["nodes"]=nodes.filter {($0["id"] as? String).map(required.contains) == true}
            data["edges"]=edges.filter {edge in
                guard let from=edge["from"] as? [String:Any],let to=edge["to"] as? [String:Any],let source=from["node"] as? String,let target=to["node"] as? String else {return false}
                return required.contains(source) && required.contains(target)
            }
            object["dataflow"]=data
            object["sourceAddresses"]=(object["sourceAddresses"] as? [[String:Any]])?.filter {($0["localID"] as? String).map(required.contains) == true}
        }
        return WorkflowArchiveCodec.hash(try JSONSerialization.data(withJSONObject:object,options:[.sortedKeys,.withoutEscapingSlashes]))
    }
    @MainActor static func currentIdentity(train:WorkflowNodeAddress,owner:WorkflowAuthoringStore)throws->String {
        let document=owner.document
        return try scientificIdentity(WorkflowCompiler.compile(document,train:train,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry,
            contractVersion:max(2,WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema:document.schemaVersion))))
    }
    @MainActor static func resolve(source:WorkflowTrainedModelSource,owner:WorkflowAuthoringStore,
                                  bindings:WorkflowPrivateFrozenModelBindings,root:URL)async throws->WorkflowFrozenModelReference {
        let lease=owner.lease(),revision=bindings.revision
        guard bindings.owner === owner,owner.isCurrent(lease),!owner.isBusy,!owner.hasPendingInspectorEdits,
              let reference=bindings.reference(for:source.address),reference.hasValidPins,reference.projectID==owner.projectID else {
            throw failure("No verified completed model is available for this component. Complete Train or select a saved checkpoint first.")
        }
        guard let node=owner.document.nodes.first(where:{$0.address==source.address}),source.endpoint.port=="model" else {throw failure("Completed model provider is unavailable.")}
        if source.isTrain {
            guard case .train=node.payload,owner.document.schemaVersion>=17 else {throw failure("Train output requires its current versioned provider.")}
        } else {
            guard case .frozenModel(let config)=node.payload,config.isEmpty else {throw failure("Saved checkpoint authority cannot stand in for Train output.")}
            return reference
        }
        let expected=try currentIdentity(train:source.address,owner:owner)
        let worker=Task.detached(priority:.userInitiated) {
            try await WorkflowHistoryVerificationWork.run {try verify(reference:reference,train:source.address,scientificIdentity:expected,root:root)}
        }
        try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        guard owner.isCurrent(lease),!owner.isBusy,!owner.hasPendingInspectorEdits,bindings.revision==revision,
              bindings.reference(for:source.address)==reference,
              try currentIdentity(train:source.address,owner:owner)==expected else {throw failure("Train or its completed model changed during verification.")}
        return reference
    }
    @MainActor static func acceptCompletedTrain(history:WorkflowHistoryReference,train:WorkflowNodeAddress,
                                               owner:WorkflowAuthoringStore,bindings:WorkflowPrivateFrozenModelBindings,root:URL)async throws->WorkflowFrozenModelReference {
        let lease=owner.lease(),revision=bindings.revision
        guard bindings.owner === owner,owner.isCurrent(lease),!owner.isBusy,!owner.hasPendingInspectorEdits,history.projectID==owner.projectID else {
            throw failure("Completed Train publication requires its current idle project.")
        }
        let expected=try currentIdentity(train:train,owner:owner)
        let worker=Task.detached(priority:.userInitiated) {
            try await WorkflowHistoryVerificationWork.run {
                let reference=try WorkflowFrozenModelSnapshot.select(history,root:root)
                try verify(reference:reference,train:train,scientificIdentity:expected,root:root)
                return reference
            }
        }
        let reference=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        guard owner.isCurrent(lease),!owner.isBusy,!owner.hasPendingInspectorEdits,bindings.revision==revision,
              try currentIdentity(train:train,owner:owner)==expected else {throw failure("Train changed before its completed model could be published.")}
        return reference
    }
    private static func verify(reference:WorkflowFrozenModelReference,train:WorkflowNodeAddress,scientificIdentity expected:String,root:URL)throws {
        try Task.checkCancellation()
        guard let id=UUID(uuidString:reference.directoryID) else {throw failure("Invalid completed model directory.")}
        let history=WorkflowHistoryReference(projectID:reference.projectID,directoryID:id)
        guard case .verified(let parent)=WorkflowHistoryReader.load(history,root:root),parent.outcome.status == .completed,parent.modelContent != nil,
              parent.capture.train==train,parent.plan.train==train,
              try scientificIdentity(parent.plan)==expected,
              try WorkflowFrozenModelSnapshot.select(history,root:root)==reference else {
            throw failure("Train output no longer matches its verified completed scientific plan. Run Train again.")
        }
        try Task.checkCancellation()
    }
    private static func failure(_ message:String)->DataFlowContractError {.init(message)}
}
