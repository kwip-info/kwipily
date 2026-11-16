import Foundation
import KwipMLCore

/// Decodes pinned lifecycle files without loading tensors or consulting mutable workspace state.
/// Tensor and optimizer compatibility remain the worker's responsibility.
struct WorkflowLifecycleParentRecord {
    let evidence:WorkflowLifecycleParentEvidence
    let metrics:[EpochMetric]
    let device:String

    private struct Manifest:Decodable {
        let schemaVersion:Int
        let status:String
        let runID:String
        let checkpoint:Checkpoint
        struct Checkpoint:Decodable {
            let schemaVersion:Int
            let completedEpoch:Int
            let bytes:Int
            let sha256:String
            let metricsSHA256:String
            let contract:Contract
            struct Contract:Decodable {let runtime:Runtime}
            struct Runtime:Decodable {let device:String}
        }
    }
    private struct Row:Decodable {
        let epoch:Int
        let trainLoss:Double
        let validationLoss:Double
        let accuracy:Double?
    }

    static func hasScheduler(_ bytes:Data)throws->Bool {
        let object=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard case .object(let checkpoint)=object["checkpoint"],case .object(let contract)=checkpoint["contract"] else {throw WorkflowPhaseError("Parent checkpoint contract is missing.")}
        guard let scheduler=contract["scheduler"] else {return false}
        guard case .object(let wrapper)=scheduler,Set(wrapper.keys)==["schemaVersion","recipe"],wrapper["schemaVersion"] == 1,let recipe=wrapper["recipe"] else {throw WorkflowPhaseError("Parent schedule identity is invalid.")}
        let configuration=try SchedulerConfiguration.decodeRecipe(recipe)
        guard case .object(let settings)=object["settings"],let selected=settings["schedulerRecipe"],try SchedulerConfiguration.decodeRecipe(selected)==configuration else {throw WorkflowPhaseError("Parent schedule settings differ from its checkpoint.")}
        return true
    }
    private static func object(_ value:JSONValue?)->[String:JSONValue] {if case .object(let map)=value {return map};return [:]}
    private struct ObjectiveIdentity {
        let configuration:LossConfiguration
        let scalarization:String
        var allowsAccuracy:Bool {configuration.componentID == "loss_nll" || configuration.componentID == "loss_cross_entropy" && configuration.values["target_mode"] == .string("class_indices")}
    }
    private static func objectiveIdentity(_ bytes:Data)throws->ObjectiveIdentity? {
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard case .object(let checkpoint)=manifest["checkpoint"],case .object(let contract)=checkpoint["contract"] else {throw WorkflowPhaseError("Parent checkpoint contract is missing.")}
        let settings=object(manifest["settings"]),provenance=object(manifest["provenance"])
        guard let raw=contract["objective"] else {
            guard settings["objectiveRecipe"] == nil,settings["targetMapping"] == nil,settings["objectiveScalarization"] == nil,provenance["objective"] == nil else {throw WorkflowPhaseError("Undeclared parent objective identity cannot be restored.")}
            return nil
        }
        guard case .object(let identity)=raw,Set(identity.keys)==["schemaVersion","recipe","targetMapping","scalarization"],identity["schemaVersion"] == 1,let recipe=identity["recipe"],let mapping=identity["targetMapping"],let scalarization=identity["scalarization"]?.stringValue,["mean","sum"].contains(scalarization) else {throw WorkflowPhaseError("Parent objective identity is malformed.")}
        let configuration=try LossConfiguration.decodeRecipe(recipe)
        let target=try JSONDecoder().decode(LossTargetMapping.self,from:JSONEncoder().encode(mapping))
        try configuration.validate(targetMapping:target)
        let canonicalMapping=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(target))
        guard try WorkflowPhaseArtifacts.encode(recipe) == WorkflowPhaseArtifacts.encode(configuration.recipeValue),try WorkflowPhaseArtifacts.encode(mapping) == WorkflowPhaseArtifacts.encode(canonicalMapping),settings["objectiveRecipe"] == recipe,settings["targetMapping"] == mapping,settings["objectiveScalarization"] == .string(scalarization),case .object(let objective)=provenance["objective"],Set(objective.keys)==["recipe","targetMapping","scalarization","torchVersion"],objective["recipe"] == recipe,objective["targetMapping"] == mapping,objective["scalarization"] == .string(scalarization),case .object(let runtime)=manifest["runtime"],let torch=runtime["torch"]?.stringValue,!torch.isEmpty,torch.utf8.count<=128,objective["torchVersion"] == .string(torch),runtime["device"] == object(contract["runtime"])["device"] else {throw WorkflowPhaseError("Parent objective settings, provenance or runtime differ from its checkpoint.")}
        return .init(configuration:configuration,scalarization:scalarization)
    }
    static func hasObjective(_ bytes:Data)throws->Bool {try objectiveIdentity(bytes) != nil}
    private static func finiteMetadata(_ value:JSONValue)->Bool {
        switch value {
        case .object(let map):return map.values.allSatisfy(finiteMetadata)
        case .array(let items):return items.allSatisfy(finiteMetadata)
        case .integer:return true
        case .number(let number):return number.isFinite
        case .string,.null:return true
        case .bool:return false
        }
    }
    static func decode(_ contents:[String:Data],dataLoading:WorkflowDataLoadingIdentity?=nil)throws->Self {
        try WorkflowCheckpointInventory.verify(contents)
        let manifest=try JSONDecoder().decode(Manifest.self,from:contents["manifest.json"]!)
        let checkpoint=manifest.checkpoint
        guard manifest.schemaVersion == 1,manifest.status == "completed",UUID(uuidString:manifest.runID) != nil,
              checkpoint.schemaVersion == 1,["cpu","mps"].contains(checkpoint.contract.runtime.device),(1...10_000).contains(checkpoint.completedEpoch),
              contents["checkpoint.pt"]!.count == checkpoint.bytes,
              WorkflowArchiveCodec.hash(contents["checkpoint.pt"]!) == checkpoint.sha256,
              WorkflowArchiveCodec.hash(contents["metrics.json"]!) == checkpoint.metricsSHA256 else {
            throw WorkflowPhaseError("Parent checkpoint or metrics differ from the completed lifecycle manifest.")
        }
        let objective=try objectiveIdentity(contents["manifest.json"]!)
        let policyManifest=try JSONDecoder().decode([String:JSONValue].self,from:contents["manifest.json"]!)
        let loadingIdentity=try dataLoading ?? WorkflowDataLoadingParentIdentity.validate(policyManifest)
        try WorkflowDataLoadingVerifier.validateAttachment(manifest:policyManifest,identity:loadingIdentity)
        let gradientIdentity=try WorkflowGradientPolicyVerifier.storedIdentity(policyManifest,dataLoading:loadingIdentity)
        let rawRows=try JSONDecoder().decode([JSONValue].self,from:contents["metrics.json"]!)
        if let objective {
            for raw in rawRows {
                guard case .object(let row)=raw,case .object(let summary)=row["objective"],Set(summary.keys)==["componentID","reduction","scalarization","training","validation"],summary["componentID"] == .string(objective.configuration.componentID),summary["reduction"] == objective.configuration.values["reduction"],summary["scalarization"] == (objective.configuration.values["reduction"] == .string("none") ? .string(objective.scalarization):.null),case .object=summary["training"],case .object=summary["validation"],finiteMetadata(.object(summary)),row["accuracy"] == nil || objective.allowsAccuracy && row["accuracy"]?.doubleValue != nil else {throw WorkflowPhaseError("Parent objective metric identity or available accuracy is invalid.")}
            }
        } else if rawRows.contains(where:{object($0)["objective"] != nil}) {throw WorkflowPhaseError("Undeclared objective metric history cannot be restored.")}
        let rows=try JSONDecoder().decode([Row].self,from:contents["metrics.json"]!)
        guard rows.count == checkpoint.completedEpoch,rows.enumerated().allSatisfy({ index,row in
            row.epoch == index+1 && row.trainLoss.isFinite && row.validationLoss.isFinite
                && (row.accuracy.map {$0.isFinite && (0...1).contains($0)} ?? (objective != nil))
        }) else {
            throw WorkflowPhaseError("Parent metrics must contain a finite observation for every completed epoch in order.")
        }
        return Self(evidence:.init(workerRunID:manifest.runID,checkpointSHA256:checkpoint.sha256,completedEpoch:checkpoint.completedEpoch,hasScheduler:try hasScheduler(contents["manifest.json"]!),hasObjective:objective != nil,gradientPolicy:gradientIdentity?.gradientPolicy,trainability:gradientIdentity?.trainability,gradientIdentity:gradientIdentity,dataLoadingIdentity:loadingIdentity),
                    metrics:rows.map { .init(epoch:$0.epoch,trainLoss:$0.trainLoss,validationLoss:$0.validationLoss,accuracy:$0.accuracy) },device:checkpoint.contract.runtime.device)
    }
}
