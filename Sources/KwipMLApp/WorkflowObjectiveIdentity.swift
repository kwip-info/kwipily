import Foundation
import KwipMLCore

/// Identity only: callers first verify anchored preparation and captured model geometry.
/// Hash/pickle verification and numeric artifact decoding retain their separate boundaries.
enum WorkflowObjectiveIdentity {
    static func validate(_ manifest:[String:JSONValue],selection:PreparedObjectiveSelection,recipe:LossRecipe,
                         inputContract:TypedTensorContract,outputContract:TypedTensorContract,expectedEpochs:Int,
                         policy:TrainingSampleCapturePolicy,explicitSampleCapture:Bool = false,dataLoading:WorkflowDataLoadingIdentity? = nil,usesNamedValues:Bool=false)throws->WorkflowObjectiveArtifactContext {
        let common=try validateEvaluationIdentity(manifest,selection:selection,recipe:recipe,outputContract:outputContract,expectedEpochs:expectedEpochs,policy:policy,explicitSampleCapture:explicitSampleCapture,dataLoading:dataLoading)
        guard case .object(let provenance)=manifest["provenance"],let actual=provenance["modelInput"] else {throw WorkflowPhaseError("Prepared model-input provenance is absent.")}
        if let plan=selection.namedInputPlan {
            try plan.validate()
            guard provenance["namedInputPlan"] == .object(plan.values),actual == .object(["kind":.string(usesNamedValues ? "namedInputs":"tensor"),"schemaVersion":1,"inputs":plan.values["inputs"]!]) else {throw WorkflowPhaseError("Named input geometry or preparation differs from its captured mapping.")}
        } else {
            guard [ContractDType.float32,.int64].contains(inputContract.dtype),inputContract.layout.sample.elementCount==selection.targetMapping.inputColumns.count else {throw WorkflowPhaseError("Prepared input columns do not fill the captured model input.")}
            let expectedInput=JSONValue.object(["dtype":.string(inputContract.dtype.rawValue),"sampleShape":.array(inputContract.layout.sample.dimensions.map {.integer(Int64($0))}),"columnOrder":.array(selection.targetMapping.inputColumns.map(JSONValue.string))])
            guard try WorkflowPhaseArtifacts.encode(actual)==WorkflowPhaseArtifacts.encode(expectedInput) else {throw WorkflowPhaseError("Saved input geometry differs from the captured model.")}
        }
        return try WorkflowObjectiveArtifactContext(recipe:recipe,runID:common.runID,inputContract:inputContract,outputSampleShape:common.outputSampleShape,trainingCount:common.trainingCount,evaluationCount:common.evaluationCount,expectedEpochs:common.expectedEpochs,scalarization:common.scalarization,device:common.device,policy:common.policy,explicitSampleCapture:common.explicitSampleCapture,namedInputPlan:selection.namedInputPlan,usesNamedValues:usesNamedValues)
    }
    /// Shared captured objective identity. Each input domain independently validates
    /// its own source, mapping and geometry before using this evaluation context.
    static func validateEvaluationIdentity(_ manifest:[String:JSONValue],selection:PreparedObjectiveSelection,recipe:LossRecipe,
                         outputContract:TypedTensorContract,expectedEpochs:Int,
                         policy:TrainingSampleCapturePolicy,explicitSampleCapture:Bool = false,dataLoading:WorkflowDataLoadingIdentity? = nil)throws->WorkflowObjectiveEvaluationContext {
        func require(_ valid:Bool,_ message:String)throws {guard valid else {throw WorkflowPhaseError(message)}}
        func object(_ raw:JSONValue?,_ name:String)throws->[String:JSONValue] {
            guard case .object(let value)=raw else {throw WorkflowPhaseError("Saved objective \(name) is missing.")};return value
        }
        func matches(_ raw:JSONValue?,_ expected:JSONValue)throws->Bool {
            guard let raw else {return false}
            return try WorkflowPhaseArtifacts.encode(raw) == WorkflowPhaseArtifacts.encode(expected)
        }
        try selection.validate()
        try require(recipe.source.domain == .control && !recipe.source.localID.isEmpty,"Objective source must be a captured control node.")
        try require(selection.objectiveRecipe == recipe.configuration.recipeValue && selection.targetMapping == recipe.targetMapping,"Prepared objective differs from the captured canvas recipe or column roles.")
        try WorkflowMappingRequirements.validateObjective(prediction:outputContract,recipe:recipe)
        let settings=try object(manifest["settings"],"settings"),provenance=try object(manifest["provenance"],"provenance")
        let checkpoint=try object(manifest["checkpoint"],"checkpoint"),contract=try object(checkpoint["contract"],"checkpoint contract")
        let runtime=try object(manifest["runtime"],"runtime")
        guard let runID=manifest["runID"]?.stringValue,UUID(uuidString:runID) != nil,
              let actualDevice=runtime["device"]?.stringValue,let device=ContractDevice(rawValue:actualDevice),[ContractDevice.cpu,.mps].contains(device),
              let torch=runtime["torch"]?.stringValue,!torch.isEmpty,torch.utf8.count<=128 else {throw WorkflowPhaseError("Objective run identity or supported runtime is invalid.")}
        let checkpointRuntime=try object(contract["runtime"],"checkpoint runtime")
        try require(checkpointRuntime["device"] == .string(actualDevice) && checkpointRuntime["torch"] == .string(torch),"Checkpoint runtime differs from the actual objective device or Torch version.")
        try require(manifest["schemaVersion"] == .integer(1) && manifest["status"] == .string("completed") && settings["epochs"]?.integerValue == expectedEpochs,"Objective artifact is not the requested completed run.")
        try require(settings["requestedDevice"] == .string("auto") || settings["requestedDevice"] == .string(actualDevice),"Actual objective device differs from its request.")
        let mapping=try JSONDecoder().decode(JSONValue.self,from:WorkflowPhaseArtifacts.encode(selection.targetMapping))
        let scalarization=JSONValue.string(selection.scalarization.rawValue)
        try require(settings["partitionMode"] == .string("prepared") && provenance["targetColumn"] == nil,"Explicit objective cannot use generated binary targets or a legacy scalar target.")
        try require(try matches(settings["objectiveRecipe"],selection.objectiveRecipe) && matches(settings["targetMapping"],mapping) && settings["objectiveScalarization"] == scalarization,"Saved objective settings differ from the captured recipe, roles or reduction.")
        try require(try matches(provenance["objectiveRecipe"],selection.objectiveRecipe) && matches(provenance["objectiveMapping"],mapping) && provenance["featureColumns"] == .array(selection.targetMapping.inputColumns.map(JSONValue.string)),"Saved objective provenance differs from its selected column roles.")
        let expectedObjective=JSONValue.object(["recipe":selection.objectiveRecipe,"targetMapping":mapping,"scalarization":scalarization,"torchVersion":.string(torch)])
        let expectedCheckpoint=JSONValue.object(["schemaVersion":.integer(1),"recipe":selection.objectiveRecipe,"targetMapping":mapping,"scalarization":scalarization])
        try require(try matches(provenance["objective"],expectedObjective) && matches(contract["objective"],expectedCheckpoint),"Checkpoint objective identity differs from its captured recipe or runtime.")
        let prepared=try object(provenance["preparation"],"preparation"),r=selection.reference
        let pins:[String:JSONValue]=["manifestSHA256":.string(r.manifestSHA256),"graphDigest":.string(r.graphDigest),"trainingNode":.string(r.trainingNode),"trainingPort":.string(r.trainingPort),"validationNode":.string(r.validationNode),"validationPort":.string(r.validationPort)]
        try require(pins.allSatisfy({prepared[$0.key] == $0.value}),"Objective training used different prepared partitions.")
        guard let partition=try WorkflowPartitionEvidence.decode(.object(provenance)) else {throw WorkflowPhaseError("Objective partition evidence is missing.")}
        try WorkflowDataLoadingVerifier.validateAttachment(manifest:manifest,identity:dataLoading)
        if let dataLoading {try require(dataLoading.originalRows==partition.trainingCount,"Data loading differs from the original prepared objective population.")}
        let evaluation=try object(manifest["evaluation"],"evaluation")
        try require(evaluation["count"]?.integerValue == partition.validationCount,"Objective evaluation count differs from its recorded held-out rows.")
        return try WorkflowObjectiveEvaluationContext(recipe:recipe,runID:runID,outputSampleShape:outputContract.layout.sample.dimensions,trainingCount:dataLoading?.retainedRows ?? partition.trainingCount,evaluationCount:partition.validationCount,expectedEpochs:expectedEpochs,scalarization:selection.scalarization,device:device,policy:policy,explicitSampleCapture:explicitSampleCapture)
    }
}
