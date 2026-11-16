import Foundation
import KwipMLCore

/// Derived from the captured graph and registry, never the currently open project.
struct TrainedModelInputContract {
    let tensor:TypedTensorContract?
    var namedInputPlan:NamedInputPlan? = nil
    var usesNamedValues=false
    private var bagRequirement:BagWorkflowInputRequirement? = nil
    private var bagMapping:BagInputMapping? = nil
    let columns:[String]
    let objectiveOutput:TypedTensorContract?
    init(graph:GraphDocument,registry:ComponentRegistry,namedInputPlan:NamedInputPlan,recipe:LossRecipe?)throws {
        try namedInputPlan.validate()
        let validation=GraphValidator.validate(graph,registry:registry,context:recipe == nil ? .preparedWorkflow:.explicitObjective)
        guard validation.isValid else {throw WorkflowArchiveError("Captured named model inputs are invalid.")}
        if let names=graph.namedInputs {guard names==Dictionary(uniqueKeysWithValues:namedInputPlan.inputs.map {($0.name,$0.nodeID!)}) else {throw WorkflowArchiveError("Named export declarations differ from the captured graph.")}}
        else {guard namedInputPlan.inputs.count==1 else {throw WorkflowArchiveError("Unnamed model export has multiple mappings.")}}
        for entry in namedInputPlan.inputs {guard let id=entry.nodeID,let contract=validation.tensorContracts[id],contract.dtype.rawValue==entry.dtype,contract.layout.sample.dimensions==entry.sampleShape else {throw WorkflowArchiveError("Named export geometry differs from its graph boundary.")}}
        self.namedInputPlan=namedInputPlan;usesNamedValues=graph.namedInputs != nil
        tensor=validation.tensorContracts[namedInputPlan.inputs[0].nodeID!]
        columns=namedInputPlan.inputs.flatMap(\.columns)
        if let recipe {let output=try WorkflowMappingRequirements.outputContract(graph,registry:registry);try WorkflowMappingRequirements.validateObjective(prediction:output,recipe:recipe);objectiveOutput=output} else {objectiveOutput=nil}
    }
    init(graph:GraphDocument,registry:ComponentRegistry,columns:[String],target:String)throws {
        try TrainingColumnMapping.validate(features:columns,target:target)
        let tensor=try WorkflowMappingRequirements.inputContract(graph,registry:registry)
        guard [.float32,.int64].contains(tensor.dtype),columns.count==tensor.layout.sample.elementCount else {
            throw WorkflowArchiveError("Export column mapping does not match the model input shape and type.")
        }
        self.tensor=tensor
        self.columns=columns
        objectiveOutput=nil
    }
    /// Explicit objective exports retain their actual captured float32 sample shape.
    init(graph:GraphDocument,registry:ComponentRegistry,recipe:LossRecipe)throws {
        try recipe.configuration.validate(targetMapping:recipe.targetMapping)
        let tensor=try WorkflowMappingRequirements.inputContract(graph,registry:registry,context:.explicitObjective)
        guard [ContractDType.float32,.int64].contains(tensor.dtype),recipe.targetMapping.inputColumns.count == tensor.layout.sample.elementCount else {
            throw WorkflowArchiveError("Objective export column mapping does not match its captured model input.")
        }
        self.tensor=tensor
        columns=recipe.targetMapping.inputColumns
        let output=try WorkflowMappingRequirements.outputContract(graph,registry:registry)
        try WorkflowMappingRequirements.validateObjective(prediction:output,recipe:recipe)
        objectiveOutput=output
    }
    init(graph:GraphDocument,registry:ComponentRegistry,bagMapping:BagInputMapping,targetColumn:String?,recipe:LossRecipe?)throws {
        let requirement=try WorkflowMappingRequirements.bagInputRequirement(graph,registry:registry,context:recipe == nil ? .preparedWorkflow:.explicitObjective)
        try bagMapping.validate(sourceConfiguration:requirement.configuration,targetColumn:targetColumn,objectiveMapping:recipe?.targetMapping)
        let output=try WorkflowMappingRequirements.outputContract(graph,registry:registry)
        if let recipe {try WorkflowMappingRequirements.validateObjective(prediction:output,recipe:recipe)}
        else if output.dtype != .float32 || output.layout.sample.dimensions != [2] {throw WorkflowArchiveError("Binary packet export requires two actual model logits.")}
        tensor=nil;columns=bagMapping.indicesColumns;objectiveOutput=recipe == nil ? nil:output
        self.bagRequirement=requirement;self.bagMapping=bagMapping
    }
    var isBag:Bool {bagRequirement != nil}
    var input:[String:JSONValue] {
        if let namedInputPlan {return ["kind":.string("namedInputs"),"plan":.object(namedInputPlan.values),"namedValues":.bool(usesNamedValues),"preprocessing":.string("capturedPreparation")]}
        if let bagRequirement,let bagMapping {
            let ports=Dictionary(uniqueKeysWithValues:bagRequirement.configuration.outputPorts.map {role in
                (role,JSONValue.object(["role":.string(role),"dtype":.string(role == "per_sample_weights" ? "float32":"int64")]))
            })
            return ["kind":.string("bagPacket"),"schemaVersion":1,"sourceID":.string(bagRequirement.sourceNodeID),
                    "configuration":.object(bagRequirement.configuration.values),"ports":.object(ports),"mapping":bagMapping.value,
                    "numEmbeddings":.integer(Int64(bagRequirement.numEmbeddings)),"preprocessing":.string("alreadyPrepared")]
        }
        guard let tensor else {preconditionFailure("Missing captured input domain")}
        return ["dtype":.string(tensor.dtype.rawValue),
         "shape":.array([.string("N")]+tensor.layout.sample.dimensions.map {.integer(Int64($0))}),
         "featureColumns":.array(columns.map(JSONValue.string)),"preprocessing":.string("alreadyPrepared")]
    }
    var pythonShape:String {
        guard let tensor else {return "whole-group records"}
        return "[N, "+tensor.layout.sample.dimensions.map(String.init).joined(separator:", ")+"]"
    }
    var outputReviewSummary:String {
        guard let output=objectiveOutput else {return "Output contains two logits and a class index per row."}
        let shape="[N, "+output.layout.sample.dimensions.map(String.init).joined(separator:", ")+"]"
        return "Output contains numeric model values · \(output.dtype.rawValue) · shape \(shape). Inference returns these values without computing the training score or assigning labels."
    }
    var reviewSummary:String {
        if let namedInputPlan {return "Prepared named inputs · "+namedInputPlan.inputs.map {$0.name+" "+$0.dtype+" "+String(describing:$0.sampleShape)}.joined(separator:" · ")}
        if let bagRequirement {return "Prepared whole-group inputs · \(bagRequirement.configuration.layout) · at most \(bagRequirement.configuration.maximumItemsPerBag) items per group"}
        return "Prepared inputs · \(tensor!.dtype.rawValue) · shape \(pythonShape)"
    }
    var valueGuidance:String {
        if namedInputPlan != nil {return "Use aligned named CPU tensors or prepared records with the captured numeric/token column mappings. Token IDs must belong to the saved vocabulary; length changes follow only the selected explicit policy."}
        if let bagRequirement {
            let c=bagRequirement.configuration
            return "Supply whole-group records with exact integer indices from 0 through \(bagRequirement.numEmbeddings-1); "+(c.layout == "packed" ? "empty groups are allowed, with at most \(c.maximumItemsPerBag) items.":"each group must contain exactly \(c.maximumItemsPerBag) items.")+(c.usesPerSampleWeights ? " Include one finite per_sample_weights value for every item.":" Do not supply item weights.")+" Offsets are constructed by the helper; do not supply them."
        }
        return tensor!.dtype == .int64
            ? "Use exact signed 64-bit integer indices within the model lookup range. Fractional numbers and booleans are rejected."
            : "Use finite 32-bit floating-point values after the captured preparation steps."
    }
}
