import Foundation
import KwipMLCore

struct BagObservationInput: Equatable {
    let indices: [Int64]
    let perSampleWeights: [Float]?
}
enum BagArtifactTarget: Equatable {
    case binary(String)
    case objective(PreparedObjectiveSelection, LossRecipe)
}
struct WorkflowBagArtifactContext {
    let requirement: BagWorkflowInputRequirement
    let mapping: BagInputMapping
    let preparation: VerifiedBagPreparation
    let runID: String
    let device: ContractDevice
    let outputContract: TypedTensorContract
    let target: BagArtifactTarget
    let policy: TrainingSampleCapturePolicy
    let explicitSampleCapture: Bool
}
struct BagPredictionRow: Equatable {
    let input: BagObservationInput
    let label: Int?
    let predicted: Int?
    let output: ObjectiveResultTensor?
    let targets: [String: ObjectiveResultTensor]
    let objectiveOutput: ObjectiveResultTensor?
    let decoded: ObjectiveResultTensor?
}
struct BagPredictionPreview: Equatable {
    let runID: String
    let evaluationCount: Int
    let inputDescriptor: JSONValue
    let captureStatus: ObjectiveCaptureStatus
    let rows: [BagPredictionRow]
}
/// Packet rows have explicit coordinate semantics; empty bags are not empty tensors.
/// The two Python semantic digests and Torch tensor digest are reported identities:
/// actual table bytes/cells are pinned here; those digests are not recomputed natively.
enum WorkflowBagArtifacts {
    static func objectiveEvaluationContext(context: WorkflowBagArtifactContext, expectedEpochs: Int, trainingCount:Int?=nil) throws -> WorkflowObjectiveEvaluationContext {
        try checkedContext(context)
        guard case .objective(let selected,let recipe)=context.target else {
            throw WorkflowPhaseError("A binary bag result has no explicit objective evaluation context.")
        }
        return try .init(recipe:recipe,runID:context.runID,outputSampleShape:context.outputContract.layout.sample.dimensions,
                         trainingCount:trainingCount ?? context.preparation.training.count,evaluationCount:context.preparation.validation.count,
                         expectedEpochs:expectedEpochs,scalarization:selected.scalarization,device:context.device,
                         policy:context.policy,explicitSampleCapture:context.explicitSampleCapture)
    }
    private static func require(_ value:Bool,_ message:String)throws {try WorkflowBagPreparedEvidence.require(value,message)}
    private static func object(_ value:JSONValue?)throws->[String:JSONValue] {try WorkflowBagPreparedEvidence.object(value)}
    private static func array(_ value:JSONValue?)throws->[JSONValue] {try WorkflowBagPreparedEvidence.array(value)}
    private static func matches(_ actual: JSONValue?, _ expected: JSONValue) throws -> Bool {
        guard let actual else { return false }
        return try WorkflowPhaseArtifacts.encode(actual) == WorkflowPhaseArtifacts.encode(expected)
    }
    private static func modelInput(_ c:WorkflowBagArtifactContext)->JSONValue {
        let config=c.requirement.configuration
        return .object(["schemaVersion":1,"kind":.string("embeddingBagPacket"),"layout":.string(config.layout),"maximumItemsPerBag":.integer(Int64(config.maximumItemsPerBag)),"perSampleWeights":.bool(config.usesPerSampleWeights),"includeLastOffset":.bool(config.includeLastOffset),"indicesDType":.string("int64"),"offsetsDType":config.layout == "packed" ? .string("int64"):.null,"weightDType":config.usesPerSampleWeights ? .string("float32"):.null,"mapping":c.mapping.value])
    }
    private static func descriptor(_ c:WorkflowBagArtifactContext)->JSONValue {
        var ports:[String:JSONValue]=[:]
        for port in c.requirement.configuration.outputPorts {
            ports[port] = .object(["role":.string(port),"dtype":.string(port == "per_sample_weights" ? "float32":"int64")])
        }
        return .object(["kind":.string("bagPacket"),"schemaVersion":1,"sourceID":.string(c.requirement.sourceNodeID),"configuration":.object(c.requirement.configuration.values),"ports":.object(ports)])
    }
    private static func checkedContext(_ c:WorkflowBagArtifactContext)throws {
        try require(!c.runID.isEmpty && [.cpu,.mps].contains(c.device) && c.outputContract.dtype == .float32,"Bag result context has invalid identity, output dtype or device.")
        try require(c.preparation.training.count>0 && c.preparation.validation.count>0 && c.preparation.validation.prefix.count == min(200,c.preparation.validation.count),"Bag context requires complete verified prepared prefix.")
        switch c.target {
        case .binary(let column):
            guard case .legacy(let selected)=c.preparation.selection else {throw WorkflowPhaseError("Binary bag result requires binary preparation.")}
            try require(column == selected.targetColumn && c.outputContract.layout.sample.dimensions == [2],"Binary bag target or output differs.")
        case .objective(let selected,let recipe):
            try require(c.preparation.selection == .objective(selected) && selected.objectiveRecipe == recipe.configuration.recipeValue && selected.targetMapping == recipe.targetMapping,"Objective bag result differs from selected recipe or role mapping.")
        }
    }
    static func validateIdentity(manifest: [String: JSONValue], context c: WorkflowBagArtifactContext) throws {
        try checkedContext(c)
        let p=try object(manifest["provenance"]),checkpoint=try object(manifest["checkpoint"]),contract=try object(checkpoint["contract"]),data=try object(contract["data"])
        let bag=try object(data["bagInput"]),sources=try object(p["bagSources"])
        let runtime=try object(manifest["runtime"]),savedRuntime=try object(contract["runtime"])
        try require(runtime["device"] == .string(c.device.rawValue) && savedRuntime["device"] == runtime["device"] && runtime["torch"] == savedRuntime["torch"] && runtime["torch"]?.stringValue?.isEmpty == false,"Bag runtime/device identity differs.")
        try require(manifest["sampleCapture"] == (c.explicitSampleCapture ? (try WorkflowBagPreparedEvidence.json(c.policy)):nil),"Bag manifest capture permission differs.")
        try require(manifest["runID"] == .string(c.runID) && p["dataset"] == .string("prepared-dataflow-bags-v1") && p["modelInput"] == modelInput(c),"Bag model-input provenance differs from its actual captured source.")
        try require(Set(bag.keys) == ["schemaVersion","modelInput","sources"] && bag["schemaVersion"] == 1 && bag["modelInput"] == p["modelInput"] && bag["sources"] == p["bagSources"] && Set(sources.keys) == ["training","validation"],"Checkpoint bag identity differs from prepared provenance.")
        try require(p["trainingSamples"]?.integerValue == c.preparation.training.count && p["validationSamples"]?.integerValue == c.preparation.validation.count && p["samples"]?.integerValue == c.preparation.training.count+c.preparation.validation.count && p["sourceSamples"] == .null && p["split"] == .string("prepared partitions; no resplitting") && p["featureColumns"] == .array(c.mapping.indicesColumns.map(JSONValue.string)),"Bag provenance counts, order or split policy differ.")
        try require(p["preparation"] == .object(c.preparation.preparationIdentity) && data["preparation"] == p["preparation"],"Bag prepared file identity differs from the pinned inventory.")
        for (role,partition) in [("training",c.preparation.training),("validation",c.preparation.validation)] {
            var expected=partition.sourceIdentity
            if case .objective = c.target {
                var target=try object(expected["targetMapping"])
                target["predictionShape"] = .array(c.outputContract.layout.sample.dimensions.map { .integer(Int64($0)) })
                expected["targetMapping"] = .object(target)
            }
            var actual=try object(sources[role])
            for key in ["tableDigest","mappingDigest"] {
                try require(actual[key]?.stringValue.map(WorkflowArchiveCodec.isHash) == true,"Bag reported semantic digest is malformed.");actual.removeValue(forKey:key)
            }
            try require(try matches(.object(actual),.object(expected)) && p[role+"RowIDs"] == .array(partition.rowIDs.map(JSONValue.string)),"Bag source identity or actual prepared rows differ.")
        }
        try require(p["sourceHash"] == c.preparation.training.lineage["sourceDigest"] && data["sourceDigest"] == p["sourceHash"],"Bag source digest differs.")
        // This JSON has only hash strings, so canonical JSON spelling is portable.
        let split:[String:JSONValue]=["training":c.preparation.training.lineage["membershipDigest"]!,"validation":c.preparation.validation.lineage["membershipDigest"]!,"partition":c.preparation.training.lineage["partitionDigest"]!]
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        let splitHash=WorkflowPhaseArtifacts.hash(try encoder.encode(split))
        try require(p["splitHash"] == .string(splitHash) && data["splitDigest"] == p["splitHash"] && data["tensorDigest"]?.stringValue.map(WorkflowArchiveCodec.isHash) == true,"Bag split or reported tensor digest is invalid.")
        switch c.target {
        case .binary(let column):try require(p["targetColumn"] == .string(column) && p["objectiveMapping"] == nil && p["objectiveRecipe"] == nil,"Binary target identity differs.")
        case .objective(let selected,let recipe):try require(p["objectiveMapping"] == (try WorkflowBagPreparedEvidence.json(recipe.targetMapping)) && (try matches(p["objectiveRecipe"],selected.objectiveRecipe)) && p["targetColumn"] == nil,"Objective target identity differs.")
        }
    }
    private static func rawShape(_ c:WorkflowBagArtifactContext,recipe:LossRecipe)throws->[Int] {
        let op=recipe.configuration.componentID,v=recipe.configuration.values,out=c.outputContract.layout.sample.dimensions
        if ["loss_cross_entropy","loss_nll"].contains(op) {
            let axis=v["class_axis"]!.integerValue!,rank=out.count+1,actual=axis<0 ? rank+axis:axis
            try require(actual>0 && actual<rank,"Class direction is outside the actual output.")
            var result=out;result.remove(at:actual-1);return result
        }
        if ["loss_ctc","loss_triplet_margin"].contains(op) {return []}
        if ["loss_mse","loss_l1"].contains(op),let target=recipe.targetMapping.roles.first(where:{$0.role == "target"}) {
            let rank=max(out.count,target.sampleShape.count),a=Array(repeating:1,count:rank-out.count)+out,b=Array(repeating:1,count:rank-target.sampleShape.count)+target.sampleShape
            try require(zip(a,b).allSatisfy {$0 == $1 || $0 == 1 || $1 == 1},"Objective target cannot broadcast to produced values.")
            return zip(a,b).map {max($0,$1)}
        }
        return out
    }
    static func predictions(data: Data, metadata: JSONValue, context c: WorkflowBagArtifactContext) throws -> BagPredictionPreview {
        try checkedContext(c)
        let m=try object(metadata)
        try require(!data.isEmpty && data.count<=1_000_000 && Set(m.keys) == ["schemaVersion","file","bytes","sha256"] && m["schemaVersion"] == 3 && m["file"] == .string("predictions.json") && m["bytes"]?.integerValue == data.count && m["sha256"] == .string(WorkflowPhaseArtifacts.hash(data)),"Bag prediction artifact differs from its bounded byte pin.")
        let p=try object(JSONDecoder().decode(JSONValue.self,from:data))
        let keys:Set<String>=["schemaVersion","runID","selection","evaluationCount","predictionInput","predictionRows"]
        try require(Set(p.keys) == (c.explicitSampleCapture ? keys.union(["captureStatus"]):keys) && p["schemaVersion"] == 3 && p["runID"] == .string(c.runID) && p["selection"] == .string("firstRowsOfValidationPartition") && p["evaluationCount"]?.integerValue == c.preparation.validation.count && p["predictionInput"] == descriptor(c),"Bag prediction geometry, source or selection differs.")
        try require(p["captureStatus"] == (c.explicitSampleCapture ? .string(c.policy.predictionRows ? "captured":"omittedByPolicy"):nil),"Bag row privacy differs from the explicit capture permission.")
        let rows=try array(p["predictionRows"])
        try require(rows.count == (c.policy.predictionRows ? min(200,c.preparation.validation.count):0),"Bag row capture count differs from permission.")
        let parsed=try rows.enumerated().map {index,value->BagPredictionRow in
            let row=try object(value),input=try object(row["input"]),expected=c.preparation.validation.prefix[index]
            let inputKeys:Set<String>=c.requirement.configuration.usesPerSampleWeights ? ["indices","per_sample_weights"]:["indices"]
            try require(Set(input.keys) == inputKeys,"Bag input contains missing or foreign coordinate roles.")
            let ids=try array(input["indices"]).map {value->Int64 in guard case .integer(let id)=value,id>=0,id<Int64(c.requirement.numEmbeddings) else {throw WorkflowPhaseError("Bag prediction ID is invalid.")};return id}
            let weights=try input["per_sample_weights"].map {try array($0).map {try WorkflowBagPreparedEvidence.finiteFloat($0)}}
            let typed=BagObservationInput(indices:ids,perSampleWeights:weights)
            try require(typed == expected.input,"Bag prediction row differs from its actual prepared observation.")
            switch c.target {
            case .binary:
                try require(Set(row.keys) == ["input","label","predicted"] && row["label"]?.integerValue == expected.binaryLabel,"Bag prediction binary target differs.")
                guard let predicted=row["predicted"]?.integerValue,[0,1].contains(predicted) else {throw WorkflowPhaseError("Bag predicted binary class is invalid.")}
                return .init(input:typed,label:expected.binaryLabel,predicted:predicted,output:nil,targets:[:],objectiveOutput:nil,decoded:nil)
            case .objective(_,let recipe):
                let raw=try rawShape(c,recipe:recipe),reduction=recipe.configuration.values["reduction"]!.stringValue!
                let classified=["loss_cross_entropy","loss_nll"].contains(recipe.configuration.componentID) && recipe.targetMapping.roles.first(where:{$0.role == "target"})?.dtype == .int64
                var rowKeys:Set<String>=["input","output","targets"]
                if reduction == "none" {rowKeys.insert("objectiveOutput")};if classified {rowKeys.insert("decoded")}
                try require(Set(row.keys) == rowKeys,"Bag objective row fields differ from the selected reduction/measurement.")
                let targetValues=try object(row["targets"]);try require(Set(targetValues.keys) == Set(recipe.targetMapping.roles.map(\.role)),"Bag objective desired-value roles differ.")
                var targets:[String:ObjectiveResultTensor]=[:]
                for role in recipe.targetMapping.roles {
                    let tensor=try ObjectiveResultTensor(json:targetValues[role.role]!,shape:role.sampleShape,dtype:role.dtype)
                    let actual=try ObjectiveResultTensor(json:expected.targets[role.role]!,shape:role.sampleShape,dtype:role.dtype)
                    try require(tensor == actual,"Bag desired values differ from actual prepared cells.");targets[role.role]=tensor
                }
                return try .init(input:typed,label:nil,predicted:nil,output:ObjectiveResultTensor(json:row["output"]!,shape:c.outputContract.layout.sample.dimensions,dtype:.float32),targets:targets,objectiveOutput:row["objectiveOutput"].map {try ObjectiveResultTensor(json:$0,shape:raw,dtype:.float32)},decoded:row["decoded"].map {try ObjectiveResultTensor(json:$0,shape:raw,dtype:.int64)})
            }
        }
        return .init(runID:c.runID,evaluationCount:c.preparation.validation.count,inputDescriptor:descriptor(c),captureStatus:c.policy.predictionRows ? .captured:.omittedByPolicy,rows:parsed)
    }
}
