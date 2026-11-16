import Foundation
import KwipMLCore

/// Coordinate preview exceptions require captured source and actual prepared whole bags.
enum WorkflowBagActivationEvidence {
    private static func require(_ condition:Bool,_ message:String)throws {
        guard condition else {throw WorkflowPhaseError(message)}
    }
    private static func object(_ value:JSONValue?)throws->[String:JSONValue] {
        try WorkflowBagPreparedEvidence.object(value)
    }
    static func validate(_ previews:[String:JSONValue],graph:GraphDocument,registry:ComponentRegistry,
                         sampleCapture:TrainingSampleCapturePolicy?,context c:WorkflowBagArtifactContext)throws {
        try require(Set(previews.keys) == Set(graph.nodes.map(\.id)) && sampleCapture == (c.explicitSampleCapture ? c.policy:nil),"Bag activation nodes or capture policy differ from the captured source.")
        let validationContext:GraphValidationContext
        switch c.target {case .binary:validationContext = .preparedWorkflow;case .objective:validationContext = .explicitObjective}
        let requirement=try WorkflowMappingRequirements.bagInputRequirement(graph,registry:registry,context:validationContext)
        try require(requirement == c.requirement,"Bag activation source differs from its captured graph requirement.")
        let validation=try GraphValidator.validate(graph,registry:registry,context:validationContext)
        try require(validation.issues.isEmpty,"Captured graph cannot resolve bag activation contracts.")
        let count=min(8,c.preparation.validation.count)
        let observations=Array(c.preparation.validation.prefix.prefix(count))
        try require(count>0 && observations.count == count,"Bag activation evidence requires the actual prepared validation prefix.")
        let ids=observations.flatMap { $0.input.indices }
        let weights=observations.flatMap { $0.input.perSampleWeights ?? [] }
        var offsets:[Int64]=[],end:Int64=0
        for observation in observations {offsets.append(end);end += Int64(observation.input.indices.count)}
        if c.requirement.configuration.includeLastOffset {offsets.append(end)}
        let status:JSONValue?=sampleCapture.map {.string($0.activationValues ? "captured":"omittedByPolicy")}
        func tensor(_ value:JSONValue?,shape:[Int],dtype:ContractDType,expected:[JSONValue]?=nil,coordinate:Bool=false,binaryInteger:Bool=false)throws {
            var item=try object(value)
            let heatmap=item.removeValue(forKey:"heatmap")
            let playback=item.removeValue(forKey:"heatmapPlayback")
            if c.policy.heatmapFrameLimit>0 {
                guard let playback else {throw WorkflowPhaseError("Missing requested heatmap playback evidence.")}
                _=try TensorHeatmapPlayback(value:playback,maximumFrames:c.policy.heatmapFrameLimit)
            } else {try require(playback==nil,"Undeclared heatmap playback capture.")}
            if c.policy.tensorHeatmaps {
                guard let heatmap else {throw WorkflowPhaseError("Missing requested tensor heatmap evidence.")}
                _=try TensorHeatmapSummary(value:heatmap,expectedShape:shape)
            } else {try require(heatmap==nil,"Undeclared tensor heatmap capture.")}
            try require(Set(item.keys) == ["shape","dtype","values"] && item["shape"] == .array(shape.map {.integer(Int64($0))}) && item["dtype"] == .string(dtype.rawValue),"Bag activation shape, dtype or fields differ from the resolved contract.")
            var elements=1
            for dimension in shape {
                try require(dimension>=0 && (coordinate || dimension>0) && dimension<=100_000_000,"Zero extents are reserved for proven empty bag coordinates.")
                if dimension == 0 {elements=0} else {try require(elements<=100_000_000/dimension,"Bag activation element budget exceeded.");elements *= dimension}
            }
            let values=try WorkflowBagPreparedEvidence.array(item["values"])
            let visible=c.policy.activationValues ? min(8,elements):0
            try require(values.count == visible,"Bag activation capture prefix or permission differs.")
            for value in values {
                if dtype == .int64 {guard case .integer=value else {throw WorkflowPhaseError("Bag integer activation values must be exact integers.")};if binaryInteger {try require(value == 0 || value == 1,"Attention mask activations must remain binary whole numbers.")}}
                else {_ = try WorkflowBagPreparedEvidence.finiteFloat(value)}
            }
            if let expected {
                try require(expected.count == elements,"Prepared coordinate evidence does not fill its shape.")
                for (actual,wanted) in zip(values,expected.prefix(visible)) {
                    if dtype == .int64 {try require(actual == wanted,"Bag coordinate activation differs from the actual prepared prefix.")}
                    else {try require(WorkflowBagPreparedEvidence.finiteFloat(actual) == WorkflowBagPreparedEvidence.finiteFloat(wanted),"Bag weight activation differs from the actual prepared prefix.")}
                }
            }
        }
        for node in graph.nodes {
            var item=try object(previews[node.id])
            try require(item.removeValue(forKey:"captureStatus") == status,"Bag activation capture status differs.")
            if node.id == requirement.sourceNodeID {
                try require(node.type == "bag_input" && Set(item.keys) == ["outputs"],"Bag coordinates require the actual captured source node.")
                let outputs=try object(item["outputs"])
                try require(Set(outputs.keys) == Set(requirement.configuration.outputPorts),"Bag activation named coordinate roles differ.")
                let fixed=requirement.configuration.layout == "fixed"
                let shape=fixed ? [count,requirement.configuration.maximumItemsPerBag]:[ids.count]
                try tensor(outputs["indices"],shape:shape,dtype:.int64,expected:ids.map(JSONValue.integer),coordinate:true)
                if !fixed {try tensor(outputs["offsets"],shape:[offsets.count],dtype:.int64,expected:offsets.map(JSONValue.integer),coordinate:true)}
                if requirement.configuration.usesPerSampleWeights {
                    try tensor(outputs["per_sample_weights"],shape:shape,dtype:.float32,expected:weights.map {.number(Double($0))},coordinate:true)
                }
            } else {
                let contracts=validation.outputTensorContracts.filter {$0.key.nodeID == node.id}
                try require(!contracts.isEmpty,"Bag computed preview has no captured ordinary tensor contract.")
                if case .object(let outputs)=item["outputs"] {
                    try require(Set(item.keys) == ["outputs"] && Set(outputs.keys) == Set(contracts.keys.map(\.portID)),"Bag computed named outputs differ from captured metadata.")
                    for (address,contract) in contracts {try tensor(outputs[address.portID],shape:[count]+contract.layout.sample.dimensions,dtype:contract.dtype,binaryInteger:AttentionConfiguration.componentIDs.contains(node.type))}
                } else {
                    guard contracts.count == 1,let contract=contracts[.init(node.id,"out")] else {throw WorkflowPhaseError("Bag computed preview requires its actual named outputs.")}
                    try tensor(.object(item),shape:[count]+contract.layout.sample.dimensions,dtype:contract.dtype,binaryInteger:AttentionConfiguration.componentIDs.contains(node.type))
                }
            }
        }
    }
    /// Reuses schema3 row admission. This synthetic live-envelope hash is not an archive file pin.
    static func presentation(_ event:[String:JSONValue],context c:WorkflowBagArtifactContext,
                             objectiveContext:WorkflowObjectiveEvaluationContext?)throws {
        try require(event["runID"] == .string(c.runID) && event["device"] == .string(c.device.rawValue) && event["evaluationCount"]?.integerValue == c.preparation.validation.count && event["predictions"] == .array([]),"Live bag completion identity or row count differs.")
        let policy=try WorkflowBagPreparedEvidence.json(c.policy)
        try require(event["sampleCapture"] == (c.explicitSampleCapture ? policy:nil),"Live bag capture permission differs.")
        guard let loss=event["validationLoss"]?.doubleValue,loss.isFinite else {throw WorkflowPhaseError("Live bag score is missing or nonfinite.")}
        switch c.target {
        case .binary:
            try require(objectiveContext == nil && event["objective"] == nil,"Binary bag completion cannot claim an explicit objective.")
            guard let accuracy=event["accuracy"]?.doubleValue,accuracy.isFinite,(0...1).contains(accuracy) else {throw WorkflowPhaseError("Binary bag completion requires measured accuracy.")}
        case .objective(let selected,let recipe):
            guard let objectiveContext else {throw WorkflowPhaseError("Bag objective completion requires its captured evaluation context.")}
            try require(objectiveContext.recipe == recipe && objectiveContext.runID == c.runID && objectiveContext.outputSampleShape == c.outputContract.layout.sample.dimensions && objectiveContext.trainingCount == c.preparation.training.count && objectiveContext.evaluationCount == c.preparation.validation.count && objectiveContext.scalarization == selected.scalarization && objectiveContext.device == c.device && objectiveContext.policy == c.policy && objectiveContext.explicitSampleCapture == c.explicitSampleCapture,"Bag objective evaluation context differs from captured evidence.")
            let evaluation=try WorkflowObjectiveArtifacts.evaluation(event["objective"] ?? .null,context:objectiveContext)
            try require(abs(loss-evaluation.effectiveScalar)<=max(1e-6,max(abs(loss),abs(evaluation.effectiveScalar))*1e-5),"Live bag objective score differs from its aggregate.")
            if let expected=evaluation.accuracy {
                guard let actual=event["accuracy"]?.doubleValue,actual.isFinite,abs(actual-expected)<=1e-6 else {throw WorkflowPhaseError("Live bag classification differs from measured counts.")}
            } else {try require(event["accuracy"] == nil,"This bag objective has no measured accuracy.")}
        }
        guard let input=event["predictionInput"],let rows=event["predictionRows"] else {throw WorkflowPhaseError("Live bag prediction rows are missing.")}
        var artifact:[String:JSONValue]=["schemaVersion":3,"runID":.string(c.runID),"selection":.string("firstRowsOfValidationPartition"),"evaluationCount":.integer(Int64(c.preparation.validation.count)),"predictionInput":input,"predictionRows":rows]
        if c.explicitSampleCapture {artifact["captureStatus"] = .string(c.policy.predictionRows ? "captured":"omittedByPolicy")}
        let bytes=try WorkflowPhaseArtifacts.encode(artifact)
        let metadata:JSONValue = .object(["schemaVersion":3,"file":.string("predictions.json"),"bytes":.integer(Int64(bytes.count)),"sha256":.string(WorkflowPhaseArtifacts.hash(bytes))])
        _ = try WorkflowBagArtifacts.predictions(data:bytes,metadata:metadata,context:c)
    }
}
