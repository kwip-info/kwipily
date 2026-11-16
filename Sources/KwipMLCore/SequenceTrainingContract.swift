import Foundation

public enum SequenceTrainingContract {
    public static func predictionRowLimit(plan:NamedInputPlan,outputShape:[Int],count:Int)throws->Int {
        guard plan.schemaVersion==2 else {return min(200,count)}
        try plan.validate()
        let elements=(plan.inputs+plan.roles).reduce(0) {$0+$1.elementCount}+outputShape.reduce(1,*)+plan.roles[0].sampleShape[0]
        let allowance=900_000/(4096+32*elements)
        guard allowance>=1 else {throw DataFlowContractError("Sequence prediction geometry exceeds its bounded allowance. Disable optional prediction-row capture.")}
        return min(8,count,allowance)
    }
    public static func validate(plan:NamedInputPlan,recipe:LossRecipe,windows:[String:JSONValue])throws {
        try plan.validate();try recipe.configuration.validate(targetMapping:recipe.targetMapping);try SequenceWindowsConfiguration.validate(windows)
        guard let seq=plan.sequenceObjective,let input=plan.inputs.first(where:{$0.name==seq.inputName}),let target=plan.roles.first,windows["mode"] == .string("partitions"),input.columns==[windows["inputColumn"]!.stringValue!],target.columns==[windows["targetColumn"]!.stringValue!],input.sampleShape==[windows["contextLength"]!.integerValue!],seq.lengthColumn==windows["lengthColumn"]?.stringValue,seq.maskColumn==(windows["incomplete"] == .string("pad") ? windows["maskColumn"]?.stringValue:""),input.paddingValue==windows["paddingID"]?.integerValue,recipe.targetMapping.schemaVersion==2,recipe.targetMapping.inputColumns==Array(plan.inputs.flatMap(\.columns).reduce(into:[String]()) {if !$0.contains($1) {$0.append($1)}}),recipe.targetMapping.roles[0].columns==target.columns,recipe.targetMapping.roles[0].sampleShape==target.sampleShape else {throw DataFlowContractError("Next-token input, target, context, valid length, mask and padding must match the captured Sequence windows configuration.")}
    }
}

extension WorkflowExecutionPlan {
    /// Original token populations are scientific replay/vocabulary evidence, not optional observations.
    public var requiredPreparationNodes:[String] {
        var nodes=Set([training.node,validation.node])
        if namedInputPlan?.sequenceObjective != nil {
            for edge in dataflow.edges where edge.to.node==training.node && ["training","validation"].contains(edge.to.port) {nodes.insert(edge.from.node)}
        }
        return nodes.sorted()
    }
}
