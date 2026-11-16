import Foundation

/// Bounded lesson controls use the genuine production recipe validators.
public struct GradientLearningConfiguration:Equatable {
    public let gradient:GradientPolicyConfiguration
    public let trainability:TrainabilityConfiguration
    public init(gradientValues:[String:JSONValue],trainabilityValues:[String:JSONValue],task:GradientLearningTask)throws {
        gradient=try GradientPolicyConfiguration(values:gradientValues)
        trainability=try TrainabilityConfiguration(values:trainabilityValues)
        try gradient.validateObjectiveCompatibility(reduction:"mean")
        for raw in trainability.owners {
            guard case .object(let owner)=raw,["first","second"].contains(owner["node"]?.stringValue ?? ""),["weight","bias"].contains(owner["parameter"]?.stringValue ?? "") else {throw LearningContractError("Choose a weight or bias array on the supplied first or second layer.")}
        }
        for raw in gradient.requests {
            guard case .object(let request)=raw,case .object(let output)=request["output"],["first","second"].contains(output["node"]?.stringValue ?? ""),output["port"]?.stringValue == "out",case .array(let targets)=request["targets"] else {throw LearningContractError("Ask about a genuine out port on the supplied first or second layer.")}
            for target in targets {
                guard case .object(let value)=target else {throw LearningContractError("Choose an input or named learned array.")}
                if value["kind"]?.stringValue == "input" {guard value["node"]?.stringValue == "input",value["port"]?.stringValue == "out" else {throw LearningContractError("Select the supplied input's out port.")}}
                else {guard ["first","second"].contains(value["node"]?.stringValue ?? ""),["weight","bias"].contains(value["parameter"]?.stringValue ?? "") else {throw LearningContractError("Choose a weight or bias array on the supplied model.")}}
            }
        }
        if task == .keepSettings || task == .nextRun {
            let supplied=try GradientPolicyConfiguration(values:task.gradientValues)
            guard gradient == supplied else {throw LearningContractError("Keep the supplied influence/update policy while choosing which settings can learn.")}
        } else {
            guard trainability == (try TrainabilityConfiguration(values:task.trainabilityValues)) else {throw LearningContractError("Keep the supplied trainability context while editing the gradient policy.")}
        }
    }
    public var trainableOwnerCount:Int {
        var flags=["first.weight":true,"first.bias":true,"second.weight":true,"second.bias":true]
        for raw in trainability.owners {if case .object(let owner)=raw,let node=owner["node"]?.stringValue,let parameter=owner["parameter"]?.stringValue,let mode=owner["mode"]?.stringValue,mode != "inherit" {flags[node+"."+parameter] = mode == "trainable"}}
        return flags.values.filter {$0}.count
    }
}
