import Foundation

/// Canonical training control settings; never a model-forward tensor operation.
public struct OptimizerConfiguration: Equatable, Sendable {
    public static let componentIDs: Set<String> = ["optimizer_sgd", "optimizer_adam", "optimizer_adamw", "optimizer_rmsprop", "optimizer_adagrad"]
    public static let scalarMaximum = 3.4028234663852886e38
    public static let epsilonMinimum = 1.1754943508222875e-38
    public static let betaMaximum = 0.9999999403953552
    public let componentID: String
    public let values: [String: JSONValue]
    public var learningRate: Double { values["lr"]!.doubleValue! }
    public var recipeValue: JSONValue { .object(["schemaVersion": 1, "componentID": .string(componentID), "configurationVersion": 1, "config": .object(values)]) }
    public init(componentID: String, values supplied: [String: JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else { throw GraphConfigurationError("Choose a supported optimizer recipe.") }
        var values: [String: JSONValue] = ["lr": .number(componentID == "optimizer_rmsprop" || componentID == "optimizer_adagrad" ? 0.01 : 0.001), "weight_decay": .number(componentID == "optimizer_adamw" ? 0.01 : 0), "maximize": .bool(false)]
        switch componentID {
        case "optimizer_sgd": values.merge(["momentum": 0, "dampening": 0, "nesterov": false]) { _, v in v }
        case "optimizer_adam", "optimizer_adamw":
            values.merge(["beta1": 0.9, "beta2": 0.999, "eps": 1e-8, "amsgrad": false]) { _, v in v }
            if componentID == "optimizer_adam" { values["decoupled_weight_decay"] = false }
        case "optimizer_rmsprop": values.merge(["alpha": 0.99, "eps": 1e-8, "momentum": 0, "centered": false]) { _, v in v }
        default: values.merge(["lr_decay": 0, "initial_accumulator_value": 0, "eps": 1e-10]) { _, v in v }
        }
        guard Set(supplied.keys).isSubset(of: Set(values.keys)) else { throw GraphConfigurationError("Use only this optimizer's declared controls; execution flags and parameter groups are managed by KwipML.") }
        values.merge(supplied) { _, v in v }
        let flags: Set<String> = ["maximize", "nesterov", "amsgrad", "decoupled_weight_decay", "centered"]
        for key in values.keys {
            if flags.contains(key) {
                guard let flag = values[key]?.boolValue else { throw GraphConfigurationError("\(key) must be a Boolean switch.") }
                values[key] = .bool(flag)
            } else {
                let minimum = key == "eps" ? Self.epsilonMinimum : 0
                let maximum: Double = ["beta1", "beta2"].contains(key) ? Self.betaMaximum : (["momentum", "dampening", "alpha"].contains(key) ? 1 : Self.scalarMaximum)
                guard let number = values[key]?.doubleValue, number.isFinite, (minimum...maximum).contains(number) else {
                    throw GraphConfigurationError("\(key) must be a finite number from \(minimum) to \(maximum).")
                }
                values[key] = .number(number)
            }
        }
        if values["nesterov"] == .bool(true) {
            guard values["momentum"]!.doubleValue! > 0, values["dampening"]!.doubleValue! == 0 else { throw GraphConfigurationError("Look ahead requires positive Momentum and zero Momentum dampening.") }
        }
        self.componentID = componentID; self.values = values
    }
    public static func decodeRecipe(_ value: JSONValue) throws -> OptimizerConfiguration {
        guard case .object(let recipe) = value, Set(recipe.keys) == ["schemaVersion", "componentID", "configurationVersion", "config"], recipe["schemaVersion"] == .integer(1), recipe["configurationVersion"] == .integer(1), let id = recipe["componentID"]?.stringValue, case .object(let config) = recipe["config"] else {
            throw GraphConfigurationError("Optimizer recipes require the supported version, component and declared configuration only.")
        }
        return try .init(componentID: id, values: config)
    }
}

/// A selected recipe's canonical settings and qualified canvas origin.
public struct OptimizerRecipe: Equatable {
    public let configuration: OptimizerConfiguration
    public let source: WorkflowNodeAddress
    public init(configuration: OptimizerConfiguration, source: WorkflowNodeAddress) {
        self.configuration = configuration; self.source = source
    }
}
