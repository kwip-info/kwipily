import Foundation

/// Canonical controls for one training scheduler; never a model-forward operator.
public struct SchedulerConfiguration: Equatable {
    public static let componentIDs: Set<String> = ["scheduler_step_lr", "scheduler_cosine_annealing_lr", "scheduler_reduce_on_plateau", "scheduler_one_cycle_lr", "scheduler_linear_lr"]
    public static let maximumTicks = 10_000
    public static let maximumTraceBytes = 4_000_000
    public let componentID: String
    public let values: [String: JSONValue]
    public var timing: String { componentID == "scheduler_one_cycle_lr" ? "afterOptimizerBatch" : "afterCompletedEpochValidation" }
    public var recipeValue: JSONValue { .object(["schemaVersion": 1, "componentID": .string(componentID), "configurationVersion": 1, "config": .object(values)]) }

    public init(componentID: String, values supplied: [String: JSONValue]) throws {
        var values: [String: JSONValue]
        switch componentID {
        case "scheduler_step_lr": values = ["step_size": 2, "gamma": 0.1]
        case "scheduler_cosine_annealing_lr": values = ["T_max": 10, "eta_min": 0]
        case "scheduler_linear_lr": values = ["start_factor": .number(1.0 / 3), "end_factor": 1, "total_iters": 5]
        case "scheduler_reduce_on_plateau": values = ["metric": .string("validation_loss"), "mode": .string("min"), "factor": 0.1, "patience": 10, "threshold": 0.0001, "threshold_mode": .string("rel"), "cooldown": 0, "min_lr": 0, "eps": 1e-8]
        case "scheduler_one_cycle_lr": values = ["max_lr": 0.1, "planned_total_batches": 10, "pct_start": 0.3, "anneal_strategy": .string("cos"), "cycle_momentum": true, "base_momentum": 0.85, "max_momentum": 0.95, "div_factor": 25, "final_div_factor": 10_000, "three_phase": false]
        default: throw GraphConfigurationError("Choose a supported scheduler recipe.")
        }
        guard Set(supplied.keys).isSubset(of: Set(values.keys)) else {
            throw GraphConfigurationError("Use only this scheduler's declared controls. Timing and continuation state are managed by KwipML.")
        }
        values.merge(supplied) { _, new in new }
        let integerFields: Set<String> = ["step_size", "T_max", "total_iters", "patience", "cooldown", "planned_total_batches"]
        let flagFields: Set<String> = ["cycle_momentum", "three_phase"]
        let choices: [String: Set<String>] = ["metric": ["validation_loss", "validation_accuracy"], "mode": ["min", "max"], "threshold_mode": ["rel", "abs"], "anneal_strategy": ["cos", "linear"]]
        for key in values.keys {
            if integerFields.contains(key) {
                let minimum = ["patience", "cooldown"].contains(key) ? 0 : (key == "planned_total_batches" ? 3 : 1)
                guard case .integer(let number) = values[key], (Int64(minimum)...Int64(Self.maximumTicks)).contains(number) else {
                    throw GraphConfigurationError("\(key) must be a whole number from \(minimum) to \(Self.maximumTicks).")
                }
            } else if flagFields.contains(key) {
                guard values[key]?.boolValue != nil else { throw GraphConfigurationError("\(key) must be a Boolean switch.") }
            } else if let allowed = choices[key] {
                guard let choice = values[key]?.stringValue, allowed.contains(choice) else { throw GraphConfigurationError("Choose a supported \(key) value.") }
            } else {
                let fractions: Set<String> = ["start_factor", "end_factor", "factor", "pct_start", "base_momentum", "max_momentum"]
                let minimum = ["div_factor", "final_div_factor"].contains(key) ? OptimizerConfiguration.epsilonMinimum : 0
                let maximum = fractions.contains(key) ? 1 : OptimizerConfiguration.scalarMaximum
                guard let number = values[key]?.doubleValue, number.isFinite, (minimum...maximum).contains(number) else {
                    throw GraphConfigurationError("\(key) must be a finite number from \(minimum) to \(maximum).")
                }
                if ["start_factor", "end_factor", "pct_start"].contains(key) && number == 0 {
                    throw GraphConfigurationError("\(key) must be greater than zero.")
                }
                if ["factor", "pct_start"].contains(key) && number == 1 {
                    throw GraphConfigurationError("\(key) must be below one.")
                }
                values[key] = .number(number)
            }
        }
        if componentID == "scheduler_one_cycle_lr" {
            let total = values["planned_total_batches"]!.doubleValue!, fraction = values["pct_start"]!.doubleValue!
            let first = fraction * total - 1
            let second = values["three_phase"] == .bool(true) ? 2 * fraction * total - 2 : total - 1
            guard first > 0, second > first, values["three_phase"] != .bool(true) || total - 1 > second else {
                throw GraphConfigurationError("Choose a cycle length and rising fraction that give every phase at least a nonzero span.")
            }
            guard values["base_momentum"]!.doubleValue! <= values["max_momentum"]!.doubleValue! else {
                throw GraphConfigurationError("Starting momentum cannot exceed maximum momentum.")
            }
            let initial = values["max_lr"]!.doubleValue! / values["div_factor"]!.doubleValue!
            let minimum = initial / values["final_div_factor"]!.doubleValue!
            guard [initial, minimum].allSatisfy({ $0.isFinite && (0...OptimizerConfiguration.scalarMaximum).contains($0) }) else {
                throw GraphConfigurationError("The peak rate and division controls exceed supported rate arithmetic.")
            }
        }
        self.componentID = componentID; self.values = values
    }

    public func validateOptimizer(_ optimizer:OptimizerConfiguration)throws {
        if componentID == "scheduler_cosine_annealing_lr",values["eta_min"]!.doubleValue!>optimizer.learningRate {throw GraphConfigurationError("The cosine minimum cannot exceed the optimizer rate for a decreasing curve.")}
        guard componentID == "scheduler_one_cycle_lr",values["cycle_momentum"] == true else {return}
        if optimizer.componentID == "optimizer_adagrad" {throw GraphConfigurationError("Turn off momentum cycling when using Adagrad.")}
        if ["optimizer_adam","optimizer_adamw"].contains(optimizer.componentID),values["max_momentum"]!.doubleValue!>0.9999999403953552 {throw GraphConfigurationError("Scheduled first-average memory must remain below one for Adam and AdamW.")}
        if optimizer.componentID == "optimizer_sgd",optimizer.values["nesterov"] == true,values["base_momentum"]!.doubleValue!<=0 {throw GraphConfigurationError("Look ahead requires positive minimum scheduled momentum.")}
    }

    public static func decodeRecipe(_ value: JSONValue) throws -> SchedulerConfiguration {
        guard case .object(let envelope) = value,
              Set(envelope.keys) == ["schemaVersion", "componentID", "configurationVersion", "config"],
              envelope["schemaVersion"] == .integer(1), envelope["configurationVersion"] == .integer(1),
              let id = envelope["componentID"]?.stringValue, case .object(let config) = envelope["config"] else {
            throw GraphConfigurationError("Scheduler recipes require the supported version, component and declared controls only.")
        }
        return try .init(componentID: id, values: config)
    }
}

public struct SchedulerRecipe: Equatable {
    public let configuration: SchedulerConfiguration
    public let source: WorkflowNodeAddress
    public init(configuration: SchedulerConfiguration, source: WorkflowNodeAddress) {
        self.configuration = configuration; self.source = source
    }
}
