import Foundation

/// Bounded exercise controls reuse the production recipe; there is no second initialization engine.
public struct InitializationLearningConfiguration:Equatable {
    public let production:InitializationConfiguration
    public var values:[String:JSONValue] {production.values}
    public init(values:[String:JSONValue])throws {
        let configuration=try InitializationConfiguration(values:values)
        guard configuration.rules.allSatisfy({["first","second"].contains($0.owner.node) && ["weight","bias"].contains($0.owner.parameter)}) else {throw LearningContractError("Choose a weight or bias array on the supplied first or second layer.")}
        production=configuration
    }
}
