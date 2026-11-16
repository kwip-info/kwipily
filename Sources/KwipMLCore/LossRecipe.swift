import Foundation

/// Selected objective, qualified canvas source and explicit source-column role binding.
public struct LossRecipe: Equatable {
    public let configuration: LossConfiguration
    public let source: WorkflowNodeAddress
    public let targetMapping: LossTargetMapping
    public init(configuration: LossConfiguration, source: WorkflowNodeAddress, targetMapping: LossTargetMapping) throws {
        try configuration.validate(targetMapping:targetMapping)
        self.configuration=configuration;self.source=source;self.targetMapping=targetMapping
    }
}
