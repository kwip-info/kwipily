import Foundation

/// Independent scalar reference for the reviewed forward-construction objective.
public struct SigmoidLessonReference: Equatable {
    public init() {}
    public var instructions: String {
        "Build a flow that transforms each incoming number separately with Sigmoid, preserving two-feature rows and their order. Include a Sigmoid on the connected path to Model Output. Sigmoid has no settings or trainable numbers. Its values are not automatically calibrated probabilities, and a row need not sum to one. This lesson's adapter also requires a Linear block and explicit weights and offsets in Parameters; passing inputs through unchanged is one useful scaffold, not a mathematical requirement of Sigmoid. Equivalent constructions are accepted."
    }
    public var formula: String { "s(x) = 1 / (1 + exp(−x))" }
    public func apply(_ x: Double) throws -> Double {
        guard x.isFinite else { throw LearningContractError("Sigmoid reference input must be finite.") }
        if x >= 0 { return 1 / (1 + exp(-x)) }
        let positive = exp(x)
        return positive / (1 + positive)
    }
}
extension LearningExamples {
    public static var sigmoidReference: SigmoidLessonReference { SigmoidLessonReference() }
}
