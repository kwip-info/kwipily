import Foundation

/// Independent scalar mathematics for the signed forward-transformation task.
public struct TanhLessonReference: Equatable {
    public init() {}
    public var instructions: String {
        "Build a flow that transforms each incoming number separately with Tanh, preserving two-feature rows and their order. Tanh turns a number into a signed value between −1 and 1; zero becomes zero. This value is not a probability. Include Tanh on the connected path to Model Output. Tanh has no settings or trainable numbers. This lesson's adapter requires a Linear block and explicit weights and fixed offsets in Parameters; passing inputs through unchanged is one useful scaffold, not a mathematical requirement of Tanh. Equivalent constructions are accepted."
    }
    public var formula: String { "tanh(x) = (exp(x) − exp(−x)) / (exp(x) + exp(−x))" }
    public func apply(_ x: Double) throws -> Double {
        guard x.isFinite else { throw LearningContractError("Tanh reference input must be finite.") }
        // Foundation supplies the stable scalar implementation; the illustrative
        // exponential formula should not be evaluated naively at extreme inputs.
        return Foundation.tanh(x)
    }
}
extension LearningExamples {
    public static var tanhReference: TanhLessonReference { TanhLessonReference() }
}
