import Foundation

/// Independent scalar reference for the reviewed clipping transformation.
public struct ReLULessonReference: Equatable {
    public init() {}
    public var instructions: String {
        "Build a flow that transforms each incoming number separately with ReLU, preserving two-feature rows and their order. ReLU replaces each negative value with zero and keeps zero and positive values unchanged. Include ReLU on the connected path to Model Output. ReLU has no settings or trainable numbers. This lesson's adapter requires a Linear block and explicit weights and fixed offsets in Parameters; passing inputs through unchanged is one useful scaffold, not a mathematical requirement of ReLU. Equivalent constructions are accepted, including repeated ReLU blocks that give the same output."
    }
    public var formula: String { "ReLU(x) = max(x, 0)" }
    public func apply(_ x: Double) throws -> Double {
        guard x.isFinite else { throw LearningContractError("ReLU reference input must be finite.") }
        return max(x, 0)
    }
}
extension LearningExamples {
    public static var reluReference: ReLULessonReference { ReLULessonReference() }
}
