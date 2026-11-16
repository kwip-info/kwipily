import Foundation

/// Public saved statistics; the learner supplies a graph, not authored result evidence.
public struct FrozenStandardizationReference: Equatable {
    public let means = [2.0, 12.0]
    public let scales = [2.0, 4.0]
    public var instructions: String {
        "A training pipeline saved feature means μ = [2, 12] and scales s = [2, 4]. Build a model graph that turns new raw numeric observations X [batch, 2] into prepared features Z with zⱼ = (xⱼ − μⱼ) / sⱼ. Feature order is fixed. Do not refit statistics on incoming rows: a row must receive the same result regardless of its batch companions. Derive a diagonal weight matrix and bias for PyTorch Linear, or use equivalent affine components. This models the saved numerical transform only; it does not load a dataflow state artifact or infer missing values."
    }
    public func apply(_ input: [Double]) throws -> [Double] {
        guard input.count == 2, input.allSatisfy(\.isFinite) else {
            throw LearningContractError("Provide two finite feature values in the declared order.")
        }
        let output = (0..<2).map { (input[$0] - means[$0]) / scales[$0] }
        guard output.allSatisfy(\.isFinite) else { throw LearningContractError("Frozen transform exceeded finite bounds.") }
        return output
    }
}
extension LearningExamples {
    public static let frozenStandardizationReference = FrozenStandardizationReference()
}
