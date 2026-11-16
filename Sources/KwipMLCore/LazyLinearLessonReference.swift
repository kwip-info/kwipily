import Foundation

public enum LazyLinearAssessment {
    public static let operation = "lazy-linear-inferred-width"
    public static let title = "LazyLinear · infer the feature width"
    public static let instructions = "Each observation contains two groups of three measurements. Build Input → LazyLinear → Model Output, choosing four output features. Either Learn bias choice is accepted. The layer should keep the two groups and infer three input features from the last direction of the array. Check construction measures the created weights, repeated results and saved-state reload; it does not train the layer."
    public static let explanation = "LazyLinear chooses its input width from the last direction of the connected array. It creates weights once: four outputs × three inputs = twelve weights, plus four bias values when Learn bias is enabled. A seed controls the random initialization; an eager Linear layer can consume random numbers in a different order, so matching seeds do not promise matching weights. Repeating a calculation and loading already-created weights should preserve the results without creating fresh random weights. This assessment checks shapes and state preservation, not learned prediction quality or numerical map mastery."
    public static func plan(runtimeVersion: String, seed: UInt64) throws -> LearningAssessmentPlan {
        var state = seed
        func next() -> Double { state = state &* 6364136223846793005 &+ 1442695040888963407; return Double(Int((state >> 32) % 41) - 20) / 4 }
        var fixtures: [LearningFixture] = []
        for (index, batch) in [1, 2, 3].enumerated() {
            let expected: [String: LearningTensor] = [
                "output-shape": .init(shape: [3], values: [Double(batch), 2, 4]),
                "inferred-width": .init(shape: [1], values: [3]),
                "weight-shape": .init(shape: [2], values: [4, 3]),
                "parameter-count": .init(shape: [1], values: [1]),
                "repeat-state": .init(shape: [1], values: [1]),
                "reload-state": .init(shape: [1], values: [1]),
                "rng-preserved": .init(shape: [1], values: [1])]
            fixtures.append(.init(id: "inferred-\(index)", inputs: ["x": .init(shape: [batch, 2, 3], values: (0..<batch*6).map { _ in next() })], expected: expected))
        }
        let descriptions = [("output-shape", "Keep two groups and produce four values per group."), ("inferred-width", "Infer three features from the final input direction."), ("weight-shape", "Create four sets of three weights."), ("parameter-count", "The measured parameter inventory matches the inferred widths and selected bias."), ("repeat-state", "Repeating a calculation preserves weights and results."), ("reload-state", "Reloading created weights preserves results."), ("rng-preserved", "Repeat and reload do not draw fresh random values.")]
        let plan = LearningAssessmentPlan(id: "assessment.lazy-linear-inferred-width", objectiveID: "objective.lazy-linear-inferred-width", task: instructions,
            runtime: .init(backend: "pytorch", version: runtimeVersion, capabilities: ["lazy_linear": ContractVersion()]), fixtureSeed: seed, fixtures: fixtures,
            criteria: descriptions.map { .init(id: $0.0, description: $0.1, nextStep: "Keep the final input direction at three features, choose four outputs, and inspect the created parameter shapes. State checks use the same created weights, without training between comparisons.", rule: .tensor(key: $0.0, absoluteTolerance: 0, relativeTolerance: 0)) })
        try plan.validate(); return plan
    }
}
