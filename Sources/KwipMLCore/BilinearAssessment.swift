import Foundation

public enum BilinearLearningTask: String, CaseIterable, Codable {
    case withBias = "bilinear-with-bias"
    case withoutBias = "bilinear-without-bias"
    public var title: String { self == .withBias ? "Bilinear interactions · learned bias" : "Bilinear interactions · without bias" }
    public var task: String {
        "Compute y[o] = ΣᵢΣⱼ a[i]W[o,i,j]b[j]" + (self == .withBias ? " + bias[o]." : ".")
        + " For L = sum(y ⊙ G), report the output, both input gradients and the shared weight gradient."
        + (self == .withBias ? " Also report the bias gradient." : " Do not add a bias or report a bias gradient.")
        + " Match all observation and sample-prefix dimensions exactly; no axes broadcast. Input gradients keep each observation separate. Shared parameter gradients sum contributions over observations and sample positions."
    }
}

public enum BilinearAssessment {
    public static func plan(task: BilinearLearningTask, runtimeVersion: String, seed: UInt64) throws -> LearningAssessmentPlan {
        var state = seed
        func tensor(_ shape: [Int]) -> LearningTensor {
            let values = (0..<shape.reduce(1, *)).map { _ -> Double in
                state = state &* 6364136223846793005 &+ 1442695040888963407
                return Double(Int((state >> 32) % 25) - 12) / 4
            }
            return .init(shape: shape, values: values)
        }
        // Prefix includes the observation axis; small exact binary fractions keep
        // expected arithmetic stable without using the learner's backend as oracle.
        let shapes: [(prefix: [Int], first: Int, second: Int, outputs: Int)] = [
            ([1], 2, 3, 2), ([3], 3, 2, 3), ([2, 2], 2, 4, 2), ([2, 1, 3], 4, 2, 3)
        ]
        let fixtures = try shapes.enumerated().map { index, shape -> LearningFixture in
            let left = tensor(shape.prefix + [shape.first]), right = tensor(shape.prefix + [shape.second])
            let weight = tensor([shape.outputs, shape.first, shape.second])
            let bias = task == .withBias ? tensor([shape.outputs]) : nil
            let reference = try BilinearLessonReference(left: left, right: right, weight: weight, bias: bias)
            let upstream = tensor(reference.output.shape)
            let gradients = try reference.gradients(for: upstream)
            var inputs = ["left": left, "right": right, "weight": weight, "upstream": upstream]
            var expected = ["output": reference.output, "leftGradient": gradients.left,
                            "rightGradient": gradients.right, "weightGradient": gradients.weight]
            if let bias { inputs["bias"] = bias }
            if let gradient = gradients.bias { expected["biasGradient"] = gradient }
            return .init(id: "\(task.rawValue)-\(seed)-\(index)", inputs: inputs, expected: expected)
        }
        func criterion(_ key: String, _ description: String, _ next: String) -> LearningCriterion {
            .init(id: key, description: description, nextStep: next,
                rule: .tensor(key: key, absoluteTolerance: 1e-6, relativeTolerance: 1e-6))
        }
        var criteria = [
            criterion("output", "Contract both feature axes with each output's weight slice.", "Check W[o,i,j], the order of the two inputs and the selected bias mode. Prefixes must match exactly."),
            criterion("leftGradient", "Keep the first input gradient in its original shape.", "For each position, sum G[o]W[o,i,j]b[j] over output o and second feature j."),
            criterion("rightGradient", "Keep the second input gradient in its original shape.", "For each position, sum G[o]a[i]W[o,i,j] over output o and first feature i."),
            criterion("weightGradient", "Accumulate every use of the shared weight tensor.", "Sum G[o]a[i]b[j] over all observation and sample positions; retain weight shape [O,H1,H2].")
        ]
        if task == .withBias {
            criteria.append(criterion("biasGradient", "Accumulate the shared bias gradient.", "Sum G[o] over all observation and sample positions, leaving one gradient per output feature."))
        }
        let plan = LearningAssessmentPlan(id: "assessment." + task.rawValue, objectiveID: "linear-algebra." + task.rawValue,
            task: task.task, runtime: .init(backend: "pytorch", version: runtimeVersion, capabilities: ["bilinear": ContractVersion()]),
            fixtureSeed: seed, fixtures: fixtures, criteria: criteria)
        try plan.validate()
        return plan
    }
}
