import Foundation

/// Public objective configuration, separate from private seeded assessment fixtures.
public enum LayerNormLearningTask: String, CaseIterable, Codable {
    case rows = "layer-norm-rows"
    case sample = "layer-norm-sample"
    case affine = "layer-norm-affine"

    public var normalizedShape: [Int] { self == .sample ? [2,3] : [3] }
    public var epsilon: Double { 0.125 }
    public var scale: LearningTensor? { self == .affine ? .init(shape:[3],values:[0.5,-1,2]) : nil }
    public var bias: LearningTensor? { self == .affine ? .init(shape:[3],values:[1,-0.5,0.25]) : nil }
    public var title: String {
        switch self {
        case .rows: return "LayerNorm · normalize rows"
        case .sample: return "LayerNorm · normalize each sample"
        case .affine: return "LayerNorm · learned scale and bias"
        }
    }
    public var task: String {
        let axes = self == .sample
            ? "For every observation [2,3], normalize all six coordinates together."
            : "For every observation [2,3], normalize each row of three coordinates separately."
        let affine = self == .affine
            ? " Apply coordinatewise scale γ = [0.5, −1, 2] and bias β = [1, −0.5, 0.25], shared across rows and observations. Compute gradients for both parameter arrays."
            : " Do not apply learned scale or bias."
        return axes + " Use the population variance (divide by the number of normalized coordinates) and epsilon 0.125 inside the square root. Preserve the shape and keep observations separate." + affine + " Given upstream coefficients g, compute the input gradient of L = sum(y ⊙ g), including the dependence of mean and variance on x. Predict which coordinates influence one another before running."
    }
}

/// Private expected values; never expose fixture tensors in the public task specification.
public enum LayerNormAssessment {
    public static func plan(task: LayerNormLearningTask, runtimeVersion: String, seed: UInt64) throws -> LearningAssessmentPlan {
        var state = seed
        func next() -> Double {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return Double((state >> 32) % 129)/16 - 4
        }
        let fixtures = try (0..<4).map { index -> LearningFixture in
            let batch = index % 2 + 1, count = batch * 6
            let values: [Double]
            switch index {
            case 0: values = [1,2,4,10,10,10] // Asymmetric and constant slices expose axes/epsilon errors.
            case 1: values = [2,2.01,1.99,-3,0,6,102,102.01,101.99,7,10,16]
            default: values = (0..<count).map { _ in next() }
            }
            let input = LearningTensor(shape:[batch,2,3],values:values)
            let upstream = LearningTensor(shape:input.shape,values:(0..<count).map { Double(($0*5)%11)-4 })
            let reference = try LayerNormLessonReference(input:input,normalizedShape:task.normalizedShape,epsilon:task.epsilon,scale:task.scale,bias:task.bias)
            let gradients = try reference.gradients(for:upstream)
            var expected = ["output":reference.output,"inputGradient":gradients.input]
            if let scale=gradients.scale { expected["scaleGradient"]=scale }
            if let bias=gradients.bias { expected["biasGradient"]=bias }
            return LearningFixture(id:"\(task.rawValue)-\(seed)-\(index)",inputs:["x":input,"upstream":upstream],expected:expected)
        }
        func criterion(_ id: String, _ key: String, _ description: String, _ nextStep: String) -> LearningCriterion {
            .init(id:id,description:description,nextStep:nextStep,rule:.tensor(key:key,absoluteTolerance:1e-6,relativeTolerance:1e-6))
        }
        var criteria = [
            criterion("normalized-values","output","Normalize the intended slices with population variance and epsilon inside the square root.","Mark the trailing dimensions, compute their mean and variance with divisor m, and use (x−μ)/√(v+ε). Check the affine options and preserve observation boundaries."),
            criterion("coupled-input-gradient","inputGradient","Differentiate through the mean, variance and any elementwise scale.","Let z=(x−μ)/√(v+ε), h=g⊙γ (or g without scale). Then dL/dx=(h−mean(h)−z mean(h⊙z))/√(v+ε), with means over each normalized slice.")
        ]
        if task == .affine {
            criteria += [
                criterion("shared-scale-gradient","scaleGradient","Accumulate each scale parameter's gradient across all shared rows and observations.","For each trailing coordinate, sum gᵢzᵢ across preceding dimensions. Do not average shared parameter contributions."),
                criterion("shared-bias-gradient","biasGradient","Accumulate each bias parameter's gradient across all shared rows and observations.","For each trailing coordinate, sum upstream gᵢ across preceding dimensions. Parameter sharing accumulates contributions from every use.")
            ]
        }
        let plan = LearningAssessmentPlan(id:"assessment."+task.rawValue,objectiveID:"normalization."+task.rawValue,task:task.task,runtime:.init(backend:"pytorch",version:runtimeVersion,capabilities:["layer_norm":ContractVersion()]),fixtureSeed:seed,fixtures:fixtures,criteria:criteria)
        try plan.validate()
        return plan
    }
}
