import Foundation

/// Small reference specifications for current Linear and train-only standardScaler operations, not complete courses.
public enum LearningExamples {
  public static let objectives: [LearningObjective] = [
    LearningObjective(
      id: "vectors.dot", title: "Vector dot products",
      explanation:
        "A dot product combines aligned coordinates and predicts the sign and magnitude of a projection."
    ),
    LearningObjective(
      id: "matrices.affine", title: "Compose affine maps",
      explanation:
        "For column vectors, y = W2(W1x + b1) + b2. Equivalent weights are W2W1 and equivalent bias is W2b1 + b2. Matrix order and bias transformation both matter.",
      prerequisiteIDs: ["vectors.dot"]),
    LearningObjective(
      id: "matrices.affine-from-probes", title: "Recover an affine map from its behavior",
      explanation: "The output at zero determines bias; subtract it from each basis-vector output to recover the corresponding matrix column.",
      prerequisiteIDs: ["vectors.dot"]),
    LearningObjective(
      id: "matrices.affine-backward-chain", title: "Backpropagate through an affine chain",
      explanation: "For y = W2(W1x + b1) + b2, row upstream gradients propagate as g W2 W1. Reverse the computational traversal, use transposed Linear weights, and omit bias shifts; this is a vector-Jacobian product, not an inverse.",
      prerequisiteIDs: ["matrices.affine"]),
    LearningObjective(
      id: "activations.relu", title: "Clip negative values with ReLU",
      explanation: "ReLU changes each negative number to zero while keeping positive numbers unchanged. It acts independently on each value and preserves rows, columns and their order.",
      prerequisiteIDs: ["matrices.affine"]),
    LearningObjective(
      id: "activations.tanh", title: "Transform signed values with Tanh",
      explanation: "Tanh changes each number independently into a smooth signed value between −1 and 1. Zero stays zero, and rows and columns keep their order. These values are not probabilities.",
      prerequisiteIDs: ["matrices.affine"]),
    LearningObjective(
      id: "activations.sigmoid", title: "Transform values with Sigmoid",
      explanation: "Sigmoid changes each value independently into a smooth value between zero and one, preserving rows and columns. These values are not automatically calibrated probabilities.",
      prerequisiteIDs: ["matrices.affine"]),
    LearningObjective(
      id: "activations.gelu-exact", title: "Construct exact GELU",
      explanation: "GELU multiplies each input by its standard-normal cumulative probability. It preserves shape, retains a negative branch and has no trainable parameters.",
      prerequisiteIDs: ["matrices.affine"]),
    LearningObjective(
      id: "activations.gelu-tanh", title: "Construct tanh-approximate GELU",
      explanation: "The cubic Tanh expression approximates GELU; Tanh alone is a different function. Approximation mode is explicit model semantics.",
      prerequisiteIDs: ["activations.gelu-exact"]),
    LearningObjective(
      id: "data.training-normalization", title: "Fit normalization on training rows",
      explanation:
        "Use training mean and population variance. Validation rows only transform; an outlier must not change fitted statistics.",
      prerequisiteIDs: ["vectors.dot"]),
  ]
  public static func affineComposition(runtimeVersion: String, seed: UInt64)
    -> LearningAssessmentPlan
  {
    var generator = FixtureGenerator(seed: seed)
    let fixtures = (0..<4).map { index -> LearningFixture in
      let x = (0..<4).map { _ in generator.value() }
      var output: [Double] = []
      for row in 0..<2 {
        let a = x[row * 2]
        let b = x[row * 2 + 1]
        output += [3 * a + 5 * b - 2.75, -3.5 * a + 5 * b - 3]
      }
      return LearningFixture(
        id: "affine-\(seed)-\(index)", inputs: ["x": LearningTensor(shape: [2, 2], values: x)],
        expected: ["output": LearningTensor(shape: [2, 2], values: output)])
    }
    return LearningAssessmentPlan(
      id: "assessment.affine-composition", objectiveID: "matrices.affine",
      assessmentVersion: ContractVersion(1, 1, 0), task: affineReference.taskText,
      runtime: LearningRuntime(
        backend: "pytorch", version: runtimeVersion,
        capabilities: ["linear": ContractVersion(1, 1, 0)]), fixtureSeed: seed, fixtures: fixtures,
      criteria: [
        LearningCriterion(
          id: "affine-output",
          description:
            "Produce shape [2,2] and equivalent numerical outputs on four independently generated input batches.",
          nextStep:
            "Multiply matrices in application order and transform the first bias by W2 before adding b2.",
          rule: .tensor(key: "output", absoluteTolerance: 1e-5, relativeTolerance: 1e-5))
      ])
  }
  /// Historical receipts select their recorded contract. Unknown revisions must not be guessed.
  public static func affineCompositionForReplay(
    runtimeVersion: String, seed: UInt64, assessmentVersion: ContractVersion,
    contentVersion: ContractVersion
  ) throws -> LearningAssessmentPlan {
    guard contentVersion == ContractVersion(),
      [ContractVersion(), ContractVersion(1, 1, 0)].contains(assessmentVersion)
    else { throw LearningContractError("Unsupported affine content or assessment revision.") }
    var plan = affineComposition(runtimeVersion: runtimeVersion, seed: seed)
    plan.assessmentVersion = assessmentVersion
    plan.runtime.capabilities["linear"] = assessmentVersion
    return plan
  }
  public static func trainingOnlyStandardization(runtimeVersion: String, seed: UInt64)
    -> LearningAssessmentPlan
  {
    var generator = FixtureGenerator(seed: seed)
    let scale = sqrt(8.0 / 3.0)
    let fixtures = (0..<3).map { index -> LearningFixture in
      let base = generator.value()
      return LearningFixture(
        id: "normalization-\(seed)-\(index)",
        inputs: [
          "training": LearningTensor(
            shape: [3, 2], values: [base, 10, base + 2, 12, base + 4, 14]),
          "validation": LearningTensor(shape: [1, 2], values: [base + 12, 22]),
        ],
        expected: [
          "mean": LearningTensor(shape: [2], values: [base + 2, 12]),
          "scale": LearningTensor(shape: [2], values: [scale, scale]),
          "validation": LearningTensor(shape: [1, 2], values: [10 / scale, 10 / scale]),
        ])
    }
    let numerical = ["mean", "scale", "validation"].map { key in
      LearningCriterion(
        id: "normalization-" + key,
        description:
          "Compute training population statistics and apply the same state to held-out rows.",
        nextStep:
          "Compute the mean and divide squared deviations by training row count; exclude the held-out outlier from fitting.",
        rule: .tensor(key: key, absoluteTolerance: 1e-10, relativeTolerance: 1e-10))
    }
    return LearningAssessmentPlan(
      id: "assessment.training-standardization", objectiveID: "data.training-normalization",
      task:
        "Use one CSV source with numeric x/y columns, a seeded split with seed 42 and validation percentage 25, then fit per-feature population standardization on three training rows. Transform the held-out outlier with that same fitted state. Return measured means, scales, transformed validation values and observed fit/partition row identities.",
      runtime: LearningRuntime(
        backend: "numpy", version: runtimeVersion,
        capabilities: [
          "kwip.data.standardScalerFit": ContractVersion(),
          "kwip.data.transform": ContractVersion(),
        ]), fixtureSeed: seed, fixtures: fixtures,
      criteria: numerical + [
        LearningCriterion(
          id: "training-membership",
          description: "Fit members must be training members, and held-out rows must be disjoint.",
          nextStep:
            "Route only the training partition to the fit input; use the resulting state on both partitions.",
          rule: .trainingOnly(training: "training", heldOut: "validation", fit: "fit"))
      ])
  }
}
private struct FixtureGenerator {
  var state: UInt64
  init(seed: UInt64) { state = seed }
  mutating func value() -> Double {
    state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
    return Double(Int((state >> 32) % 2001) - 1000) / 100.0
  }
}


extension LearningExamples {
  /// Revision 1.1 varies offsets, spreads and held-out values independently; 1.0 stays unchanged.
  public static func trainingOnlyStandardizationV11(runtimeVersion: String, seed: UInt64) -> LearningAssessmentPlan {
    var generator = FixtureGenerator(seed: seed)
    var plan = trainingOnlyStandardization(runtimeVersion: runtimeVersion, seed: seed)
    plan.assessmentVersion = ContractVersion(1, 1, 0)
    plan.fixtures = (0..<4).map { index in
      var offsets = [generator.value(), generator.value()]
      // Exact integer constants avoid rounding residue masquerading as tiny variance.
      if index == 2 { offsets[0] = offsets[0].rounded() }
      if index == 3 { offsets[1] = offsets[1].rounded() }
      let steps = [index == 2 ? 0.0 : Double(index + 1), index == 3 ? 0.0 : Double(index + 2)]
      var training: [Double] = []
      for row in 0..<3 { for column in 0..<2 { training.append(offsets[column] + Double(row) * steps[column]) } }
      let held = [offsets[0] + Double(index + 5), offsets[1] - Double(index + 3)]
      var means: [Double] = []; var scales: [Double] = []
      for column in 0..<2 {
        let values = (0..<3).map { training[$0 * 2 + column] }
        let mean = values.reduce(0, +) / 3
        let variance = values.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / 3
        means.append(mean); scales.append(variance == 0 ? 1 : sqrt(variance))
      }
      return LearningFixture(id: "normalization-v11-\(seed)-\(index)", inputs: [
        "training": LearningTensor(shape: [3,2], values: training),
        "validation": LearningTensor(shape: [1,2], values: held)], expected: [
        "mean": LearningTensor(shape: [2], values: means),
        "scale": LearningTensor(shape: [2], values: scales),
        "validation": LearningTensor(shape: [1,2], values: (0..<2).map { (held[$0] - means[$0]) / scales[$0] })])
    }
    plan.task += " Use population variance (divide by training count); use scale 1 for a constant training feature. Assessment fixtures vary each feature independently."
    return plan
  }
}

extension LearningExamples {
  /// Revision 1.2 grades both transformed partitions. Historical plans remain byte-stable.
  public static func trainingOnlyStandardizationV12(runtimeVersion: String, seed: UInt64) -> LearningAssessmentPlan {
    var plan = trainingOnlyStandardizationV11(runtimeVersion: runtimeVersion, seed: seed)
    plan.assessmentVersion = ContractVersion(1, 2, 0)
    plan.fixtures = plan.fixtures.map { fixture in
      var result = fixture
      result.id = fixture.id.replacingOccurrences(of: "normalization-v11-", with: "normalization-v12-")
      let training = fixture.inputs["training"]!
      let means = fixture.expected["mean"]!.values
      let scales = fixture.expected["scale"]!.values
      result.expected["training"] = LearningTensor(shape: training.shape, values: training.values.enumerated().map { index, value in
        (value - means[index % 2]) / scales[index % 2]
      })
      return result
    }
    plan.criteria.append(LearningCriterion(
      id: "normalization-training",
      description: "Apply training-fitted population standardization to every selected training value.",
      nextStep: "Connect the training partition and the fitted state to Apply Fitted Transform; select its table as the training result.",
      rule: .tensor(key: "training", absoluteTolerance: 1e-10, relativeTolerance: 1e-10)))
    plan.task += " Select transformed training and transformed validation tables. Both numerical outputs are assessed independently of their row identities."
    return plan
  }
}
