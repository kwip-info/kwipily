import Foundation

public struct LinearParameterValues: Codable, Equatable {
  public var weight: LearningTensor
  public var bias: LearningTensor
  public init(weight: LearningTensor, bias: LearningTensor) {
    self.weight = weight
    self.bias = bias
  }
  public static func zeros(input: Int, output: Int) throws -> Self {
    guard (1...512).contains(input), (1...512).contains(output), input * output <= 4096 else {
      throw LearningContractError(
        "Stored lesson weights allow at most 4096 values, plus a dormant bias vector of at most 512 values."
      )
    }
    return Self(
      weight: LearningTensor(
        shape: [output, input], values: Array(repeating: 0, count: input * output)),
      bias: LearningTensor(shape: [output], values: Array(repeating: 0, count: output)))
  }
  public func validate(input: Int, output: Int) throws {
    guard (1...512).contains(input), (1...512).contains(output), weight.wellFormed, bias.wellFormed,
      weight.shape == [output, input],
      bias.shape == [output], weight.values.count <= 4096, bias.values.count <= 512,
      (weight.values + bias.values).allSatisfy({ abs($0) <= 1_000_000 })
    else {
      throw LearningContractError(
        "Parameter shapes must match the selected Linear dimensions; values must be finite within ±1,000,000. Reset explicitly after a shape change."
      )
    }
  }
}
/// Only parameters owned by the submitted layer; dormant editor values are not execution inputs.
public struct LinearSubmittedParameters: Codable, Equatable {
  public var weight: LearningTensor
  public var bias: LearningTensor?
  public init(weight: LearningTensor, bias: LearningTensor? = nil) {
    self.weight = weight
    self.bias = bias
  }
  public init(_ stored: LinearParameterValues, biasEnabled: Bool = true) {
    self.init(weight: stored.weight, bias: biasEnabled ? stored.bias : nil)
  }
  private enum CodingKeys: String, CodingKey { case weight, bias }
  public init(from decoder: Decoder) throws {
    let raw = try decoder.container(keyedBy: LearningParameterKey.self)
    guard Set(raw.allKeys.map(\.stringValue)).isSubset(of: ["weight", "bias"]),
      raw.contains(LearningParameterKey("weight"))
    else {
      throw LearningContractError(
        "Submitted Linear parameters require weight and optional owned bias only.")
    }
    let values = try decoder.container(keyedBy: CodingKeys.self)
    weight = try values.decode(LearningTensor.self, forKey: .weight)
    // A present null is malformed; absence alone represents disabled bias.
    bias = values.contains(.bias) ? try values.decode(LearningTensor.self, forKey: .bias) : nil
  }
  public func validate(input: Int, output: Int, biasEnabled: Bool) throws {
    guard weight.values.count + (bias?.values.count ?? 0) <= 4096 else {
      throw LearningContractError("Active parameter tensor budget exceeds 4096 values.")
    }
    guard weight.wellFormed, weight.shape == [output, input],
      weight.values.allSatisfy({ abs($0) <= 1_000_000 })
    else {
      throw LearningContractError(
        "Submitted weight must match [out,in] and contain finite values within ±1,000,000.")
    }
    if biasEnabled {
      guard let bias else {
        throw LearningContractError("Bias-enabled Linear requires an explicit bias tensor.")
      }
      try LinearParameterValues(weight: weight, bias: bias).validate(input: input, output: output)
    } else if bias != nil {
      throw LearningContractError("Bias-disabled Linear owns weight only; omit the bias field.")
    }
  }
}
private struct LearningParameterKey: CodingKey {
  let stringValue: String
  var intValue: Int? { nil }
  init(_ value: String) { stringValue = value }
  init?(stringValue: String) { self.stringValue = stringValue }
  init?(intValue: Int) { return nil }
}
public struct AffineLearningSubmission: Codable, Equatable {
  public var schemaVersion = 2
  public var kind = "affineGraph"
  public var graph: GraphDocument
  public var parameters: [String: LinearSubmittedParameters]
  public init(graph: GraphDocument, parameters: [String: LinearParameterValues]) {
    self.graph = graph
    self.parameters = parameters.mapValues { LinearSubmittedParameters($0) }
    for node in graph.nodes where node.type == "linear" && node.config["bias"]?.boolValue == false {
      self.parameters[node.id]?.bias = nil
    }
  }
  public init(
    graph: GraphDocument, submittedParameters: [String: LinearSubmittedParameters],
    schemaVersion: Int = 2
  ) {
    self.graph = graph
    self.parameters = submittedParameters
    self.schemaVersion = schemaVersion
  }
  private enum CodingKeys: String, CodingKey { case schemaVersion, kind, graph, parameters }
  public init(from decoder: Decoder) throws {
    let raw = try decoder.container(keyedBy: LearningParameterKey.self)
    guard
      Set(raw.allKeys.map(\.stringValue)) == Set(["schemaVersion", "kind", "graph", "parameters"])
    else { throw LearningContractError("Invalid affine submission fields.") }
    let values = try decoder.container(keyedBy: CodingKeys.self)
    schemaVersion = try values.decode(Int.self, forKey: .schemaVersion)
    kind = try values.decode(String.self, forKey: .kind)
    guard [1, 2].contains(schemaVersion), kind == "affineGraph" else {
      throw LearningContractError("Unsupported affine submission version or kind.")
    }
    graph = try values.decode(GraphDocument.self, forKey: .graph)
    parameters = try values.decode([String: LinearSubmittedParameters].self, forKey: .parameters)
    if schemaVersion == 1 {
      guard parameters.values.allSatisfy({ $0.bias != nil }),
        !graph.nodes.contains(where: {
          $0.type == "linear" && $0.config["bias"]?.boolValue == false
        })
      else {
        throw LearningContractError(
          "Legacy affine submissions require enabled bias and weight/bias parameters.")
      }
    }
  }
  public func validateParameters(registry: ComponentRegistry? = nil) throws {
    guard [1, 2].contains(schemaVersion), kind == "affineGraph" else {
      throw LearningContractError("Unsupported affine submission version or kind.")
    }
    let validation = GraphValidator.validate(graph, registry: try registry ?? .builtIn())
    guard validation.isValid else {
      throw LearningContractError(validation.issues.map(\.message).joined(separator: " · "))
    }
    let linear = graph.nodes.filter { $0.type == "linear" }
    guard Set(parameters.keys) == Set(linear.map(\.id)) else {
      throw LearningContractError(
        "Provide explicit parameters for every Linear component and no extra nodes.")
    }
    for node in linear {
      let enabled = node.config["bias"]?.boolValue ?? true
      guard schemaVersion == 2 || enabled else {
        throw LearningContractError("Legacy affine submissions require enabled bias.")
      }
      guard let value = parameters[node.id], let input = node.config["in_features"]?.integerValue,
        let output = node.config["out_features"]?.integerValue
      else { throw LearningContractError("Missing or nonintegral Linear configuration.") }
      try value.validate(input: input, output: output, biasEnabled: enabled)
    }
    let active = parameters.values.reduce(0) {
      $0 + $1.weight.values.count + ($1.bias?.values.count ?? 0)
    }
    let projected = active + validation.widths.values.reduce(0, +) * 2
    guard projected <= 4096 else {
      throw LearningContractError(
        "Assessment tensor budget exceeds 4096 elements across active parameters and a two-row fixture's intermediate outputs."
      )
    }
  }
}
public struct AffineLearningDraft: Codable, Equatable {
  public var lessonIdentity: AffineLessonIdentity?
  public var schemaVersion = 1
  public var lessonID = "assessment.affine-composition"
  public var contentVersion = ContractVersion()
  public var projectID: String
  public var returnProjectID: String?
  public var graph: GraphDocument
  public var parameters: [String: LinearParameterValues] = [:]
  public var mode: LearningMode
  public var notes = ""
  public var hintIDs: [String] = []
  public init(
    projectID: String, graph: GraphDocument, mode: LearningMode = .guided,
    returnProjectID: String? = nil, lessonIdentity: AffineLessonIdentity? = nil
  ) {
    self.projectID = projectID
    self.graph = graph
    self.mode = mode
    self.returnProjectID = returnProjectID
    self.lessonIdentity = lessonIdentity
    if let lessonIdentity { self.schemaVersion = 2; self.lessonID = lessonIdentity.assessmentID; self.contentVersion = lessonIdentity.contentVersion }
  }
  public func resolvedLessonIdentity() throws -> AffineLessonIdentity {
    let identity = try resolveAffineIdentity(schemaVersion: schemaVersion, identity: lessonIdentity)
    guard lessonID == identity.assessmentID, contentVersion == identity.contentVersion else {
      throw LearningContractError("Draft lesson identity does not match its declared content.")
    }
    return identity
  }
  public func validate() throws {
    _ = try resolvedLessonIdentity()
    guard
      contentVersion == ContractVersion(), UUID(uuidString: projectID) != nil,
      returnProjectID.map({ UUID(uuidString: $0) != nil }) ?? true, notes.count <= 8000,
      parameters.count <= 256, hintIDs.count <= 16, Set(hintIDs).count == hintIDs.count,
      hintIDs.allSatisfy({ !$0.isEmpty && $0.count <= 128 })
    else { throw LearningContractError("Unsupported or unbounded affine lesson draft.") }
    guard [1, 2].contains(graph.schemaVersion), graph.nodes.count <= 128, graph.edges.count <= 256,
      graph.nodes.allSatisfy({
        $0.x.isFinite && $0.y.isFinite && $0.config.values.allSatisfy(\.isFiniteJSON)
      })
    else { throw LearningContractError("Lesson graph exceeds document bounds.") }
    for (id, value) in parameters {
      guard !id.isEmpty, id.count <= 128, value.weight.shape.count == 2, value.bias.shape.count == 1
      else { throw LearningContractError("Malformed lesson parameter identity or shape.") }
      try value.validate(input: value.weight.shape[1], output: value.weight.shape[0])
    }
  }
  public func submission(registry: ComponentRegistry? = nil) throws -> AffineLearningSubmission {
    try validate()
    try ModelConstructionLessonTasks.validateConstruction(graph,identity:resolvedLessonIdentity())
    let validation = GraphValidator.validate(graph, registry: try registry ?? .builtIn())
    guard validation.isValid else {
      throw LearningContractError(validation.issues.map(\.message).joined(separator: " · "))
    }
    var selected: [String: LinearSubmittedParameters] = [:]
    for node in graph.nodes where node.type == "linear" {
      guard let value = parameters[node.id] else {
        throw LearningContractError(
          "Enter explicit \(node.config["bias"]?.boolValue == false ? "weights" : "weights and bias") for \(node.name.isEmpty ? node.id:node.name)."
        )
      }
      let enabled = node.config["bias"]?.boolValue ?? true
      let submitted = LinearSubmittedParameters(value, biasEnabled: enabled)
      try submitted.validate(
        input: node.config["in_features"]?.integerValue ?? 0,
        output: node.config["out_features"]?.integerValue ?? 0, biasEnabled: enabled)
      selected[node.id] = submitted
    }
    let submission = AffineLearningSubmission(graph: graph, submittedParameters: selected)
    try submission.validateParameters(registry: registry)
    return submission
  }
}
public enum AffineLearningCodec {
  public static func encodeDraft(_ value: AffineLearningDraft) throws -> Data {
    try value.validate()
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(value)
    guard data.count <= 2_000_000 else { throw LearningContractError("Lesson draft exceeds 2 MB.") }
    return data
  }
  public static func decodeDraft(_ data: Data) throws -> AffineLearningDraft {
    guard data.count <= 2_000_000 else { throw LearningContractError("Lesson draft exceeds 2 MB.") }
    let value = try JSONDecoder().decode(AffineLearningDraft.self, from: data)
    try value.validate()
    return value
  }
  public static func encodeSubmission(_ value: AffineLearningSubmission) throws -> Data {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    let data = try encoder.encode(value)
    guard data.count <= 2_000_000 else {
      throw LearningContractError("Lesson submission exceeds 2 MB.")
    }
    return data
  }
}
public struct AffineLearningReceipt: Codable {
  public var lessonIdentity: AffineLessonIdentity?
  public var schemaVersion = 1
  public var attempt: LearningAttempt
  public var projectID: String
  public var submission: AffineLearningSubmission
  public var submissionSHA256: String
  public var runtime: LearningRuntime
  public var fixtureSeed: UInt64
  public var fixtureDigest: String
  public var notes: String
  public var evidence: LearningEvidence?
  public init(
    attempt: LearningAttempt, projectID: String, submission: AffineLearningSubmission,
    submissionSHA256: String, runtime: LearningRuntime, fixtureSeed: UInt64, fixtureDigest: String,
    notes: String, evidence: LearningEvidence? = nil, lessonIdentity: AffineLessonIdentity? = nil
  ) {
    self.evidence = evidence
    self.lessonIdentity = lessonIdentity
    if lessonIdentity != nil { self.schemaVersion = 2 }
    self.attempt = attempt
    self.projectID = projectID
    self.submission = submission
    self.submissionSHA256 = submissionSHA256
    self.runtime = runtime
    self.fixtureSeed = fixtureSeed
    self.fixtureDigest = fixtureDigest
    self.notes = notes
  }
  public func resolvedLessonIdentity() throws -> AffineLessonIdentity {
    try resolveAffineIdentity(schemaVersion: schemaVersion, identity: lessonIdentity)
  }
  public func replayPlan() throws -> LearningAssessmentPlan {
    let identity = try resolvedLessonIdentity()
    let plan = try AffineLessonTasks.plan(identity: identity, runtimeVersion: runtime.version,
      fixtureSeed: fixtureSeed, assessmentVersion: attempt.assessmentVersion)
    guard runtime == plan.runtime, attempt.objectiveID == plan.objectiveID,
      attempt.contentVersion == plan.contentVersion,
      attempt.grade.assessmentID == plan.id, attempt.grade.assessmentVersion == plan.assessmentVersion,
      attempt.grade.contentVersion == plan.contentVersion, attempt.grade.objectiveID == plan.objectiveID,
      fixtureDigest == plan.fixtureDigest, attempt.grade.fixtureDigest == plan.fixtureDigest else {
      throw LearningContractError("Receipt task, versions, runtime or fixture binding do not match.")
    }
    return plan
  }
}

private func resolveAffineIdentity(schemaVersion: Int, identity: AffineLessonIdentity?) throws -> AffineLessonIdentity {
  if schemaVersion == 1 && identity == nil { return .composition }
  guard schemaVersion == 2, let identity else { throw LearningContractError("Unsupported affine document identity or schema.") }
  try identity.validate()
  return identity
}
