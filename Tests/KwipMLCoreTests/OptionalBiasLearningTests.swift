import XCTest

@testable import KwipMLCore

final class OptionalBiasLearningTests: XCTestCase {
  func testDisabledBiasIsOmittedWhileDraftRetainsValuesForReenable() throws {
    var draft = AffineLearningDraft(projectID: UUID().uuidString, graph: graph())
    var values = try LinearParameterValues.zeros(input: 2, output: 2)
    values.weight.values = [1, 2, 3, 4]
    values.bias.values = [7, 8]
    draft.parameters["map"] = values
    draft.graph.nodes[1].config["bias"] = false
    let submission = try draft.submission()
    XCTAssertEqual(submission.schemaVersion, 2)
    XCTAssertNil(submission.parameters["map"]?.bias)
    let data = try AffineLearningCodec.encodeSubmission(submission)
    let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    let parameters = try XCTUnwrap(object["parameters"] as? [String: [String: Any]])
    XCTAssertEqual(Set(try XCTUnwrap(parameters["map"]).keys), ["weight"])
    XCTAssertEqual(draft.parameters["map"], values)
    draft = try AffineLearningCodec.decodeDraft(AffineLearningCodec.encodeDraft(draft))
    draft.graph.nodes[1].config["bias"] = true
    XCTAssertEqual(try draft.submission().parameters["map"]?.bias?.values, [7, 8])
    XCTAssertEqual(draft.parameters["map"]?.weight.values, [1, 2, 3, 4])
  }
  func testStrictSubmittedOwnershipAndNullRejection() throws {
    let weight = LearningTensor(shape: [2, 2], values: [1, 0, 0, 1])
    let bias = LearningTensor(shape: [2], values: [0, 0])
    XCTAssertThrowsError(
      try JSONDecoder().decode(
        LinearSubmittedParameters.self,
        from: Data(#"{"weight":{"shape":[2,2],"values":[1,0,0,1]},"bias":null}"#.utf8)))
    XCTAssertThrowsError(
      try LinearSubmittedParameters(weight: weight, bias: nil).validate(
        input: 2, output: 2, biasEnabled: true))
    XCTAssertThrowsError(
      try LinearSubmittedParameters(weight: weight, bias: bias).validate(
        input: 2, output: 2, biasEnabled: false))
    XCTAssertNoThrow(
      try LinearSubmittedParameters(weight: weight, bias: nil).validate(
        input: 2, output: 2, biasEnabled: false))
  }
  func testVersionedReplayKeepsFixturesAndLegacyCapabilityPin() throws {
    let latest = LearningExamples.affineComposition(runtimeVersion: "2.14.1", seed: 734)
    let old = try LearningExamples.affineCompositionForReplay(
      runtimeVersion: "2.14.1", seed: 734, assessmentVersion: ContractVersion(),
      contentVersion: ContractVersion())
    XCTAssertEqual(latest.assessmentVersion, ContractVersion(1, 1, 0))
    XCTAssertEqual(latest.runtime.capabilities["linear"], ContractVersion(1, 1, 0))
    XCTAssertEqual(old.assessmentVersion, ContractVersion())
    XCTAssertEqual(old.runtime.capabilities["linear"], ContractVersion())
    XCTAssertEqual(latest.fixtureDigest, old.fixtureDigest)
    XCTAssertEqual(latest.fixtures, old.fixtures)
    XCTAssertThrowsError(
      try LearningExamples.affineCompositionForReplay(
        runtimeVersion: "2.14.1", seed: 734, assessmentVersion: ContractVersion(9),
        contentVersion: ContractVersion()))
  }

  func testDormantStorageAndActiveExecutionBudgetsAreSeparate() throws {
    let stored = try LinearParameterValues.zeros(input: 64, output: 64)
    XCTAssertEqual(stored.weight.values.count, 4096)
    XCTAssertEqual(stored.bias.values.count, 64)
    XCTAssertNoThrow(
      try LinearSubmittedParameters(stored, biasEnabled: false).validate(
        input: 64, output: 64, biasEnabled: false))
    XCTAssertThrowsError(
      try LinearSubmittedParameters(stored).validate(input: 64, output: 64, biasEnabled: true))
    for hidden in [57, 64] {
      let widths = [2, 64, hidden, 2]
      var nodes = [GraphNode(id: "input", type: "input", config: ["features": 2])]
      var values: [String: LinearParameterValues] = [:]
      for i in 0..<3 {
        let id = "l\(i)"
        nodes.append(
          GraphNode(
            id: id, type: "linear",
            config: [
              "in_features": .integer(Int64(widths[i])),
              "out_features": .integer(Int64(widths[i + 1])), "bias": false,
            ]))
        values[id] = try .zeros(input: widths[i], output: widths[i + 1])
      }
      nodes.append(GraphNode(id: "output", type: "output"))
      let edges = (0..<(nodes.count - 1)).map {
        GraphEdge(source: nodes[$0].id, target: nodes[$0 + 1].id)
      }
      var draft = AffineLearningDraft(
        projectID: UUID().uuidString, graph: GraphDocument(nodes: nodes, edges: edges))
      draft.parameters = values
      XCTAssertThrowsError(try draft.submission()) { error in
        XCTAssertTrue(error.localizedDescription.contains("budget"))
      }
    }
  }
  private func graph() -> GraphDocument {
    GraphDocument(
      nodes: [
        GraphNode(id: "input", type: "input", config: ["features": 2]),
        GraphNode(id: "map", type: "linear", config: ["in_features": 2, "out_features": 2]),
        GraphNode(id: "output", type: "output"),
      ],
      edges: [
        GraphEdge(source: "input", target: "map"), GraphEdge(source: "map", target: "output"),
      ])
  }
}
