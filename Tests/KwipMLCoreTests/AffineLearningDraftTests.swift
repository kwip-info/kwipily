import XCTest

@testable import KwipMLCore

final class AffineLearningDraftTests: XCTestCase {
  func testExplicitZeroParametersAndShapeChangeDoNotSilentlyRewriteValues() throws {
    var draft = AffineLearningDraft(projectID: UUID().uuidString, graph: graph())
    XCTAssertThrowsError(try draft.submission())
    draft.parameters["map"] = try LinearParameterValues.zeros(input: 2, output: 2)
    XCTAssertEqual(try draft.submission().parameters["map"]?.weight.values, [0, 0, 0, 0])
    draft.parameters["map"]?.weight.values[0] = 3
    draft.graph.nodes[1].config["out_features"] = 3
    XCTAssertThrowsError(try draft.submission())
    XCTAssertEqual(draft.parameters["map"]?.weight.values[0], 3)
    draft.graph.nodes[1].config["out_features"] = 2
    XCTAssertNoThrow(try draft.submission())
    draft.parameters["deleted-node"] = try LinearParameterValues.zeros(input: 2, output: 2)
    XCTAssertNil(try draft.submission().parameters["deleted-node"])
  }
  func testRoundTripDraftMalformedVersionAndParameterBounds() throws {
    var draft = AffineLearningDraft(projectID: UUID().uuidString, graph: graph(), mode: .practice)
    draft.parameters["map"] = try LinearParameterValues.zeros(input: 2, output: 2)
    draft.notes = "Predict shape before multiplying."
    XCTAssertEqual(
      try AffineLearningCodec.decodeDraft(AffineLearningCodec.encodeDraft(draft)), draft)
    draft.schemaVersion = 9
    XCTAssertThrowsError(try AffineLearningCodec.encodeDraft(draft))
    XCTAssertThrowsError(try LinearParameterValues.zeros(input: 512, output: 512))
    var values = try LinearParameterValues.zeros(input: 2, output: 2)
    values.bias.values[0] = .infinity
    XCTAssertThrowsError(try values.validate(input: 2, output: 2))
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
