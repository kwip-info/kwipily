import XCTest

@testable import KwipMLCore

final class TypedGraphConfigurationTests: XCTestCase {
  func testLegacyDecodePreservesRawConfigWhileEditorMigrationMakesBiasExplicit() throws {
    let bytes = Data(
      #"{"schemaVersion":1,"name":"Old","nodes":[{"id":"x","type":"linear","name":"L","x":0,"y":0,"config":{"in_features":2.0,"out_features":2}}],"edges":[]}"#
        .utf8)
    let raw = try JSONDecoder().decode(GraphDocument.self, from: bytes)
    XCTAssertEqual(raw.schemaVersion, 1)
    XCTAssertNil(raw.nodes[0].config["bias"])
    let registry = try ComponentRegistry.builtIn()
    let migrated = try GraphMigration.toCurrent(raw, registry: registry)
    XCTAssertEqual(migrated.schemaVersion, 2)
    XCTAssertEqual(migrated.nodes[0].config["bias"], .bool(true))
    XCTAssertNil(raw.nodes[0].config["bias"])
    XCTAssertEqual(
      try GraphSemantics.canonicalData(raw, registry: registry),
      try GraphSemantics.canonicalData(migrated, registry: registry))
    var without = migrated
    without.nodes[0].config["bias"] = .bool(false)
    XCTAssertNotEqual(
      try GraphSemantics.canonicalData(raw, registry: registry),
      try GraphSemantics.canonicalData(without, registry: registry))
  }
  func testStrictConfigurationRejectsBooleanWidthsAndNumericBiasAndMissingNewBias() throws {
    let registry = try ComponentRegistry.builtIn()
    var node = GraphNode(
      type: "linear", config: ["in_features": 2, "out_features": 2, "bias": true])
    XCTAssertNoThrow(
      try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 2, registry: registry))
    for malformed: JSONValue in [
      0, 1, .number(0.0), .string("true"), .null, .array([]), .object([:]),
    ] {
      node.config["bias"] = malformed
      XCTAssertThrowsError(
        try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 2, registry: registry))
    }
    node.config["bias"] = true
    node.config["unknown"] = 2
    XCTAssertThrowsError(
      try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 2, registry: registry))
    node.config.removeValue(forKey: "unknown")

    node.config.removeValue(forKey: "bias")
    XCTAssertThrowsError(
      try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 2, registry: registry))
    XCTAssertNoThrow(
      try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 1, registry: registry))
    node.config["in_features"] = true
    XCTAssertThrowsError(
      try GraphMigration.effectiveConfiguration(of: node, schemaVersion: 1, registry: registry))
  }
  func testLegacySubmissionCanonicalBytesRemainUnchanged() throws {
    let original = Data(
      #"{"graph":{"edges":[],"name":"Old","nodes":[{"config":{"in_features":2,"out_features":2},"id":"x","name":"L","type":"linear","x":0,"y":0}],"schemaVersion":1},"kind":"affineGraph","parameters":{},"schemaVersion":1}"#
        .utf8)
    let submission = try JSONDecoder().decode(AffineLearningSubmission.self, from: original)
    XCTAssertEqual(try AffineLearningCodec.encodeSubmission(submission), original)

  }
  func testTypedCompositeOverrideAndPinnedLegacyDefinitionPreserved() throws {
    let registry = try ComponentRegistry.builtIn()
    var project = CompositeExamples.parallelClassifier
    project.definitions[0].parameters.append(
      CompositeParameter(
        ComponentField(
          key: "useBias", title: "Bias", kind: .boolean, defaultValue: true, min: nil, max: nil),
        targets: [CompositeParameterTarget("branch-a", "bias")]))
    project.root.instances[0].overrides["useBias"] = false
    let expanded = try CompositeExpander.expand(project, registry: registry)
    let id = try XCTUnwrap(
      expanded.nodeOrigins.first(where: { $0.value.localNodeID == "branch-a" })?.key)
    XCTAssertEqual(expanded.graph.nodes.first(where: { $0.id == id })?.config["bias"], false)
    project.root.instances[0].overrides["useBias"] = 0
    XCTAssertThrowsError(try CompositeExpander.expand(project, registry: registry))
    project.root.instances[0].overrides["useBias"] = true
    project.definitions[0].parameters[1].targets = [
      CompositeParameterTarget("branch-a", "out_features")
    ]
    XCTAssertThrowsError(try CompositeExpander.expand(project, registry: registry))
    var legacy = CompositeExamples.parallelClassifier
    legacy.schemaVersion = 1
    legacy.root.configurationVersion = nil
    for i in legacy.root.nodes.indices { legacy.root.nodes[i].config.removeValue(forKey: "bias") }
    for i in legacy.definitions.indices {
      legacy.definitions[i].body.configurationVersion = nil
      for n in legacy.definitions[i].body.nodes.indices {
        legacy.definitions[i].body.nodes[n].config.removeValue(forKey: "bias")
      }
    }
    let pinned = legacy.definitions
    let migrated = try CompositeMigration.toCurrent(legacy, registry: registry)
    XCTAssertEqual(migrated.definitions, pinned)
    let oldExpansion = try CompositeExpander.expand(legacy, registry: registry)
    let newExpansion = try CompositeExpander.expand(migrated, registry: registry)
    XCTAssertEqual(oldExpansion, newExpansion)
  }

  func testBiasOwnershipChangesPreallocationLimitAtExactBoundary() throws {
    let registry = try ComponentRegistry.builtIn()
    let widths = [2, 512, 512, 512, 512, 512, 512, 512, 512, 319, 2]
    var nodes = [GraphNode(id: "input", type: "input", config: ["features": 2])]
    for index in 0..<(widths.count - 1) {
      nodes.append(
        GraphNode(
          id: "l\(index)", type: "linear",
          config: [
            "in_features": .integer(Int64(widths[index])),
            "out_features": .integer(Int64(widths[index + 1])), "bias": false,
          ]))
    }
    nodes.append(GraphNode(id: "output", type: "output"))
    let edges = (0..<(nodes.count - 1)).map {
      GraphEdge(source: nodes[$0].id, target: nodes[$0 + 1].id)
    }
    var graph = GraphDocument(nodes: nodes, edges: edges)
    XCTAssertTrue(
      GraphValidator.validate(graph, registry: registry).isValid,
      "1,999,998 weights are within the bound")
    graph.nodes[graph.nodes.count - 2].config["bias"] = true
    XCTAssertTrue(
      GraphValidator.validate(graph, registry: registry).isValid,
      "Two output biases reach exactly 2,000,000 parameters")
    graph.nodes[1].config["bias"] = true
    XCTAssertTrue(
      GraphValidator.validate(graph, registry: registry).issues.contains {
        $0.message.contains("2 million")
      })
  }

  func testSemanticOrderingUsesOriginalOddIdentifiersAndRejectsEmptyUnknownSchema() throws {
    let registry = try ComponentRegistry.builtIn()
    let identifiers = ["z", "quote\"id", "line\nfeed", "é", "\\backslash"]
    let graph = GraphDocument(nodes: identifiers.map { GraphNode(id: $0, type: "relu") })
    let value = try XCTUnwrap(
      JSONSerialization.jsonObject(with: GraphSemantics.canonicalData(graph, registry: registry))
        as? [String: Any])
    let nodes = try XCTUnwrap(value["nodes"] as? [[String: Any]])
    XCTAssertEqual(nodes.compactMap { $0["id"] as? String }, identifiers.sorted())
    XCTAssertThrowsError(
      try GraphMigration.toCurrent(GraphDocument(schemaVersion: 99), registry: registry))
    XCTAssertThrowsError(
      try GraphSemantics.canonicalData(GraphDocument(schemaVersion: 99), registry: registry))
  }

}
