import XCTest
import KwipMLCore
@testable import KwipMLApp

final class CompositeSourceTests: XCTestCase {
    @MainActor func testAdoptionRetainsSourceAndOnlySemanticModelEditsDetachAttribution() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false); workspace.activeProjectID = UUID().uuidString
        workspace.graph.name = "Owner project name"
        let editor = CompositeWorkspaceStore(workspace: workspace); editor.save()
        let expansion = try CompositeExpander.expand(editor.project, registry: workspace.registry)
        try workspace.adoptCompositeExpansion(project: editor.project, expansion: expansion, sourceURL: editor.saveURL)
        XCTAssertEqual(workspace.graph.name, "Owner project name")
        XCTAssertTrue(workspace.compositeSourceIsCurrent)
        let reference = try XCTUnwrap(workspace.execution.compositeSource)
        let snapshot = try JSONDecoder().decode(CompositeSourceSnapshot.self, from: WorkspaceDocumentReader.read(URL(fileURLWithPath: reference.path)))
        XCTAssertEqual(snapshot.sourceCompositeDocument, editor.project)
        XCTAssertEqual(snapshot.nodeOrigins.count, workspace.graph.nodes.count)
        workspace.graph.nodes[0].name = "Renamed"; workspace.graph.nodes[0].x += 30
        XCTAssertTrue(workspace.compositeSourceIsCurrent)
        let request = root.appendingPathComponent("request"); try FileManager.default.createDirectory(at: request, withIntermediateDirectories: true)
        XCTAssertEqual(try workspace.compositeTrainingArguments(requestDirectory: request).first, "--composite-source")
        workspace.graph.nodes[0].config["features"] = 3
        XCTAssertFalse(workspace.compositeSourceIsCurrent)
        XCTAssertTrue(try workspace.compositeTrainingArguments(requestDirectory: request).isEmpty)
    }
    @MainActor func testMismatchedExpansionCannotReplaceModelAndModifiedSnapshotCannotTrain() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false); workspace.activeProjectID = UUID().uuidString
        let editor = CompositeWorkspaceStore(workspace: workspace); editor.save()
        var expansion = try CompositeExpander.expand(editor.project, registry: workspace.registry)
        expansion.graph.name = "Different source"
        let before = workspace.graph
        XCTAssertThrowsError(try workspace.adoptCompositeExpansion(project: editor.project, expansion: expansion, sourceURL: editor.saveURL))
        XCTAssertEqual(workspace.graph, before)
        editor.useExpandedArchitecture()
        let reference = try XCTUnwrap(workspace.execution.compositeSource)
        try Data("{}".utf8).write(to: URL(fileURLWithPath: reference.path), options: .atomic)
        XCTAssertThrowsError(try workspace.compositeTrainingArguments(requestDirectory: root))
    }
    @MainActor func testLegacySourceHashSurvivesEditableMigrationWithoutRewritingSnapshot() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let workspace = WorkspaceStore(rootOverride: root, detectCapabilities: false)
        let expansion = try CompositeExpander.expand(CompositeExamples.parallelClassifier, registry: workspace.registry)
        var snapshot = CompositeSourceSnapshot(project: CompositeExamples.parallelClassifier, expansion: expansion)
        snapshot.expandedGraph.schemaVersion = 1
        for index in snapshot.expandedGraph.nodes.indices { snapshot.expandedGraph.nodes[index].config.removeValue(forKey: "bias") }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(snapshot)
        let source = root.appendingPathComponent("legacy-source.json"); try bytes.write(to: source)
        workspace.graph = try GraphMigration.toCurrent(snapshot.expandedGraph, registry: workspace.registry)
        workspace.graph.execution = ExecutionSettings()
        workspace.graph.execution?.compositeSource = CompositeSourceReference(path: source.path, sha256: CompositeArchitectureIdentity.hash(bytes), expandedSemanticSHA256: try CompositeArchitectureIdentity.legacyDigest(snapshot.expandedGraph))
        XCTAssertTrue(workspace.compositeSourceIsCurrent)
        let staging = root.appendingPathComponent("request"); try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        XCTAssertEqual(try workspace.compositeTrainingArguments(requestDirectory: staging).first, "--composite-source")
        XCTAssertEqual(try Data(contentsOf: staging.appendingPathComponent("composite-source.json")), bytes)
        XCTAssertEqual(try Data(contentsOf: source), bytes)
        let linear = try XCTUnwrap(workspace.graph.nodes.firstIndex { $0.type == "linear" })
        workspace.graph.nodes[linear].config["bias"] = .bool(false)
        XCTAssertFalse(workspace.compositeSourceIsCurrent)
        XCTAssertTrue(try workspace.compositeTrainingArguments(requestDirectory: staging).isEmpty)
    }
    /// Frozen schema-1 data from 438372c's runner graph fixture, with a matching source title.
    /// No current ExampleFlows, initializer defaults or expander generate these bytes.
    private static let historicalSnapshotBytes = Data(#"{"consumedDefinitions":[],"expandedGraph":{"edges":[{"id":"e0","source":"input","sourcePort":"out","target":"left","targetPort":"in"},{"id":"e1","source":"input","sourcePort":"out","target":"right","targetPort":"in"},{"id":"e2","source":"left","sourcePort":"out","target":"relu","targetPort":"in"},{"id":"e3","source":"right","sourcePort":"out","target":"tanh","targetPort":"in"},{"id":"e4","source":"relu","sourcePort":"out","target":"merge","targetPort":"a"},{"id":"e5","source":"tanh","sourcePort":"out","target":"merge","targetPort":"b"},{"id":"e6","source":"merge","sourcePort":"out","target":"head","targetPort":"in"},{"id":"e7","source":"head","sourcePort":"out","target":"output","targetPort":"in"}],"name":"Test classifier","nodes":[{"config":{"features":2},"id":"input","name":"input","type":"input","x":0,"y":0},{"config":{"in_features":2,"out_features":16},"id":"left","name":"left","type":"linear","x":0,"y":0},{"config":{"in_features":2,"out_features":16},"id":"right","name":"right","type":"linear","x":0,"y":0},{"config":{},"id":"relu","name":"relu","type":"relu","x":0,"y":0},{"config":{},"id":"tanh","name":"tanh","type":"tanh","x":0,"y":0},{"config":{},"id":"merge","name":"merge","type":"concat","x":0,"y":0},{"config":{"in_features":32,"out_features":2},"id":"head","name":"head","type":"linear","x":0,"y":0},{"config":{},"id":"output","name":"output","type":"output","x":0,"y":0}],"schemaVersion":1},"nativeExpansionContractVersion":1,"nodeOrigins":{"head":{"instancePath":[],"localNodeID":"head"},"input":{"instancePath":[],"localNodeID":"input"},"left":{"instancePath":[],"localNodeID":"left"},"merge":{"instancePath":[],"localNodeID":"merge"},"output":{"instancePath":[],"localNodeID":"output"},"relu":{"instancePath":[],"localNodeID":"relu"},"right":{"instancePath":[],"localNodeID":"right"},"tanh":{"instancePath":[],"localNodeID":"tanh"}},"sourceCompositeDocument":{"definitions":[],"name":"Test classifier","root":{"edges":[{"id":"e0","source":"input","sourcePort":"out","target":"left","targetPort":"in"},{"id":"e1","source":"input","sourcePort":"out","target":"right","targetPort":"in"},{"id":"e2","source":"left","sourcePort":"out","target":"relu","targetPort":"in"},{"id":"e3","source":"right","sourcePort":"out","target":"tanh","targetPort":"in"},{"id":"e4","source":"relu","sourcePort":"out","target":"merge","targetPort":"a"},{"id":"e5","source":"tanh","sourcePort":"out","target":"merge","targetPort":"b"},{"id":"e6","source":"merge","sourcePort":"out","target":"head","targetPort":"in"},{"id":"e7","source":"head","sourcePort":"out","target":"output","targetPort":"in"}],"instances":[],"nodes":[{"config":{"features":2},"id":"input","name":"input","type":"input","x":0,"y":0},{"config":{"in_features":2,"out_features":16},"id":"left","name":"left","type":"linear","x":0,"y":0},{"config":{"in_features":2,"out_features":16},"id":"right","name":"right","type":"linear","x":0,"y":0},{"config":{},"id":"relu","name":"relu","type":"relu","x":0,"y":0},{"config":{},"id":"tanh","name":"tanh","type":"tanh","x":0,"y":0},{"config":{},"id":"merge","name":"merge","type":"concat","x":0,"y":0},{"config":{"in_features":32,"out_features":2},"id":"head","name":"head","type":"linear","x":0,"y":0},{"config":{},"id":"output","name":"output","type":"output","x":0,"y":0}]},"schemaVersion":1}}"#.utf8)

    func testFrozenHistoricalSchemaOneSnapshotVerifiesWithoutMutatingItsBytes() throws {
        let bytes = Self.historicalSnapshotBytes
        XCTAssertEqual(CompositeArchitectureIdentity.hash(bytes), "07481f7531f3fac367a0901b3b535eb35b5291e8a3e077e54e06cb56a8f246a5")
        let snapshot = try JSONDecoder().decode(CompositeSourceSnapshot.self, from: bytes)
        XCTAssertEqual(snapshot.expandedGraph.schemaVersion, 1)
        XCTAssertNil(snapshot.sourceCompositeDocument.root.configurationVersion)
        XCTAssertTrue(snapshot.expandedGraph.nodes.filter { $0.type == "linear" }.allSatisfy { $0.config["bias"] == nil })
        let verified = try snapshot.verifiedExpansion(registry: .builtIn())
        XCTAssertEqual(verified.graph.schemaVersion, 2)
        XCTAssertTrue(verified.graph.nodes.filter { $0.type == "linear" }.allSatisfy { $0.config["bias"] == .bool(true) })
        XCTAssertEqual(verified.nodeOrigins, snapshot.nodeOrigins)
        XCTAssertEqual(verified.consumedDefinitions, snapshot.consumedDefinitions)
        XCTAssertEqual(Self.historicalSnapshotBytes, bytes)
    }

    func testLegacySourceVerificationStillRejectsAlteredTopologyOwnershipAndContract() throws {
        let original = try JSONDecoder().decode(CompositeSourceSnapshot.self, from: Self.historicalSnapshotBytes)
        var edge = original; edge.expandedGraph.edges[0].target = "right"
        var origin = original; origin.nodeOrigins["left"]?.localNodeID = "right"
        var version = original; version.nativeExpansionContractVersion = 2
        var pins = original; pins.consumedDefinitions = [CompositeDefinitionReference("invented-pin")]
        for changed in [edge, origin, version, pins] {
            XCTAssertThrowsError(try changed.verifiedExpansion(registry: .builtIn()))
        }
        // Current schema stays byte-structurally exact: semantic equivalence cannot excuse
        // removed settings, changed display ownership or a forged expansion.
        let expansion = try CompositeExpander.expand(CompositeExamples.parallelClassifier, registry: .builtIn())
        let current = CompositeSourceSnapshot(project: CompositeExamples.parallelClassifier, expansion: expansion)
        let linear = try XCTUnwrap(current.expandedGraph.nodes.firstIndex { $0.type == "linear" })
        var missing = current; missing.expandedGraph.nodes[linear].config.removeValue(forKey: "bias")
        var changed = current; changed.expandedGraph.nodes[linear].config["bias"] = .bool(false)
        var moved = current; moved.expandedGraph.nodes[linear].x += 1
        for changed in [missing, changed, moved] {
            XCTAssertThrowsError(try changed.verifiedExpansion(registry: .builtIn()))
        }
    }

    @MainActor func testRealTrainingArchivesExactReusableSource() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.loadExample()
        workspace.setExecution { $0.device = "cpu"; $0.epochs = 2 }
        let editor = CompositeWorkspaceStore(workspace: workspace)
        editor.useExpandedArchitecture()
        let source = try XCTUnwrap(workspace.execution.compositeSource, editor.statusMessage)
        workspace.run()
        let deadline = Date().addingTimeInterval(30)
        while workspace.isRunning && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertTrue(workspace.lastCompleted, workspace.runError ?? workspace.statusMessage)
        let output = try XCTUnwrap(workspace.outputDirectory)
        XCTAssertEqual(try Data(contentsOf: output.appendingPathComponent("composite/source.json")), try Data(contentsOf: URL(fileURLWithPath: source.path)))
        let manifest = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("manifest.json"))) as? [String: Any])
        let provenance = try XCTUnwrap(manifest["provenance"] as? [String: Any])
        XCTAssertEqual((provenance["composite"] as? [String: Any])?["sha256"] as? String, source.sha256)
    }

}
