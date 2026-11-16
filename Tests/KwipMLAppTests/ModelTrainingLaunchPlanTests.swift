import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelTrainingLaunchPlanTests: XCTestCase {
    private func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }; return url
    }
    private func capturedBytes(_ inputs: [WorkflowPreviewCapture.Input], _ path: String) throws -> Data {
        guard case .bytes(_, let bytes) = try XCTUnwrap(inputs.first { $0.path == path }) else { throw NSError(domain: "Expected immutable bytes", code: 1) }
        return bytes
    }
    func testCSVIsCapturedWhileGraphRetainsOriginalProvenanceAndArgumentsUseOwnedPaths() throws {
        let root = try root(), csv = root.appendingPathComponent("my observations.csv")
        let original = Data("x,y,label\n1,2,0\n".utf8); try original.write(to: csv)
        let settings = ExecutionSettings(device: "cpu", epochs: 3, datasetCSV: csv.path, featureColumns: ["x", "y"], targetColumn: "label")
        var graph = ExampleFlows.parallelClassifier; graph.execution = settings
        let plan = try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, sourcePermissions: root.appendingPathComponent("grants"), supportedGradientDevices: [])
        try Data("changed".utf8).write(to: csv)
        XCTAssertEqual(try capturedBytes(plan.inputs, "input/data/my observations.csv"), original)
        XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self, from: capturedBytes(plan.inputs, "input/graph.json")), graph)
        let directory = root.appendingPathComponent("operation"), args = plan.arguments(in: directory)
        XCTAssertTrue(args.contains(directory.appendingPathComponent("input/data/my observations.csv").path))
        XCTAssertFalse(args.contains(csv.path)); XCTAssertFalse(args.contains("--registry"))
        XCTAssertEqual(plan.graph.execution?.datasetCSV, csv.path)
        XCTAssertThrowsError(try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, sourcePermissions: root,
            supportedGradientDevices: [], checkCancellation: { throw CancellationError() }))
    }
    func testPreparedSelectionCapturesEntireInventoryAndOmitsSamplingWithoutChangingGraphReference() throws {
        let root = try root(), manifest = root.appendingPathComponent("manifest.json"), digest = String(repeating: "a", count: 64)
        let table = Data("{}".utf8); try table.write(to: root.appendingPathComponent("table.json"))
        let raw = try JSONSerialization.data(withJSONObject: ["schemaVersion": 1, "kind": "dataflowExecution", "status": "completed", "graphDigest": digest,
            "artifacts": [["path": "table.json", "bytes": table.count, "sha256": WorkflowArchiveCodec.hash(table)]]])
        try raw.write(to: manifest)
        var settings = ExecutionSettings(device: "cpu")
        settings.preparedTraining = .init(manifestPath: manifest.path, manifestSHA256: WorkflowArchiveCodec.hash(raw), graphDigest: digest,
            trainingNode: "split", trainingPort: "training", validationNode: "split", validationPort: "validation", featureColumns: ["x", "y"], targetColumn: "label")
        var graph = ExampleFlows.parallelClassifier; graph.execution = settings
        let plan = try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, sourcePermissions: root, supportedGradientDevices: [])
        XCTAssertEqual(Set(plan.inputs.map(\.path)), ["input/graph.json", "input/preparation/manifest.json", "input/preparation/table.json"])
        guard case .file(_, let source, let pin) = try XCTUnwrap(plan.inputs.first { $0.path == "input/preparation/table.json" }) else { return XCTFail("Expected pinned artifact") }
        XCTAssertEqual(source, root.appendingPathComponent("table.json")); XCTAssertEqual(pin, WorkflowArchiveCodec.hash(table))
        XCTAssertFalse(plan.arguments(in: root).contains("--sample-limit")); XCTAssertEqual(plan.graph.execution?.preparedTraining, settings.preparedTraining)
        try Data("changed".utf8).write(to: manifest)
        XCTAssertThrowsError(try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, sourcePermissions: root, supportedGradientDevices: []))
    }
    func testCheckpointTriplePinsAndResumeCompatibilityArePreserved() throws {
        let root = try root(), weights = Data("checkpoint".utf8), metrics = Data("[]".utf8)
        var settings = ExecutionSettings(device: "cpu", epochs: 2)
        var graph = ExampleFlows.parallelClassifier; graph.execution = settings
        let runID = UUID().uuidString
        let manifest = try JSONSerialization.data(withJSONObject: ["status": "completed", "runID": runID,
            "graph": JSONSerialization.jsonObject(with: JSONEncoder().encode(graph)),
            "checkpoint": ["schemaVersion": 1, "completedEpoch": 2, "bytes": weights.count,
                "sha256": WorkflowArchiveCodec.hash(weights), "metricsSHA256": WorkflowArchiveCodec.hash(metrics),
                "contract": ["runtime": ["device": "cpu"]]]])
        try manifest.write(to: root.appendingPathComponent("manifest.json")); try weights.write(to: root.appendingPathComponent("checkpoint.pt")); try metrics.write(to: root.appendingPathComponent("metrics.json"))
        let parent = CheckpointParent(directory: root, runID: runID, completedEpoch: 2,
            manifestSHA256: WorkflowArchiveCodec.hash(manifest), checkpointSHA256: WorkflowArchiveCodec.hash(weights), metricsSHA256: WorkflowArchiveCodec.hash(metrics),
            graph: graph, device: "cpu", lineageMode: "newTraining", parentRunID: nil)
        settings.epochs = 3; graph.execution = settings
        let continuation = CheckpointContinuation(parent: parent, mode: .exactResume)
        let plan = try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, continuation: continuation, sourcePermissions: root, supportedGradientDevices: [])
        XCTAssertEqual(try capturedBytes(plan.inputs, "input/continuation-parent/manifest.json"), manifest)
        XCTAssertEqual(try capturedBytes(plan.inputs, "input/continuation-parent/metrics.json"), metrics)
        guard case .file(_, _, let pin) = try XCTUnwrap(plan.inputs.first { $0.path == "input/continuation-parent/checkpoint.pt" }) else { return XCTFail("Expected streamed pinned checkpoint") }
        XCTAssertEqual(pin, parent.checkpointSHA256); XCTAssertTrue(plan.arguments(in: root).contains("--resume-checkpoint"))
        settings.seed += 1
        XCTAssertThrowsError(try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, continuation: continuation, sourcePermissions: root, supportedGradientDevices: []))
        let fineTune = try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, continuation: .init(parent: parent, mode: .fineTune), sourcePermissions: root, supportedGradientDevices: [])
        XCTAssertTrue(fineTune.arguments(in: root).contains("--finetune-checkpoint"))
        try Data("changed metrics".utf8).write(to: root.appendingPathComponent("metrics.json"))
        XCTAssertThrowsError(try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, continuation: .init(parent: parent, mode: .fineTune), sourcePermissions: root, supportedGradientDevices: []))
    }
    func testLegacyCompositePinAndArchitectureDetachmentArePreserved() throws {
        let root = try root(), registry = try ComponentRegistry.builtIn()
        let expansion = try CompositeExpander.expand(CompositeExamples.parallelClassifier, registry: registry)
        let bytes = try JSONEncoder().encode(CompositeSourceSnapshot(project: CompositeExamples.parallelClassifier, expansion: expansion))
        let url = root.appendingPathComponent("source.json"); try bytes.write(to: url)
        var settings = ExecutionSettings(device: "cpu")
        settings.compositeSource = .init(path: url.path, sha256: WorkflowArchiveCodec.hash(bytes), expandedSemanticSHA256: try CompositeArchitectureIdentity.digest(expansion.graph))
        var graph = expansion.graph; graph.execution = settings
        let plan = try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, sourcePermissions: root, supportedGradientDevices: [])
        XCTAssertEqual(try capturedBytes(plan.inputs, "input/composite/source.json"), bytes)
        XCTAssertTrue(plan.arguments(in: root).contains(WorkflowArchiveCodec.hash(bytes)))
        let index = try XCTUnwrap(graph.nodes.firstIndex { $0.type == "linear" }); graph.nodes[index].config["bias"] = .bool(false)
        let detached = try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, sourcePermissions: root, supportedGradientDevices: [])
        XCTAssertFalse(detached.inputs.contains { $0.path == "input/composite/source.json" })
        try Data("changed".utf8).write(to: url)
        XCTAssertThrowsError(try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings, sourcePermissions: root, supportedGradientDevices: []))
    }
}
