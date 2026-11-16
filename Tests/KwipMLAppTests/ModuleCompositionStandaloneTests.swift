import Foundation
import Darwin
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModuleCompositionStandaloneTests: XCTestCase {
    func testStandaloneSourceThreeCaptureUsesVerifiedSourceDrivenExpansionVersion() throws {
        var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        document.schemaVersion = 11
        document.definitions[0].composition = .init(kind: .graph, children: document.definitions[0].body.nodes.map { .init(localID: $0.id) })
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let p = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress, modelRegistry: .builtIn(), dataRegistry: .load(from: repository.appendingPathComponent("runner/dataflow-operators.json")), contractVersion: 10)
        let source = try XCTUnwrap(p.model.provenance)
        // Secure input readers refuse symlink ancestors, including macOS /var aliases.
        let pointer = try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path, nil))
        defer { free(pointer) }
        let root = URL(fileURLWithPath: String(cString: pointer)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = try WorkflowArchiveCodec.encode(source), url = root.appendingPathComponent("source.json")
        try bytes.write(to: url)
        var settings = ExecutionSettings(); settings.device = "cpu"
        settings.compositeSource = .init(path: url.path, sha256: WorkflowArchiveCodec.hash(bytes), expandedSemanticSHA256: p.model.semanticSHA256)
        let captured = try ModelTrainingLaunchPlan.capture(graph: p.model.graph, settings: settings,
            sourcePermissions: root.appendingPathComponent("grants"), supportedGradientDevices: [], registry: .builtIn())
        XCTAssertTrue(captured.arguments(in: root).contains("--composite-source"))
        XCTAssertTrue(captured.arguments(in: root).contains(WorkflowArchiveCodec.hash(bytes)))
        XCTAssertThrowsError(try ModelTrainingLaunchPlan.capture(graph: p.model.graph, settings: settings,
            sourcePermissions: root.appendingPathComponent("grants"), supportedGradientDevices: []))
        let directory = root.appendingPathComponent("workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
        var files: [WorkflowArchiveFile] = []
        for input in captured.inputs {
            guard case .bytes(let path, let data) = input else { return XCTFail("Expected immutable source fixture bytes") }
            let destination = directory.appendingPathComponent(path)
            try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: destination)
            files.append(.init(path: path, bytes: data.count, sha256: WorkflowArchiveCodec.hash(data)))
        }
        let arguments = captured.arguments(in: directory)
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.verifyInputs(arguments: arguments, files: files, directory: directory))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.verifyInputs(arguments: arguments, files: files.filter { $0.path != "input/model-registry.json" }, directory: directory))
        let changed = Data("{\"schemaVersion\":1,\"components\":[]}".utf8)
        try changed.write(to: directory.appendingPathComponent("input/model-registry.json"))
        files = files.map { $0.path == "input/model-registry.json" ? .init(path: $0.path, bytes: changed.count, sha256: WorkflowArchiveCodec.hash(changed)) : $0 }
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.verifyInputs(arguments: arguments, files: files, directory: directory))
    }
}
