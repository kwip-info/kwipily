import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LossNativePreparationTests: XCTestCase {
    private let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    private func run(_ args: [String], log: URL) throws -> String {
        let python = repo.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw XCTSkip("Actual local Python runtime is required") }
        FileManager.default.createFile(atPath: log.path, contents: Data())
        let output = try FileHandle(forWritingTo: log); defer { try? output.close() }
        let process = Process(); process.executableURL = python; process.currentDirectoryURL = repo
        process.arguments = [repo.appendingPathComponent("runner/main.py").path] + args
        process.standardOutput = output; process.standardError = output
        try process.run(); let deadline = Date().addingTimeInterval(40)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.025) }
        if process.isRunning { process.terminate(); XCTFail("Actual worker exceeded its bounded test deadline") }
        process.waitUntilExit()
        let bytes = try Data(contentsOf: log); XCTAssertLessThan(bytes.count, 2000000)
        let text = String(decoding: bytes, as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, String(text.suffix(3000)))
        return text
    }
    private func fixture(_ task: LossLearningTask, at root: URL) throws -> WorkflowPhaseExpectation {
        let modelRegistry = try ComponentRegistry.builtIn()
        let dataRegistry = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let plan = try WorkflowCompiler.compile(WorkflowExamples.preparedLossObjective(task: task, device: "cpu"), train: WorkflowExamples.trainAddress, modelRegistry: modelRegistry, dataRegistry: dataRegistry, contractVersion: 12)
        let graph = root.appendingPathComponent("dataflow.json")
        try DataFlowCodec.encodeGraph(plan.dataflow).write(to: graph)
        let resource = try XCTUnwrap(BundledLossComponents.entries[task.componentID]?.learning?.sourceResource)
        let bindings = root.appendingPathComponent("bindings.json")
        try WorkflowPhaseArtifacts.encode(["observations":repo.appendingPathComponent("Sources/KwipMLApp/Resources/"+resource+".csv").path]).write(to: bindings)
        let output = root.appendingPathComponent("preparation/artifacts")
        try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
        _ = try run(["--execute-dataflow",graph.path,"--source-bindings",bindings.path,"--dataflow-cache-policy",plan.preparationCachePolicy.rawValue,"--output",output.path], log: root.appendingPathComponent("preparation.log"))
        return .init(plan: plan, modelRegistrySHA256: try WorkflowArchiveCodec.registryHash(modelRegistry), dataRegistrySHA256: try WorkflowArchiveCodec.registryHash(dataRegistry), compilerContractVersion: 12)
    }
    func testEveryRealPreparedObjectiveProducesExactNativeTrainingArgumentsAndRuns() throws {
        for task in LossLearningTask.allCases {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let expected = try fixture(task, at: root)
            do {
                let prepared = try WorkflowPhaseArtifacts.preparation(expectation: expected, root: root)
                guard case .objective(let selected) = prepared.selection else { XCTFail("Typed objective was reduced to legacy scalar mapping"); continue }
                XCTAssertEqual(selected.targetMapping, expected.plan.lossRecipe?.targetMapping)
                XCTAssertEqual(selected.objectiveRecipe, expected.plan.lossRecipe?.configuration.recipeValue)
                let args = try WorkflowPhaseArtifacts.preparedArguments(prepared.selection)
                XCTAssertFalse(args.contains("--target-column")); XCTAssertFalse(args.contains("--feature-columns-json"))
                XCTAssertTrue(args.contains("--objective-recipe-json")); XCTAssertTrue(args.contains("--objective-target-mapping-json"))
                let graph = root.appendingPathComponent("model.json")
                try WorkflowPhaseArtifacts.encode(expected.plan.model.graph).write(to: graph)
                let out = root.appendingPathComponent("trained")
                _ = try run(["--graph",graph.path,"--output",out.path,"--device","cpu","--epochs","3","--batch-size","32","--seed","42","--learning-rate","0.03"]+args, log:root.appendingPathComponent("training.log"))
                let manifest = try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:out.appendingPathComponent("manifest.json")))
                guard case .object(let settings) = manifest["settings"] else { XCTFail("Missing saved objective settings"); continue }
                XCTAssertEqual(settings["objectiveRecipe"],try JSONDecoder().decode(JSONValue.self,from:WorkflowPhaseArtifacts.encode(selected.objectiveRecipe)))
                XCTAssertEqual(settings["targetMapping"],try JSONDecoder().decode(JSONValue.self,from:WorkflowPhaseArtifacts.encode(selected.targetMapping)))
                XCTAssertTrue(FileManager.default.fileExists(atPath:out.appendingPathComponent("objective_recipe.py").path))
            } catch { XCTFail("\(task.rawValue): \(error)") }
        }
    }
    func testNativePreparationRefusesChangedObservedArtifact() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);defer { try? FileManager.default.removeItem(at:root) }
        let expected = try fixture(.triplet,at:root)
        do { _ = try WorkflowPhaseArtifacts.preparation(expectation:expected,root:root) } catch { XCTFail("Real typed preparation must verify before tampering: \(error)") }
        let artifacts = root.appendingPathComponent("preparation/artifacts")
        let table = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: artifacts, includingPropertiesForKeys: nil).first { $0.lastPathComponent.hasPrefix("node-") })
        for file in [artifacts.appendingPathComponent("dataflow.json"), table] {
            let original = try Data(contentsOf: file)
            var changed = original; changed.append(32); try changed.write(to: file)
            XCTAssertThrowsError(try WorkflowPhaseArtifacts.preparation(expectation: expected, root: root), file.lastPathComponent)
            try original.write(to: file)
        }
        let moved = root.appendingPathComponent("moved-preparation")
        try FileManager.default.moveItem(at: root.appendingPathComponent("preparation"), to: moved)
        try FileManager.default.createSymbolicLink(at: root.appendingPathComponent("preparation"), withDestinationURL: moved)
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.preparation(expectation: expected, root: root), "Artifact hashing must reject descendant directory links.")
    }
    func testLegacyPreparedArgumentsRetainTheirExactOrderAndMappingFlags() throws {
        let old = PreparedTrainingSelection(manifestPath:"/prepared/manifest.json",manifestSHA256:String(repeating:"a",count:64),graphDigest:String(repeating:"b",count:64),trainingNode:"t",trainingPort:"table",validationNode:"v",validationPort:"table",featureColumns:["x","y"],targetColumn:"class")
        let direct = try WorkflowPhaseArtifacts.preparedArguments(old)
        XCTAssertEqual(try WorkflowPhaseArtifacts.preparedArguments(.legacy(old)), direct)
        XCTAssertEqual(Array(direct.suffix(4)),["--feature-columns-json","[\"x\",\"y\"]","--target-column","class"])
    }
}
