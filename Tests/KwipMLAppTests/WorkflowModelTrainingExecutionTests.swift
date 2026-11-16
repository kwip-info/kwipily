import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowModelTrainingExecutionTests: XCTestCase {
    func testActualEarlyStopAndHeatmapPlaybackPassNativeEvidenceVerification() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        let runtime=try XCTUnwrap(workspace.runtime)
        let policy=TrainingControlPolicy(modelSelection:"bestValidation",earlyStopping:.init(enabled:true,patience:2,minimumImprovement:1_000_000))
        let capture=TrainingSampleCapturePolicy.factory.changing(activationValues:false,predictionRows:false,tensorHeatmaps:true,heatmapFrameLimit:6)
        var graph=ExampleFlows.parallelClassifier
        var settings=ExecutionSettings();settings.epochs=8;settings.device="cpu";settings.sampleLimit=20;settings.trainingControl=policy;graph.execution=settings
        let bytes=try JSONEncoder().encode(graph)
        let controlJSON=String(decoding:try JSONEncoder().encode(policy),as:UTF8.self)
        let captureJSON=String(decoding:try JSONEncoder().encode(capture),as:UTF8.self)
        let finished=expectation(description:"Verified early stop and heatmaps")
        let window=try ExecutionBudgetWindow(policy:.resolve(global:.factory,projectMaximumRunSeconds:nil),startedAtUptime:ProcessInfo.processInfo.systemUptime)
        let handle=try await WorkflowModelTrainingExecution.launch(workspace:root,projectID:UUID().uuidString,runtime:runtime,window:window,
            inputs:{_ in [.bytes(path:"input/graph.json",data:bytes)]},arguments:{owner in ["--graph",owner.appendingPathComponent("input/graph.json").path,"--output",owner.appendingPathComponent("artifacts").path,"--device","cpu","--seed",String(settings.seed),"--epochs","8","--batch-size",String(settings.batchSize),"--learning-rate",String(settings.learningRate),"--sample-limit","20","--split-seed",String(settings.splitSeed),"--validation-percent",String(settings.validationPercent),"--training-control-json",controlJSON,"--sample-capture-json",captureJSON]},
            onLine:{_ in},onExit:{code,message,output in XCTAssertEqual(code,0,message);XCTAssertNotNil(output);finished.fulfill()})
        _=handle
        await fulfillment(of:[finished],timeout:40)
    }
    func testSampledResidentMemoryRefusalRetainsMeaningfulTerminalReason() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime)
        var graph = ExampleFlows.parallelClassifier
        var settings = ExecutionSettings(); settings.device = "cpu"; settings.epochs = 1; graph.execution = settings
        let bytes = try JSONEncoder().encode(graph)
        let resources = try ExecutionResourcePolicy(cpuThreads: 2, maximumResidentBytes: 134_217_728)
        let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil, resources: resources), startedAtUptime: ProcessInfo.processInfo.systemUptime)
        let finished = expectation(description: "memory policy refusal")
        let handle = try await WorkflowModelTrainingExecution.launch(workspace: root, projectID: UUID().uuidString, runtime: runtime,
            window: window, inputs: { _ in [.bytes(path: "input/graph.json", data: bytes)] },
            arguments: { owner in ["--graph", owner.appendingPathComponent("input/graph.json").path,
                "--output", owner.appendingPathComponent("artifacts").path, "--device", "cpu", "--seed", String(settings.seed),
                "--epochs", "1", "--batch-size", String(settings.batchSize), "--learning-rate", String(settings.learningRate),
                "--sample-limit", "20", "--split-seed", String(settings.splitSeed), "--validation-percent", String(settings.validationPercent)] },
            onLine: { _ in }, onExit: { code, message, output in
                XCTAssertEqual(code, 125, message); XCTAssertNil(output)
                XCTAssertTrue(message.lowercased().contains("sampled resident memory"), message)
                finished.fulfill()
            })
        _ = handle
        await fulfillment(of: [finished], timeout: 30)
    }
    func testActualTrainingVerificationJoinsExitAndRejectsChangedEvidence() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        let runtime = try XCTUnwrap(workspace.runtime)
        var graph = ExampleFlows.parallelClassifier
        var settings = ExecutionSettings(); settings.epochs = 1; settings.device = "cpu"; settings.sampleLimit = 20
        graph.execution = settings
        let graphBytes = try JSONEncoder().encode(graph)
        for scenario in ["performanceValid", "performanceMismatch", "captureValid", "captureMismatch", "captureLeak", "valid", "wrongRun", "wrongPrediction", "wrongGradient", "duplicateCompletion", "resourcePolicy", "resourceMismatch", "metrics", "lifecycle", "cancel", "timeout"] {
            let operationID = UUID().uuidString
            let resources = scenario.hasPrefix("resource") ? try ExecutionResourcePolicy(cpuThreads: 2) : nil
            let window = try ExecutionBudgetWindow(policy: .resolve(global: .factory, projectMaximumRunSeconds: nil, resources: resources), startedAtUptime: ProcessInfo.processInfo.systemUptime)
            var directory: URL?, pendingExit: (() -> Void)?, callbacks = 0
            let held = expectation(description: "worker " + scenario), finished = expectation(description: "verified " + scenario)
            let handle = try await WorkflowModelTrainingExecution.launch(workspace: root, projectID: UUID().uuidString,
                runtime: runtime, window: window, operationID: operationID, storageLimitBytes: 450_000_000,
                inputs: { _ in [.bytes(path: "input/graph.json", data: graphBytes)] },
                arguments: { owner in ["--graph", owner.appendingPathComponent("input/graph.json").path,
                    "--output", owner.appendingPathComponent("artifacts").path, "--device", "cpu", "--seed", String(settings.seed),
                    "--epochs", "1", "--batch-size", String(settings.batchSize), "--learning-rate", String(settings.learningRate),
                    "--sample-limit", "20", "--split-seed", String(settings.splitSeed), "--validation-percent", String(settings.validationPercent)] + (scenario.hasPrefix("capture") ? ["--sample-capture-json", #"{"schemaVersion":1,"activationValues":false,"predictionRows":false}"#] : scenario.hasPrefix("performance") ? ["--sample-capture-json", #"{"schemaVersion":2,"activationValues":false,"predictionRows":false,"performanceMeasurements":true}"#] : []) },
                launcher: { executable, args, cwd, line, exit in
                    let index = try XCTUnwrap(args.firstIndex(of: "--kwip-operation-supervise"))
                    let owner = URL(fileURLWithPath: args[index + 1]); directory = owner
                    XCTAssertEqual(owner.lastPathComponent, operationID)
                    return try WorkflowExecutionSession.launchWorker(executable, args, cwd, { text in
                        if let data = text.data(using: .utf8),
                           var event = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any], event["event"] as? String == "completed" {
                            switch scenario {
                            case "performanceMismatch": event["performanceCapture"] = ["schemaVersion":1,"file":"performance.json","bytes":1,"sha256":String(repeating:"a",count:64)]
                            case "captureMismatch": event["sampleCapture"] = ["schemaVersion":1,"activationValues":true,"predictionRows":true]
                            case "captureLeak":
                                var previews = event["previews"] as! [String: [String: Any]]
                                let first = previews.keys.sorted().first!
                                previews[first]!["values"] = [42]
                                event["previews"] = previews
                            case "wrongRun": event["runID"] = UUID().uuidString
                            case "wrongPrediction": event["predictionRows"] = []
                            case "wrongGradient": event["gradientCapture"] = ["schemaVersion": 1, "file": "gradient-summary.json"]
                            case "duplicateCompletion": line(text)
                            default: break
                            }
                            line(String(decoding: try! JSONSerialization.data(withJSONObject: event), as: UTF8.self))
                        } else { line(text) }
                    }, { code, message in
                        pendingExit = { exit(scenario == "timeout" ? 124 : code, message) }; held.fulfill()
                    })
                }, onLine: { _ in }, onExit: { code, message, output in
                    callbacks += 1
                    if scenario == "valid" || scenario == "resourcePolicy" || scenario == "captureValid" || scenario == "performanceValid" { XCTAssertEqual(code, 0, message); XCTAssertEqual(output, directory?.appendingPathComponent("artifacts")) }
                    else { XCTAssertNotEqual(code, 0); XCTAssertNil(output) }
                    finished.fulfill()
                })
            await fulfillment(of: [held], timeout: 30)
            let owner = try XCTUnwrap(directory), output = owner.appendingPathComponent("artifacts")
            XCTAssertEqual(callbacks, 0)
            XCTAssertThrowsError(try WorkflowFileLease(directory: owner, name: "owner.lock", mode: .shared))
            if scenario.hasPrefix("resource") {
                let manifest = output.appendingPathComponent("manifest.json")
                var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any])
                var checkpoint = try XCTUnwrap(object["checkpoint"] as? [String: Any])
                var contract = try XCTUnwrap(checkpoint["contract"] as? [String: Any])
                var schedule = try XCTUnwrap(contract["schedule"] as? [String: Any])
                XCTAssertEqual(schedule["numThreads"] as? Int, 2, "The actual worker must honor captured CPU policy")
                if scenario == "resourceMismatch" {
                    schedule["numThreads"] = 1; contract["schedule"] = schedule; checkpoint["contract"] = contract; object["checkpoint"] = checkpoint
                    try JSONSerialization.data(withJSONObject: object).write(to: manifest)
                }
            }
            if scenario == "metrics" { try Data("[]".utf8).write(to: output.appendingPathComponent("metrics.json")) }
            if scenario == "lifecycle" {
                let manifest = output.appendingPathComponent("manifest.json")
                var object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: manifest)) as? [String: Any])
                object["lineage"] = ["mode": "exactResume"]
                try JSONSerialization.data(withJSONObject: object).write(to: manifest)
            }
            if scenario == "cancel" { handle.cancel() }
            pendingExit?()
            await fulfillment(of: [finished], timeout: 10)
            XCTAssertEqual(callbacks, 1)
            XCTAssertFalse(FileManager.default.fileExists(atPath: owner.appendingPathComponent("input").path))
            let released = try WorkflowFileLease(directory: owner, name: "owner.lock", mode: .exclusive); released.close()
        }
    }
}
