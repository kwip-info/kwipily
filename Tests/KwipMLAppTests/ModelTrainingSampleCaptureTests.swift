import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelTrainingSampleCaptureTests: XCTestCase {
    private let directory = URL(fileURLWithPath: "/private/tmp/workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
    private func plan(_ policy: TrainingSampleCapturePolicy? = nil) throws -> ModelTrainingLaunchPlan {
        let settings = ExecutionSettings(device: "cpu", epochs: 2)
        var graph = ExampleFlows.parallelClassifier; graph.execution = settings
        return try ModelTrainingLaunchPlan.capture(graph: graph, settings: settings,
            sourcePermissions: directory, supportedGradientDevices: [], sampleCapture: policy)
    }
    private func files(_ plan: ModelTrainingLaunchPlan) throws -> [WorkflowArchiveFile] {
        try plan.inputs.map { input in
            guard case .bytes(let path, let data) = input else { throw WorkflowFileLeaseError.unsafePath }
            return .init(path: path, bytes: data.count, sha256: WorkflowArchiveCodec.hash(data))
        }
    }
    func testExplicitCaptureIsPinnedInArgumentsWithoutChangingGraphOrInputs() throws {
        var policy = TrainingSampleCapturePolicy(activationValues: false, predictionRows: true)
        let captured = try plan(policy), legacy = try plan()
        policy = .factory
        let args = captured.arguments(in: directory)
        let index = try XCTUnwrap(args.firstIndex(of: "--sample-capture-json"))
        XCTAssertEqual(args[index + 1], "{\"activationValues\":false,\"predictionRows\":true,\"schemaVersion\":1}")
        XCTAssertNotEqual(try JSONDecoder().decode(TrainingSampleCapturePolicy.self, from: Data(args[index + 1].utf8)), policy)
        XCTAssertEqual(captured.graph, legacy.graph)
        XCTAssertEqual(try files(captured).map(\.path), try files(legacy).map(\.path))
        for (left, right) in zip(captured.inputs, legacy.inputs) {
            guard case .bytes(_, let a) = left, case .bytes(_, let b) = right else { return XCTFail("Expected captured graph bytes") }
            XCTAssertEqual(try JSONDecoder().decode(GraphDocument.self, from: a),
                           try JSONDecoder().decode(GraphDocument.self, from: b))
        }
        XCTAssertFalse(legacy.arguments(in: directory).contains("--sample-capture-json"))
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: args, files: files(captured), directory: directory))
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: legacy.arguments(in: directory), files: files(legacy), directory: directory))
    }
    func testNativeAdmissionRejectsMalformedNullUnknownAndOversizedCapturePolicies() throws {
        let capture = try plan(), args = capture.arguments(in: directory), inventory = try files(capture)
        for invalid in ["null", "{}", "[]", "{\"schemaVersion\":1,\"activationValues\":0,\"predictionRows\":true}",
                        "{\"schemaVersion\":2,\"activationValues\":true,\"predictionRows\":true}",
                        "{\"schemaVersion\":1,\"activationValues\":true,\"predictionRows\":true,\"extra\":false}",
                        String(repeating: " ", count: 1024) + "{\"schemaVersion\":1,\"activationValues\":true,\"predictionRows\":true}"] {
            XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + ["--sample-capture-json", invalid], files: inventory, directory: directory))
        }
        let valid = "{\"schemaVersion\":1,\"activationValues\":false,\"predictionRows\":false}"
        XCTAssertNoThrow(try WorkflowModelTrainingToolContract.validate(arguments: args + ["--sample-capture-json", valid], files: inventory, directory: directory))
        XCTAssertThrowsError(try WorkflowModelTrainingToolContract.validate(arguments: args + ["--sample-capture-json", valid, "--sample-capture-json", valid], files: inventory, directory: directory))
    }
}
