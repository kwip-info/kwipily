import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModelHostExecutionPolicyTests: XCTestCase {
    func testTrainingAndPreviewCaptureProjectPolicyAndRefuseTinyStorageBeforeLaunch() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for preview in [false, true] {
            for refuse in [false, true] {
                let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
                defer { try? FileManager.default.removeItem(at: root) }
                let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
                await verifyRuntimeForIntegration(store)
                store.loadExample(); store.setExecution { $0.device = "cpu"; $0.epochs = 1 }
                XCTAssertTrue(store.saveManagedProject())
                let project = try XCTUnwrap(store.activeProjectID)
                XCTAssertTrue(store.setExecutionPreferences(try .init(maximumRunSeconds: 120, managedStorageLimitBytes: refuse ? 1 : 2_000_000_000)))
                XCTAssertTrue(store.setProjectExecutionOverrides(try .init(maximumRunSeconds: 90, cpuThreads: 2, memory: .limit(1_073_741_824)), projectID: project))
                let hidden = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false)
                XCTAssertTrue(store.setProjectSampleCapture(hidden, projectID: project))
                var captured: WorkflowToolRequest?
                let launch: WorkflowWorkerLauncher = { _, arguments, _, _, exit in
                    let index = try XCTUnwrap(arguments.firstIndex(of: "--kwip-operation-supervise"))
                    captured = try WorkflowToolRequest.decode(Data(contentsOf: URL(fileURLWithPath: arguments[index + 1]).appendingPathComponent("request.json")))
                    exit(143, "Test stopped after capture")
                    return .init(cancel: {})
                }
                store.trainingWorkerLauncher = launch; store.previewWorkerLauncher = launch
                if preview { store.selectNode("output"); store.previewSelected() } else { store.run() }
                let deadline = Date().addingTimeInterval(15)
                while (store.isRunning || store.isPreviewing) && Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
                XCTAssertFalse(store.isRunning || store.isPreviewing)
                if refuse { XCTAssertNil(captured, "Insufficient storage must refuse before worker launch") }
                else {
                    let request = try XCTUnwrap(captured, store.runError ?? store.statusMessage)
                    if preview { XCTAssertFalse(request.arguments.contains("--sample-capture-json")) }
                    else {
                        let index = try XCTUnwrap(request.arguments.firstIndex(of: "--sample-capture-json"))
                        XCTAssertEqual(try JSONDecoder().decode(TrainingSampleCapturePolicy.self,
                            from: Data(request.arguments[index + 1].utf8)), hidden)
                    }
                    XCTAssertEqual(request.executionBudget.maximumRunSeconds, 90)
                    XCTAssertEqual(request.executionBudget.source, .project)
                    XCTAssertEqual(request.executionBudget.resources?.cpuThreads, 2)
                    XCTAssertEqual(request.executionBudget.resources?.maximumResidentBytes, 1_073_741_824)
                }
            }
        }
    }
}
