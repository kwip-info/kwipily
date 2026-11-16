import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RMSNormIntegrationTests:XCTestCase {
    func testRealManagedTrainingCapturedModelAndScaleOwnership()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertNotNil(workspace.rmsNormRuntimeCapability,"Discovery must independently admit RMSNorm")
        XCTAssertTrue(workspace.loadComponentWorkflowExample("rms_norm",device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        for _ in 0..<1500 {
            if (host.frozenBindings.reference(for:WorkflowExamples.trainAddress) != nil || host.statusMessage.hasPrefix("Trained model unavailable")) && !workspace.isBusy && !host.owner.isBusy {break}
            try await Task.sleep(nanoseconds:20_000_000)
        }
        let reference=try XCTUnwrap(host.frozenBindings.reference(for:WorkflowExamples.trainAddress),host.statusMessage)
        let package=try WorkflowHistoryReader.stageTrainedExport(.init(projectID:reference.projectID,directoryID:try XCTUnwrap(UUID(uuidString:reference.directoryID))),root:root,accountingRoot:workspace.applicationDirectory,storageLimitBytes:500_000_000)
        defer {package.remove()}
        let source=try String(contentsOf:package.directory.appendingPathComponent("model.py"),encoding:.utf8)
        XCTAssertTrue(source.contains("nn.RMSNorm("),source)
        XCTAssertTrue(workspace.beginWorkflowProject(id:host.owner.projectID,forceReload:true),workspace.statusMessage)
        let restored=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(restored.owner.document.nodes.contains {if case .modelPrimitive(let type,_,_ )=$0.payload {return type=="rms_norm"};return false})
    }
}
