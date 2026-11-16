import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TrainMappingPublicationTests:XCTestCase {
    func testVisiblePredictionReadDoesNotPreventCompletedModelPublication()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        XCTAssertTrue(owner.select(.init(nodeAddresses:[WorkflowExamples.trainAddress]),lease:owner.lease()))
        XCTAssertTrue(host.run(),host.statusMessage)
        let launchDeadline=Date().addingTimeInterval(5)
        while host.execution==nil && Date()<launchDeadline {try await Task.sleep(for:.milliseconds(20))}
        let execution=try XCTUnwrap(host.execution)
        var query:TableViewQuerySession?
        let observed=execution.$phase.filter {$0 == .completed}.prefix(1).sink { _ in
            // Mimic the selected Train's Output viewer scheduled by completion.
            Task { @MainActor in
                query=host.predictionTableViewSession(for:.init(address:WorkflowExamples.trainAddress,port:"predictions"))
                XCTAssertTrue(query?.load() == true,query?.statusMessage ?? "No prediction query")
            }
        }
        defer {observed.cancel()}
        let deadline=Date().addingTimeInterval(20)
        while Date()<deadline && (host.frozenBindings.reference(for:WorkflowExamples.trainAddress)==nil || query?.page==nil) {
            try await Task.sleep(for:.milliseconds(20))
            if host.statusMessage.hasPrefix("Trained model unavailable:") && query?.isBusy==false {break}
        }
        XCTAssertNotNil(query?.page,query?.statusMessage ?? "No prediction page")
        XCTAssertEqual(query?.page?.totalRows,16)
        XCTAssertNotNil(host.frozenBindings.reference(for:WorkflowExamples.trainAddress),host.statusMessage)
    }
}
