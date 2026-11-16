import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

final class WorkflowSourceProviderTests:XCTestCase {
    private let source=WorkflowNodeAddress(domain:.data,localID:"csv")
    private var repo:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    @MainActor private struct Fixture {
        let root:URL
        let workspace:WorkspaceStore
        let owner:WorkflowAuthoringStore
        let bindings:WorkflowPrivateSourceBindings
        let provider:WorkflowSourceProvider
        let csv:URL
    }
    @MainActor private func setup()throws->Fixture {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(";")]))])
        _=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        workspace.activeProjectID=owner.projectID
        XCTAssertTrue(owner.select(.init(nodeAddresses:[source]),lease:owner.lease()))
        let bindings=try WorkflowPrivateSourceBindings(owner:owner,directory:root)
        let provider=try WorkflowSourceProvider(workspace:workspace,owner:owner,bindings:bindings)
        let csv=root.appendingPathComponent("observations.csv")
        try Data("a;b;label\n1;2;0\n3;4;1\n".utf8).write(to:csv)
        provider.rememberSource={_ in}
        return Fixture(root:root,workspace:workspace,owner:owner,bindings:bindings,provider:provider,csv:csv)
    }
    @MainActor private final class PendingInspection {
        var exit: ((Int32, String, SourceInspectionResult?) -> Void)?
        var digest = ""
        var separator = ""
        var sourceURL: URL?
        var cancellations = 0
        func install(_ provider: WorkflowSourceProvider) {
            provider.inspectionExecute = { _, _, _, _, source, hash, separator, exit in
                self.sourceURL = source; self.digest = hash; self.separator = separator; self.exit = exit
                return WorkflowWorkerHandle(cancel: { self.cancellations += 1 })
            }
        }
        func complete() {
            exit?(0, "", .init(columns: ["a", "b", "label"], rows: [["1", "2", "0"], ["3", "4", "1"]], rowCount: 2, sourceSHA256: digest))
        }
    }
    @MainActor private func waitFor(_ predicate: () -> Bool) async throws {
        for _ in 0..<500 {
            if predicate() { return }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("Source inspection did not reach its expected state")
    }
    @MainActor func testPickerRechecksLeaseSelectionAndBindingRevisionBeforeRememberingGrant() throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        var remembers = 0
        f.provider.rememberSource = { _ in remembers += 1 }
        f.provider.pickSource = {
            _ = f.owner.apply(.rename(address: self.source, name: "Changed during dialog"), lease: f.owner.lease())
            return f.csv
        }
        XCTAssertFalse(f.provider.chooseSource(for: source, lease: f.owner.lease()))
        XCTAssertEqual(remembers, 0); XCTAssertTrue(f.bindings.bindings.isEmpty)
        f.provider.pickSource = {
            _ = f.owner.select(.init(), lease: f.owner.lease()); return f.csv
        }
        XCTAssertFalse(f.provider.chooseSource(for: source, lease: f.owner.lease()))
        XCTAssertEqual(remembers, 0)
        _ = f.owner.select(.init(nodeAddresses: [source]), lease: f.owner.lease())
        f.provider.pickSource = {
            _ = f.bindings.setPath("/tmp/other.csv", for: self.source, lease: f.owner.lease(), expectedRevision: f.bindings.revision)
            return f.csv
        }
        XCTAssertFalse(f.provider.chooseSource(for: source, lease: f.owner.lease()))
        XCTAssertEqual(remembers, 0); XCTAssertEqual(f.bindings.path(for: source), "/tmp/other.csv")
    }
    @MainActor func testReplacementWithPendingParsingDraftPreservesDraftAndCancelKeepsBinding() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        XCTAssertTrue(f.bindings.setPath(f.csv.path, for: source, lease: f.owner.lease(), expectedRevision: 0))
        let context = try WorkflowDataNodeContext(owner: f.owner, address: source, sources: f.provider)
        context.setNumericPending(origin: context.numericOrigin(key: "csvParsingDraft"), key: "csvParsingDraft", pending: true)
        let replacement = f.root.appendingPathComponent("replacement.csv")
        // Fail inspection before worker startup; replacement authority is independent of parsing.
        try Data("a;b\n2;3\n".utf8).write(to: replacement)
        f.provider.rememberSource = { _ in }
        f.provider.pickSource = { nil }
        XCTAssertFalse(context.chooseSource())
        XCTAssertEqual(f.bindings.path(for: source), f.csv.path)
        f.provider.pickSource = { replacement }
        XCTAssertTrue(context.chooseSource())
        XCTAssertEqual(f.bindings.path(for: source), replacement.path)
        XCTAssertTrue(f.owner.hasPendingInspectorEdits)
        // Binding was committed even if the asynchronous inspection is currently active.
        XCTAssertEqual(context.sourceDisplayName, "replacement.csv")
        f.provider.cancel()
        try await waitFor { !f.provider.isInspecting }
    }
    @MainActor func testCapturedSnapshotHeadersPublishOnlyAfterExitAndLiveVerification() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        XCTAssertTrue(f.bindings.setPath(f.csv.path, for: source, lease: f.owner.lease(), expectedRevision: 0))
        let pending = PendingInspection(); pending.install(f.provider)
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        XCTAssertTrue(f.workspace.isPreparingData); XCTAssertTrue(f.owner.isBusy)
        try await waitFor { pending.exit != nil }
        XCTAssertEqual(pending.separator, ";")
        XCTAssertNotEqual(pending.sourceURL?.path, f.csv.path)
        XCTAssertTrue(f.provider.recordedSchema(for: source, in: f.owner.document).isEmpty)
        pending.complete()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertFalse(f.workspace.isPreparingData); XCTAssertFalse(f.owner.isBusy)
        XCTAssertEqual(f.provider.recordedSchema(for: source, in: f.owner.document).map(\.name), ["a", "b", "label"])
        try FileManager.default.removeItem(at: f.csv)
        for _ in 0..<10 {
            XCTAssertEqual(f.provider.recordedSchema(for: source, in: f.owner.document).count, 3)
        }
        XCTAssertEqual(f.provider.bindingDisplayName(for: source), "observations.csv")
        let verifiedStatus = f.provider.sourceInspectionStatus(for: source)
        XCTAssertNotNil(verifiedStatus)
        XCTAssertNotNil(f.owner.apply(.move(addresses:[source],offset:.init(x:10,y:20)),lease:f.owner.lease()))
        XCTAssertEqual(f.provider.sourceInspectionStatus(for: source)?.phase, verifiedStatus?.phase)
        XCTAssertNotNil(f.provider.sourceInspectionPreview(for: source))
        XCTAssertFalse(f.provider.isInspecting, "Moving never starts a source read")
        XCTAssertNotNil(f.owner.apply(.configure(address: source, config: ["separator": .string(",")]), lease: f.owner.lease()))
        XCTAssertTrue(f.provider.recordedSchema(for: source, in: f.owner.document).isEmpty)
    }
    @MainActor func testCancellationHoldsBusyUntilExitAndOldExitCannotReleaseNewReservation() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        XCTAssertTrue(f.bindings.setPath(f.csv.path, for: source, lease: f.owner.lease(), expectedRevision: 0))
        let pending = PendingInspection(); pending.install(f.provider)
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        try await waitFor { pending.exit != nil }
        f.provider.cancel()
        XCTAssertEqual(pending.cancellations, 1)
        XCTAssertTrue(f.provider.isBusy); XCTAssertTrue(f.workspace.isPreparingData)
        let oldExit = pending.exit
        pending.complete()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertFalse(f.workspace.isPreparingData)
        XCTAssertTrue(f.provider.recordedSchema(for: source, in: f.owner.document).isEmpty)
        pending.exit = nil
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        try await waitFor { pending.exit != nil }
        oldExit?(0, "", nil)
        XCTAssertTrue(f.workspace.isPreparingData)
        f.workspace.cancelDataPreparation = {}
        let newReservation = f.workspace.dataPreparationReservationID
        f.provider.invalidate(); pending.complete()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertTrue(f.workspace.isPreparingData); XCTAssertNotNil(f.workspace.cancelDataPreparation)
        XCTAssertEqual(f.workspace.dataPreparationReservationID, newReservation)
        f.workspace.isPreparingData = false; f.workspace.cancelDataPreparation = nil
    }
    @MainActor func testChangedSourceStaleOwnerAndLaunchFailureNeverPublishHeaders() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        XCTAssertTrue(f.bindings.setPath(f.csv.path, for: source, lease: f.owner.lease(), expectedRevision: 0))
        let pending = PendingInspection(); pending.install(f.provider)
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        try await waitFor { pending.exit != nil }
        try Data("x;y;label\n1;2;0\n3;4;1\n".utf8).write(to: f.csv)
        pending.complete()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertTrue(f.provider.recordedSchema(for: source, in: f.owner.document).isEmpty)
        XCTAssertTrue(f.provider.statusMessage.contains("changed"))
        pending.exit = nil
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        try await waitFor { pending.exit != nil }
        f.workspace.activeProjectID = UUID().uuidString; pending.complete()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertFalse(f.workspace.isPreparingData)
        XCTAssertTrue(f.provider.recordedSchema(for: source, in: f.owner.document).isEmpty)
        f.workspace.activeProjectID = f.owner.projectID
        f.provider.inspectionExecute = { _, _, _, _, _, _, _, _ in throw NSError(domain: "Launch failure", code: 1) }
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        try await waitFor { !f.provider.isInspecting }
        XCTAssertFalse(f.workspace.isPreparingData); XCTAssertNil(f.workspace.cancelDataPreparation); XCTAssertFalse(f.owner.isBusy)
    }
    @MainActor func testImmediateCancellationJoinsCaptureAndReleasesReservation() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        XCTAssertTrue(f.bindings.setPath(f.csv.path, for: source, lease: f.owner.lease(), expectedRevision: 0))
        var launches = 0
        f.provider.inspectionExecute = { _, _, _, _, _, _, _, _ in launches += 1; return .init(cancel: {}) }
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        f.provider.cancel()
        XCTAssertTrue(f.workspace.isPreparingData)
        try await waitFor { !f.provider.isInspecting }
        XCTAssertEqual(launches, 0); XCTAssertFalse(f.workspace.isPreparingData); XCTAssertFalse(f.owner.isBusy)
    }
    @MainActor func testActualWorkerInspectsSemicolonSnapshotAndPropagatesRecordedColumns() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        _ = try XCTUnwrap(f.workspace.runtime, "Actual development worker is required")
        XCTAssertTrue(f.bindings.setPath(f.csv.path, for: source, lease: f.owner.lease(), expectedRevision: 0))
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        try await waitFor { !f.provider.isInspecting }
        XCTAssertFalse(f.workspace.isPreparingData)
        XCTAssertEqual(f.provider.recordedSchema(for: source, in: f.owner.document).map(\.name), ["a", "b", "label"], f.provider.statusMessage)
        XCTAssertEqual(f.provider.recordedSchema(for: source, in: f.owner.document).map(\.dtype), ["string", "string", "string"])
    }
    @MainActor func testGrantRememberedButBindingSaveFailureIsExplicitAndDoesNotInspect()throws {
        let f=try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        try Data("blocks sidecar directory".utf8).write(to:f.root.appendingPathComponent("workflow-sources"))
        var remembers=0,launches=0
        f.provider.pickSource={f.csv}
        f.provider.rememberSource={_ in remembers+=1}
        f.provider.inspectionExecute={_,_,_,_,_,_,_,_ in launches+=1;return .init(cancel:{})}
        XCTAssertFalse(f.provider.chooseSource(for:source,lease:f.owner.lease()))
        XCTAssertEqual(remembers,1);XCTAssertEqual(launches,0);XCTAssertTrue(f.bindings.bindings.isEmpty)
        XCTAssertTrue(f.provider.statusMessage.lowercased().contains("grant"))
        XCTAssertFalse(f.workspace.isPreparingData)
    }

    @MainActor func testOmittedSeparatorUsesAuthoritativeRegistryDefaultWithoutChangingSource() async throws {
        let f = try setup(); defer { try? FileManager.default.removeItem(at: f.root) }
        try Data("a,b,label\n1,2,0\n3,4,1\n".utf8).write(to: f.csv)
        XCTAssertNotNil(f.owner.apply(.configure(address: source, config: [:]), lease: f.owner.lease()))
        XCTAssertTrue(f.bindings.setPath(f.csv.path, for: source, lease: f.owner.lease(), expectedRevision: 0))
        let original = f.owner.document
        let pending = PendingInspection(); pending.install(f.provider)
        XCTAssertTrue(f.provider.inspectSource(for: source, lease: f.owner.lease()))
        try await waitFor { pending.exit != nil }
        XCTAssertEqual(pending.separator, ",")
        pending.complete()
        try await waitFor { !f.provider.isInspecting }
        XCTAssertEqual(f.provider.recordedSchema(for: source, in: f.owner.document).map(\.name), ["a", "b", "label"])
        XCTAssertEqual(f.owner.document, original)
    }
    @MainActor func testSourceFailureIsVisibleForExactNodeAndRecoveryClearsIt() async throws {
        let f=try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        XCTAssertTrue(f.bindings.setPath(f.csv.path,for:source,lease:f.owner.lease(),expectedRevision:0))
        let context=try WorkflowDataNodeContext(owner:f.owner,address:source,sources:f.provider)
        let pending=PendingInspection();pending.install(f.provider)
        XCTAssertTrue(context.refreshSource())
        XCTAssertEqual(context.sourceInspectionStatus?.phase,.inspecting)
        try await waitFor{pending.exit != nil};pending.complete()
        try await waitFor{!f.provider.isInspecting}
        XCTAssertEqual(context.sourceInspectionStatus?.phase,.verified)
        XCTAssertNotNil(context.sourceInspectionPreview)
        // Capture failure after a verified preview must replace success visibly,
        // clear sampled rows, and keep the original source binding intact.
        try FileManager.default.removeItem(at:f.csv);pending.exit=nil
        XCTAssertTrue(context.refreshSource())
        try await waitFor{!f.provider.isInspecting}
        let failure=try XCTUnwrap(context.sourceInspectionStatus)
        XCTAssertEqual(failure.phase,.failed)
        XCTAssertEqual(failure.title,"Source preview unavailable")
        XCTAssertFalse(failure.detail.isEmpty)
        XCTAssertTrue(failure.action?.contains("Choose file") == true)
        XCTAssertTrue(failure.action?.contains("Refresh") == true)
        XCTAssertNil(context.sourceInspectionPreview)
        XCTAssertNil(pending.exit,"Capture refusal must not launch a worker")
        XCTAssertEqual(f.bindings.path(for:source),f.csv.path)
        XCTAssertFalse(f.workspace.isPreparingData);XCTAssertFalse(f.owner.isBusy)
        // Restoring access and explicitly refreshing clears the retained failure.
        try Data("a;b;label\n1;2;0\n3;4;1\n".utf8).write(to:f.csv)
        XCTAssertTrue(context.refreshSource())
        XCTAssertEqual(context.sourceInspectionStatus?.phase,.inspecting)
        try await waitFor{pending.exit != nil};pending.complete()
        try await waitFor{!f.provider.isInspecting}
        XCTAssertEqual(context.sourceInspectionStatus?.phase,.verified)
        XCTAssertNil(context.sourceInspectionStatus?.action)
        XCTAssertNotNil(context.sourceInspectionPreview)
    }

    @MainActor func testInspectionFailureNeverAppearsOnAnotherNodeOrBindingRevision() async throws {
        let f=try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        XCTAssertTrue(f.bindings.setPath(f.csv.path,for:source,lease:f.owner.lease(),expectedRevision:0))
        let pending=PendingInspection();pending.install(f.provider)
        XCTAssertTrue(f.provider.inspectSource(for:source,lease:f.owner.lease()))
        try await waitFor{pending.exit != nil}
        pending.exit?(1,"Permission denied by test",nil)
        try await waitFor{!f.provider.isInspecting}
        XCTAssertEqual(f.provider.sourceInspectionStatus(for:source)?.phase,.failed)
        let other=WorkflowNodeAddress(domain:.data,localID:"other")
        XCTAssertNil(f.provider.sourceInspectionStatus(for:other))
        XCTAssertTrue(f.bindings.setPath(f.root.appendingPathComponent("replacement.csv").path,
            for:source,lease:f.owner.lease(),expectedRevision:f.bindings.revision))
        XCTAssertNil(f.provider.sourceInspectionStatus(for:source),"Binding changes retire failure authority")
    }

    @MainActor func testLateFailureCannotOverwriteNewInspectionStatus() async throws {
        let f=try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        XCTAssertTrue(f.bindings.setPath(f.csv.path,for:source,lease:f.owner.lease(),expectedRevision:0))
        let pending=PendingInspection();pending.install(f.provider)
        XCTAssertTrue(f.provider.inspectSource(for:source,lease:f.owner.lease()))
        try await waitFor{pending.exit != nil};let oldExit=pending.exit
        oldExit?(1,"Earlier refusal",nil)
        try await waitFor{!f.provider.isInspecting}
        XCTAssertEqual(f.provider.sourceInspectionStatus(for:source)?.phase,.failed)
        pending.exit=nil
        XCTAssertTrue(f.provider.inspectSource(for:source,lease:f.owner.lease()))
        try await waitFor{pending.exit != nil}
        oldExit?(1,"Late earlier refusal",nil)
        XCTAssertEqual(f.provider.sourceInspectionStatus(for:source)?.phase,.inspecting)
        XCTAssertTrue(f.provider.isInspecting)
        pending.complete();try await waitFor{!f.provider.isInspecting}
        XCTAssertEqual(f.provider.sourceInspectionStatus(for:source)?.phase,.verified)
    }

    @MainActor func testCancellationHasDistinctStatusAndOwnerChangeHidesIt() async throws {
        let f=try setup();defer{try? FileManager.default.removeItem(at:f.root)}
        XCTAssertTrue(f.bindings.setPath(f.csv.path,for:source,lease:f.owner.lease(),expectedRevision:0))
        let pending=PendingInspection();pending.install(f.provider)
        XCTAssertTrue(f.provider.inspectSource(for:source,lease:f.owner.lease()))
        try await waitFor{pending.exit != nil}
        f.provider.cancel();pending.exit?(130,"Cancelled",nil)
        try await waitFor{!f.provider.isInspecting}
        XCTAssertEqual(f.provider.sourceInspectionStatus(for:source)?.phase,.cancelled)
        XCTAssertNil(f.provider.sourceInspectionPreview(for:source))
        XCTAssertNotNil(f.owner.apply(.rename(address:source,name:"Changed source owner revision"),lease:f.owner.lease()))
        XCTAssertNil(f.provider.sourceInspectionStatus(for:source))
    }

}
