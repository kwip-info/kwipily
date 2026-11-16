import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ProjectSettingsAuthorityTests:XCTestCase {
    private func fixture() throws -> WorkspaceStore {
        _ = NSApplication.shared
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let windows=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)})
        let window=try windows.newFlow(present:false)
        addTeardownBlock {await MainActor.run {for window in windows.openControllers {_=windows.close(window)};try? FileManager.default.removeItem(at:directory)}}
        return window.session.workspace
    }
    func testSettingsRevisionIsSeparateAndResetPreservesOtherGroups() throws {
        let store=try fixture(), id=try XCTUnwrap(store.activeProjectID)
        let original=try store.projectSettingsSnapshot(projectID:id)
        let capture=try store.previewProjectSettings(.trainingCapture(.init(activationValues:false,predictionRows:false)),baseline:original)
        let afterCapture=try store.applyProjectSettings(capture)
        XCTAssertNotEqual(original.revision,afterCapture.revision)
        XCTAssertEqual(original.documentRevision,afterCapture.documentRevision)
        let afterResource=try store.applyProjectSettings(store.previewProjectSettings(.execution(try .init(cpuThreads:3,memory:.unlimited)),baseline:afterCapture))
        let reset=try store.applyProjectSettings(store.previewProjectSettings(.reset(.execution),baseline:afterResource))
        XCTAssertNil(reset.overrides.cpuThreads)
        XCTAssertEqual(reset.overrides.sampleCapture,afterCapture.overrides.sampleCapture)
        XCTAssertEqual(reset.history,original.history)
    }
    func testGlobalChangeInvalidatesNativeDraftWithoutGraphChange() throws {
        let store=try fixture(), id=try XCTUnwrap(store.activeProjectID)
        let baseline=try store.projectSettingsSnapshot(projectID:id)
        let proposal=try store.previewProjectSettings(.execution(try .init(cpuThreads:3)),baseline:baseline)
        XCTAssertTrue(store.setExecutionPreferences(try .init(cpuThreads:2)))
        XCTAssertThrowsError(try store.applyProjectSettings(proposal))
        XCTAssertThrowsError(try store.previewProjectSettings(.derivativeCapture(.init(values:true)),baseline:baseline))
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID:id),.factory)
    }
    func testAtomicGroupsAndCorruptPolicyRefuseWithoutWriting() throws {
        let store=try fixture(), id=try XCTUnwrap(store.activeProjectID)
        let baseline=try store.projectSettingsSnapshot(projectID:id)
        let proposal=try store.previewProjectSettings(.historyCapture(.none),baseline:baseline)
        let changed=try store.applyProjectSettings(proposal)
        XCTAssertEqual(changed.history,.none)
        XCTAssertEqual(changed.overrides,baseline.overrides)
        XCTAssertThrowsError(try store.applyProjectSettings(proposal))
        let invalid=TrainingSampleCapturePolicy(activationValues:true,predictionRows:true,tensorHeatmaps:false,heatmapFrameLimit:2)
        XCTAssertThrowsError(try store.previewProjectSettings(.trainingCapture(invalid),baseline:changed))
        try Data("corrupt".utf8).write(to:store.appPreferenceStore.fileURL)
        XCTAssertThrowsError(try store.projectSettingsSnapshot(projectID:id))
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID:id),baseline.overrides)
    }
    func testActiveProjectRefuses() throws {
        let store=try fixture(), id=try XCTUnwrap(store.activeProjectID)
        let baseline=try store.projectSettingsSnapshot(projectID:id)
        let proposal=try store.previewProjectSettings(.derivativeCapture(.init(values:true)),baseline:baseline)
        store.activeProjectID=UUID().uuidString
        XCTAssertThrowsError(try store.applyProjectSettings(proposal))
        XCTAssertEqual(try store.projectExecutionPreferenceStore.load(projectID:id),.factory)
    }
    func testLegacyNativeFormsPreserveExistingProjectsWithoutEnablingMCP() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let store=WorkspaceStore(storageOverride:directory,detectCapabilities:false)
        let id=UUID().uuidString;store.activeProjectID=id
        XCTAssertThrowsError(try store.projectSettingsSnapshot(projectID:id))
        let baseline=try store.projectSettingsSnapshot(projectID:id,allowLegacy:true)
        let saved=try store.applyProjectSettings(store.previewProjectSettings(.execution(try .init(cpuThreads:2)),baseline:baseline))
        XCTAssertEqual(saved.overrides.cpuThreads,2)
        XCTAssertThrowsError(try store.previewProjectSettings(.historyCapture(.none),baseline:saved))
        XCTAssertEqual(try store.executionPolicy(for:id).budget.resources?.cpuThreads,2)
    }
    func testBusyAndChangedDocumentRefuseAndMemoryRemainsThreeState() throws {
        let store=try fixture(),id=try XCTUnwrap(store.activeProjectID)
        let baseline=try store.projectSettingsSnapshot(projectID:id)
        let proposal=try store.previewProjectSettings(.execution(try .init(memory:.unlimited)),baseline:baseline)
        store.isInspectingData=true
        XCTAssertThrowsError(try store.applyProjectSettings(proposal))
        store.isInspectingData=false
        let after=try store.applyProjectSettings(proposal)
        XCTAssertEqual(after.overrides.memory,.unlimited)
        XCTAssertEqual(after.effective.maximumResidentBytesSource,.project)
        let reset=try store.applyProjectSettings(store.previewProjectSettings(.reset(.execution),baseline:after))
        XCTAssertEqual(reset.overrides.memory,.inherit)
        XCTAssertEqual(reset.effective.maximumResidentBytesSource,.application)
        let next=try store.previewProjectSettings(.derivativeCapture(.init(values:true)),baseline:reset)
        let owner=try XCTUnwrap(store.workflowSession?.owner)
        XCTAssertNotNil(owner.apply(.insert(.init(address:.init(domain:.control,localID:"late"),payload:.saveModel(config:[:]))),lease:owner.lease()))
        XCTAssertThrowsError(try store.applyProjectSettings(next))
    }

    func testPendingInspectorDraftAndNativeWriteInvalidateProposal() throws {
        let store=try fixture(),id=try XCTUnwrap(store.activeProjectID),owner=try XCTUnwrap(store.workflowSession?.owner)
        let address=WorkflowNodeAddress(domain:.control,localID:"save")
        XCTAssertNotNil(owner.apply(.insert(.init(address:address,payload:.saveModel(config:[:]))),lease:owner.lease()))
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let baseline=try store.projectSettingsSnapshot(projectID:id)
        let proposal=try store.previewProjectSettings(.trainingCapture(.init(activationValues:false,predictionRows:false)),baseline:baseline)
        let editor=UUID()
        XCTAssertTrue(owner.setInspectorEditPending(contextID:editor,address:address,key:"name",pending:true,lease:owner.lease()))
        XCTAssertThrowsError(try store.applyProjectSettings(proposal))
        owner.clearInspectorEdits(contextID:editor)
        XCTAssertTrue(store.setProjectDerivativeCapture(.init(values:true),projectID:id))
        XCTAssertThrowsError(try store.applyProjectSettings(proposal))
        let latest=try store.projectSettingsSnapshot(projectID:id)
        XCTAssertTrue(latest.overrides.derivativeCapture?.values == true)
        XCTAssertNil(latest.overrides.sampleCapture)
        let accepted=latest.effective
        _ = try store.applyProjectSettings(store.previewProjectSettings(.derivativeCapture(.factory),baseline:latest))
        XCTAssertTrue(accepted.derivativeCapture.values,"A previously accepted immutable policy does not change")
        let reopened=WorkspaceStore(storageOverride:store.applicationDirectory,detectCapabilities:false)
        XCTAssertEqual(try reopened.executionPolicy(for:id),try store.executionPolicy(for:id))
    }

    func testPerformanceConvenienceReportsPartialSecondStoreFailure() throws {
        let store=try fixture(),host=try XCTUnwrap(store.workflowSession)
        try host.owner.setRunHistoryPolicy(.none)
        let historyDirectory=store.workspaceDirectory.appendingPathComponent("run-history-policy")
        try FileManager.default.setAttributes([.posixPermissions:0o500],ofItemAtPath:historyDirectory.path)
        defer {try? FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:historyDirectory.path)}
        XCTAssertFalse(host.setPerformanceCaptureEnabled(true))
        XCTAssertTrue(try store.executionPolicy(for:host.owner.projectID).sampleCapture.performanceMeasurements)
        XCTAssertEqual(host.owner.runHistoryPolicy,.none)
        XCTAssertFalse(host.performanceCaptureEnabled)
        XCTAssertTrue(host.statusMessage.contains("choice saved"),host.statusMessage)
        XCTAssertTrue(host.statusMessage.contains("History"),host.statusMessage)
        XCTAssertTrue(host.statusMessage.contains("off"),host.statusMessage)
    }

}
