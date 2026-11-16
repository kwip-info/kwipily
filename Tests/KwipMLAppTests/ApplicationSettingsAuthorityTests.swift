import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ApplicationSettingsAuthorityTests:XCTestCase {
    private func fixture() -> WorkspaceStore {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        return WorkspaceStore(storageOverride:root,detectCapabilities:false)
    }
    func testStorageAndCaptureEditsPreserveVersionedWorkloadSelection() throws {
        let store = fixture()
        let policy = try ExecutionWorkloadPolicy(preset: .balanced)
        var state = try store.applicationSettingsSnapshot()
        state = try store.applyApplicationSettings(store.previewApplicationSettings(.execution(try .init(workloadPolicy: policy)), baseline: state))
        XCTAssertEqual(state.preferences.execution.workloadPolicy, policy)
        let accepted = try store.executionPolicy(for: nil)
        state = try store.applyApplicationSettings(store.previewApplicationSettings(.storage(try .init(preparationCacheLimitBytes: 100)), baseline: state))
        XCTAssertEqual(state.preferences.execution.workloadPolicy, policy)
        state = try store.applyApplicationSettings(store.previewApplicationSettings(.trainingCapture(.factory), baseline: state))
        XCTAssertEqual(state.preferences.execution.workloadPolicy, policy)
        var draft = ExecutionSettingsDraft(global: state.preferences.execution)
        draft.threads = "2"
        XCTAssertEqual(try draft.application().workloadPolicy, policy)
        let reset = try store.applyApplicationSettings(store.previewApplicationSettings(.reset(.execution), baseline: state))
        XCTAssertNil(reset.preferences.execution.workloadPolicy)
        XCTAssertEqual(accepted.workloadPolicy, policy, "An accepted operation keeps its capture after preference reset")
    }
    func testFullFingerprintRejectsConcurrentGroupChangeAndNoOpIsStable() throws {
        let store=fixture(),baseline=try store.applicationSettingsSnapshot()
        let proposal=try store.previewApplicationSettings(.execution(try .init(cpuThreads:3)),baseline:baseline)
        XCTAssertTrue(store.setAppearance(.init(mode:.light)))
        XCTAssertThrowsError(try store.applyApplicationSettings(proposal))
        XCTAssertEqual(try store.appPreferenceStore.load().execution.cpuThreads,1)
        let fresh=try store.applicationSettingsSnapshot()
        let unchanged=try store.applyApplicationSettings(store.previewApplicationSettings(.appearance(fresh.preferences.appearance),baseline:fresh))
        XCTAssertEqual(fresh.revision,unchanged.revision)
    }
    func testGroupsAndResetPreserveOtherSettingsAndAcceptedPolicies() throws {
        let store=fixture()
        let accepted=try store.executionPolicy(for:nil)
        var before=try store.applicationSettingsSnapshot()
        before=try store.applyApplicationSettings(store.previewApplicationSettings(.trainingCapture(.init(activationValues:false,predictionRows:false)),baseline:before))
        before=try store.applyApplicationSettings(store.previewApplicationSettings(.execution(try .init(maximumRunSeconds:90,cpuThreads:2,managedStorageLimitBytes:999999)),baseline:before))
        XCTAssertNil(before.preferences.execution.managedStorageLimitBytes,"Execution group must not modify storage")
        XCTAssertFalse(before.preferences.execution.sampleCapture.activationValues)
        before=try store.applyApplicationSettings(store.previewApplicationSettings(.storage(try .init(managedStorageLimitBytes:5000000,preparationCacheLimitBytes:100)),baseline:before))
        XCTAssertEqual(before.preferences.execution.cpuThreads,2)
        let reset=try store.applyApplicationSettings(store.previewApplicationSettings(.reset(.execution),baseline:before))
        XCTAssertEqual(reset.preferences.execution.cpuThreads,1)
        XCTAssertEqual(reset.preferences.execution.managedStorageLimitBytes,5000000)
        XCTAssertFalse(reset.preferences.execution.sampleCapture.activationValues)
        XCTAssertTrue(accepted.sampleCapture.activationValues)
        XCTAssertEqual(accepted.budget.maximumRunSeconds,3600)
    }
    func testPresentationAllowedWhileBusyAndInvalidFieldsNeverPersist() throws {
        let store=fixture(),before=try store.applicationSettingsSnapshot()
        store.isRunning=true
        let after=try store.applyApplicationSettings(store.previewApplicationSettings(.canvas(.init(zoomSensitivity:0.5,restoreAtLaunch:false)),baseline:before))
        XCTAssertEqual(after.preferences.effectiveCanvasZoomSensitivity,0.5)
        XCTAssertThrowsError(try store.previewApplicationSettings(.canvas(.init(zoomSensitivity:.nan,restoreAtLaunch:true)),baseline:after))
        XCTAssertThrowsError(try store.previewApplicationSettings(.appearance(.init(schemaVersion:9)),baseline:after))
        XCTAssertEqual(try store.applicationSettingsSnapshot().revision,after.revision)
        store.isRunning=false
    }
    func testCorruptAndFailedPersistenceDoNotPublishSavedState() throws {
        let store=fixture(),before=try store.applicationSettingsSnapshot()
        let proposal=try store.previewApplicationSettings(.appearance(.init(mode:.light)),baseline:before)
        try FileManager.default.createDirectory(at:store.appPreferenceStore.fileURL,withIntermediateDirectories:true)
        XCTAssertThrowsError(try store.applyApplicationSettings(proposal))
        XCTAssertEqual(store.appPreferences.appearance,.factory)
        XCTAssertThrowsError(try store.applicationSettingsSnapshot())
    }
    func testTwoWindowPropagationAndProjectDraftInvalidation() throws {
        _ = NSApplication.shared
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let source=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:source,storageOverride:root,detectCapabilities:false)})
        defer {for window in registry.openControllers {_=registry.close(window)};try? FileManager.default.removeItem(at:root)}
        let first=try registry.newFlow(present:false).session.workspace
        let second=try registry.newFlow(present:false).session.workspace
        let id=try XCTUnwrap(second.activeProjectID),draft=try second.projectSettingsSnapshot(projectID:id)
        let proposal=try second.previewProjectSettings(.execution(try .init(cpuThreads:2)),baseline:draft)
        var baseline=try first.applicationSettingsSnapshot()
        baseline=try first.applyApplicationSettings(first.previewApplicationSettings(.appearance(.init(mode:.light,palette:.teal)),baseline:baseline))
        XCTAssertEqual(second.appPreferences.appearance,baseline.preferences.appearance)
        XCTAssertEqual(second.appearanceState.preferences,baseline.preferences.appearance)
        baseline=try first.applyApplicationSettings(first.previewApplicationSettings(.canvas(.init(zoomSensitivity:0.75,restoreAtLaunch:false)),baseline:baseline))
        XCTAssertEqual(second.appearanceState.canvasZoomSensitivity,0.75)
        baseline=try first.applyApplicationSettings(first.previewApplicationSettings(.execution(try .init(cpuThreads:4)),baseline:baseline))
        XCTAssertEqual(second.appPreferences.execution.cpuThreads,4)
        XCTAssertThrowsError(try second.applyProjectSettings(proposal))
        XCTAssertEqual(second.workflowSession?.owner.document.revision,draft.documentRevision)
        let third=try registry.newFlow(present:false).session.workspace
        XCTAssertEqual(third.appPreferences,baseline.preferences)
    }
    func testFeedbackPersistsBeforeReconfigurationAndNeverRequestsOSPermission() throws {
        let store=fixture()
        defer {RunFeedbackController.shared.configure(.factory)}
        var baseline=try store.applicationSettingsSnapshot()
        baseline=try store.applyApplicationSettings(store.previewApplicationSettings(.feedback(.init(showStatusIcon:false,notifyCompleted:true)),baseline:baseline))
        XCTAssertTrue(baseline.preferences.effectiveRunFeedback.notifyCompleted)
        XCTAssertFalse(baseline.feedbackDeliveryStatus.contains("Requesting notification permission"))
        XCTAssertNotEqual(baseline.feedbackDeliveryStatus,"Notifications are off.")
        let proposal=try store.previewApplicationSettings(.reset(.feedback),baseline:baseline)
        try FileManager.default.setAttributes([.posixPermissions:0o500],ofItemAtPath:store.applicationDirectory.path)
        defer {try? FileManager.default.setAttributes([.posixPermissions:0o700],ofItemAtPath:store.applicationDirectory.path)}
        XCTAssertThrowsError(try store.applyApplicationSettings(proposal))
        XCTAssertTrue(store.appPreferences.effectiveRunFeedback.notifyCompleted)
        XCTAssertNotEqual(RunFeedbackController.shared.deliveryMessage,"Notifications are off.")
    }

}
