import AppKit
import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class DirectUpdateSafetyTests: XCTestCase {
    func testInactiveProjectActivityPreventsTerminationUntilEveryOperationFinishes() throws {
        _ = NSApplication.shared
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: {
            WorkspaceStore(rootOverride: root, storageOverride: storage, detectCapabilities: false)
        })
        let foreground = try registry.newFlow(present: false)
        let background = try registry.newFlow(present: false)
        registry.select(window: foreground.window)
        let work = background.session.workspace
        let activities: [ReferenceWritableKeyPath<WorkspaceStore, Bool>] = [
            \.isRunning, \.isPreviewing, \.isPreparingData, \.isInspectingData
        ]
        for activity in activities {
            work[keyPath: activity] = true
            XCTAssertTrue(registry.hasActiveOperations)
            XCTAssertFalse(registry.prepareToTerminate(), "An inactive project's worker must prevent update termination")
            XCTAssertTrue(background.session.workspace === work)
            work[keyPath: activity] = false
            XCTAssertFalse(registry.hasActiveOperations)
        }
        work.setLearningAssessmentActive(true)
        XCTAssertFalse(registry.prepareToTerminate())
        work.setLearningAssessmentActive(false)
        XCTAssertFalse(registry.hasActiveOperations)
        // Requesting cancellation is not the worker's completion acknowledgement.
        work.isRunning = true
        foreground.session.workspace.isPreviewing = true
        work.cancelRun()
        XCTAssertTrue(registry.hasActiveOperations)
        XCTAssertFalse(registry.prepareToTerminate())
        work.isRunning = false
        XCTAssertFalse(registry.prepareToTerminate(), "A different project is still previewing")
        foreground.session.workspace.isPreviewing = false
        XCTAssertTrue(registry.prepareToTerminate())
        for controller in registry.openControllers { XCTAssertTrue(registry.close(controller)) }
    }

    func testPendingInspectorEditsRefuseUpdateTerminationWithoutChangingSavedBytes() throws {
        _ = NSApplication.shared
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: {
            WorkspaceStore(rootOverride: root, storageOverride: storage, detectCapabilities: false)
        })
        let controller = try registry.newFlow(present: false)
        let owner = try XCTUnwrap(controller.session.workspace.workflowSession?.owner)
        // Use a real authored node so the pending edit belongs to the current owner.
        XCTAssertTrue(controller.session.workspace.loadSeparableConv2dWorkflowExample(device: "cpu"))
        let authored = try XCTUnwrap(registry.openControllers.first { $0 !== controller })
        let source = try XCTUnwrap(authored.session.workspace.workflowSession?.owner)
        let address = try XCTUnwrap(source.document.nodes.first?.address)
        let context = UUID()
        XCTAssertTrue(source.select(.init(nodeAddresses: [address]), lease: source.lease()))
        XCTAssertTrue(source.setInspectorEditPending(contextID: context, address: address, key: "pending-field", pending: true, lease: source.lease()))
        let url = URL(fileURLWithPath: try source.currentProjectRecord().path)
        let before = try Data(contentsOf: url)
        XCTAssertFalse(registry.prepareToTerminate())
        XCTAssertEqual(try Data(contentsOf: url), before)
        XCTAssertTrue(source.hasPendingInspectorEdits)
        XCTAssertTrue(controller.session.workspace.workflowSession?.owner === owner)
        source.clearDataEdits(contextID: context)
        XCTAssertTrue(registry.prepareToTerminate())
        for window in registry.openControllers { XCTAssertTrue(registry.close(window)) }
    }
}
