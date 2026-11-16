import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ProjectWindowRegistryTests: XCTestCase {
    func testTemplateCreatesSeparateProjectAndPersistsOriginWithoutBindings() throws {
        _=NSApplication.shared
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)})
        let existing=try registry.newFlow(present:false)
        let original=try XCTUnwrap(existing.session.workspace.workflowSession?.owner).document
        let template=try registry.newFlow(templateID:"sales-exceptions",present:false)
        let owner=try XCTUnwrap(template.session.workspace.workflowSession?.owner)
        XCTAssertEqual(existing.session.workspace.workflowSession?.owner.document,original)
        XCTAssertNotEqual(owner.projectID,original.projectID)
        XCTAssertEqual(owner.document.templateOrigin?.id,"sales-exceptions")
        XCTAssertEqual(owner.selection.nodeAddresses,[WorkflowNodeAddress(domain:.data,localID:"source")])
        XCTAssertTrue(registry.save(window:template.window))
        guard case .workflow(let saved)=try template.session.workspace.projectLibrary.loadSource(id:owner.projectID) else {return XCTFail("Expected workflow")}
        XCTAssertEqual(saved.templateOrigin,owner.document.templateOrigin)
        XCTAssertTrue(registry.close(template));XCTAssertTrue(registry.close(existing))
    }
    func testRootsKeepIndependentOwnersDedupeAndCloseOnlyOneSession() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { WorkspaceStore(rootOverride: root, storageOverride: directory, detectCapabilities: false) })
        let a = try registry.newFlow(present: false)
        let b = try registry.newFlow(present: false)
        XCTAssertFalse(a.session.workspace === b.session.workspace)
        let first = try XCTUnwrap(a.session.workspace.workflowSession?.owner)
        let second = try XCTUnwrap(b.session.workspace.workflowSession?.owner)
        XCTAssertNotEqual(first.projectID, second.projectID)
        XCTAssertTrue(first.rename("First changed", lease: first.lease()))
        a.session.workspace.generatedCode = "First output"
        XCTAssertTrue(try registry.open(try first.currentProjectRecord(), present: false) === a)
        XCTAssertEqual(a.session.workspace.generatedCode, "First output")
        XCTAssertEqual(first.document.name, "First changed")
        XCTAssertEqual(second.document.name, "Untitled flow")
        XCTAssertTrue(registry.save(window: a.window))
        XCTAssertFalse(registry.hasActiveOperations)
        b.session.workspace.isRunning = true
        XCTAssertTrue(registry.hasActiveOperations, "An inactive root must still prevent update relaunch")
        XCTAssertFalse(registry.close(b), "An active operation must not be abandoned by closing its tab")
        b.session.workspace.isRunning = false
        XCTAssertFalse(registry.hasActiveOperations)
        XCTAssertTrue(registry.close(b))
        XCTAssertTrue(a.session.workspace.workflowSession?.owner === first)
        XCTAssertTrue(first.canUndo)
        XCTAssertEqual(registry.projectCount, 1)
        XCTAssertTrue(registry.close(a))
    }
    func testNativeNewTabAndLaunchRoutingDoNotReplaceRootAndPreferencesStayLive() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        var allocations = 0
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { allocations += 1; return WorkspaceStore(rootOverride: root, storageOverride: directory, detectCapabilities: false) })
        let a = try registry.newFlow(present: false)
        let owner = try XCTUnwrap(a.session.workspace.workflowSession?.owner)
        let initialAllocations = allocations
        _ = try registry.open(try owner.currentProjectRecord(), present: false)
        XCTAssertEqual(allocations, initialAllocations, "Focusing an open root must not initialize another runtime workspace")
        XCTAssertNotNil(a.session.workspace.exampleInWindow)
        XCTAssertNotNil(a.session.tensor.startInWindow)
        XCTAssertNotNil(a.session.affine.startInWindow)
        XCTAssertNotNil(a.session.normalization.startInWindow)
        let b = try registry.newFlow(present: false)
        var defaults = a.session.workspace.appPreferences.canvasDefaults
        defaults.flowDirection = .topToBottom
        XCTAssertTrue(a.session.workspace.setCanvasDefaults(defaults))
        XCTAssertEqual(b.session.workspace.appPreferences.canvasDefaults, defaults)
        a.newWindowForTab(nil)
        XCTAssertEqual(registry.projectCount, 3, "Native + creates a persisted independent project")
        XCTAssertTrue(a.session.tensor.start(operation: "reshape"))
        XCTAssertEqual(registry.projectCount, 4)
        XCTAssertTrue(a.session.workspace.loadSeparableConv2dWorkflowExample(device: "cpu"))
        XCTAssertEqual(registry.projectCount, 5)
        XCTAssertEqual(try a.session.workspace.projectLibrary.listSources().count, 5, "Lesson start must not create an unrelated return project")
        XCTAssertTrue(a.session.workspace.workflowSession?.owner === owner)
        XCTAssertFalse(a.session.tensor.isActive)
        XCTAssertTrue(registry.openControllers.contains { $0.session.tensor.isActive })
        registry.select(window: a.window)
        XCTAssertTrue(registry.activeSession === a.session)
        let unrelated = NSWindow(contentRect: .zero, styleMask: [.titled], backing: .buffered, defer: false)
        unrelated.isReleasedWhenClosed = false
        // A closing child panel can resign after the document already regained key focus.
        NotificationCenter.default.post(name: NSWindow.didResignKeyNotification, object: unrelated)
        XCTAssertTrue(registry.activeSession === a.session, "A late child resign must not clear the current document’s commands")
        let dropdown = AttachedDropdownPanel(contentRect: .zero, styleMask: [.borderless], backing: .buffered, defer: false)
        dropdown.isReleasedWhenClosed = false
        a.window?.addChildWindow(dropdown, ordered: .above)
        registry.select(window: dropdown)
        XCTAssertTrue(registry.activeSession === a.session, "An attached choice panel belongs to its exact document, including during menu tracking")
        a.window?.removeChildWindow(dropdown); dropdown.close()
        a.window?.addChildWindow(unrelated, ordered: .above)
        registry.select(window: unrelated)
        XCTAssertNil(registry.activeSession, "Child/settings windows must not inherit last active root commands")
        XCTAssertFalse(registry.save(window: unrelated))
        a.window?.removeChildWindow(unrelated)
        var reopenedSection: String?
        registry.reopenLibrary = { reopenedSection = $0 }
        registry.showLibrary(section: "Examples")
        XCTAssertEqual(reopenedSection, "Examples", "Learn must reopen a closed library window")
        for controller in registry.openControllers { XCTAssertTrue(registry.close(controller)) }
        unrelated.close()
    }
    func testPreparingThenCancellingTerminationKeepsLessonOwnerAlive() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { WorkspaceStore(rootOverride: root, storageOverride: directory, detectCapabilities: false) })
        let busy = try registry.newFlow(present: false)
        XCTAssertTrue(busy.session.tensor.start(operation: "reshape"))
        let lesson = try XCTUnwrap(registry.openControllers.first { $0.session.tensor.isActive })
        let identity = lesson.session.workspace.activeProjectID
        XCTAssertTrue(lesson.session.prepareToClose(), "Preparation may persist but must not retire a live lesson")
        XCTAssertTrue(lesson.session.tensor.isActive)
        busy.session.workspace.isRunning = true
        XCTAssertFalse(registry.prepareToTerminate())
        XCTAssertTrue(lesson.session.tensor.isActive)
        XCTAssertEqual(lesson.session.workspace.activeProjectID, identity)
        XCTAssertNotNil(lesson.session.workspace.lessonContext)
        busy.session.workspace.isRunning = false
        for controller in registry.openControllers { XCTAssertTrue(registry.close(controller)) }
    }
    func testManagedMainOpensExactPinsWithoutImportOrLegacyFallback() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: { WorkspaceStore(rootOverride: root, storageOverride: directory, detectCapabilities: false) })
        let workspace = WorkspaceStore(rootOverride: root, storageOverride: directory, detectCapabilities: false)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try workspace.projectLibrary.saveSource(.workflow(source))
        let record = try workspace.projectLibrary.activateFileBackedWorkflow(id: source.projectID)
        let originalBytes=try Data(contentsOf:URL(fileURLWithPath:record.path))
        let window = try registry.openFile(URL(fileURLWithPath: record.path), present: false)
        var expected = try WorkflowModelBoundary.consolidate(RetiredToolAuthoringUpgrade.plan(source))
        expected.revision = source.revision + 1
        XCTAssertEqual(window.session.workspace.workflowSession?.owner.document, expected)
        let backup=URL(fileURLWithPath:record.path).appendingPathExtension("before-retired-tools-r\(source.revision)")
        XCTAssertEqual(try Data(contentsOf:backup),originalBytes,"Exact authoring upgrade preserves the original recoverable file and dependency pins")
        XCTAssertNotNil(window.session.workspace.workflowSession?.owner.pinnedFlowDocument)
        XCTAssertTrue(try registry.open(record, present: false) === window)
        XCTAssertEqual(try workspace.projectLibrary.listSources().count, 1)
        let owner = try XCTUnwrap(window.session.workspace.workflowSession?.owner)
        let address = try XCTUnwrap(owner.document.nodes.first?.address)
        let contextID = UUID()
        XCTAssertTrue(owner.select(.init(nodeAddresses: [address]), lease: owner.lease()))
        XCTAssertTrue(owner.setInspectorEditPending(contextID: contextID, address: address, key: "pending-field", pending: true, lease: owner.lease()))
        let bytes = try Data(contentsOf: URL(fileURLWithPath: record.path))
        XCTAssertFalse(window.session.save())
        XCTAssertFalse(window.session.workspace.saveManagedProject(), "Toolbar Save must share pending inspector protection")
        XCTAssertEqual(window.session.workspace.saveError, "Apply or revert the pending configuration edits before saving or closing this project.")
        XCTAssertFalse(registry.close(window))
        XCTAssertTrue(window.session.workspace.workflowSession?.owner === owner)
        XCTAssertTrue(owner.hasPendingInspectorEdits)
        XCTAssertEqual(try Data(contentsOf: URL(fileURLWithPath: record.path)), bytes)
        owner.clearDataEdits(contextID: contextID)
        XCTAssertTrue(window.session.workspace.saveManagedProject())
        XCTAssertNil(window.session.workspace.saveError, "A successful save clears the visible save-error notice")
        XCTAssertTrue(registry.close(window))
    }
}


extension ProjectWindowRegistryTests {
    func testWelcomeIsNotACanvasAndLegacyLibraryTabIsNotRestoredOrRecaptured() throws {
        _ = NSApplication.shared
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let factory = { WorkspaceStore(rootOverride: root, storageOverride: directory, detectCapabilities: false) }
        let registry = ProjectWindowRegistry(presentsWindows: false, makeWorkspace: factory)
        let session = ProjectWindowSession(workspace: factory())
        session.workspace.appPreferences.restoreCanvasSessionAtLaunch = false
        let welcome = NSWindow(contentRect: .init(x: 0, y: 0, width: 900, height: 560), styleMask: [.titled, .closable], backing: .buffered, defer: false)
        welcome.isReleasedWhenClosed = false
        defer { welcome.close(); for controller in registry.openControllers { _ = registry.close(controller) } }
        registry.registerLauncher(welcome, session: session)
        XCTAssertEqual(welcome.tabbingMode, .disallowed)
        let project = try registry.newFlow(present: false)
        let id = try XCTUnwrap(project.session.workspace.activeProjectID.flatMap(UUID.init(uuidString:)))
        let store = CanvasTabSessionStore(directory: session.workspace.applicationDirectory.appendingPathComponent("canvas-session"))
        try store.save(.init(groups: [.init(tabs: [.library, .managed(id)], selected: 1)], activeGroup: 0))
        let coordinator = registry.makeRestorationCoordinator()
        coordinator.retry()
        XCTAssertTrue(coordinator.failures.isEmpty)
        XCTAssertFalse(project.window?.tabbedWindows?.contains(where: { $0 === welcome }) == true)
        XCTAssertTrue(try coordinator.captureAfterPreparation { true })
        XCTAssertEqual(try store.load()?.groups.flatMap(\.tabs), [.managed(id)])
        XCTAssertTrue(registry.session(for: welcome) === session, "Welcome retains application routing when reopened")
    }
}
