import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class OptimizerWindowDiscoveryTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func event(_ store: WorkspaceStore, descriptor: JSONValue?) throws -> String {
        // Synthetic discovery fixture identity; not a claim about the unpinned development runtime.
        let version = store.runtime?.packageVersions["torch"] ?? "2.14.1"
        let contract = try BuiltInCapabilityBridge.executionContract(registry: store.registry, runtimeVersion: version, availableDevices: [.cpu])
        var raw: [String: Any] = ["event":"capabilities", "torchVersion":version, "devices":["cpu"], "defaultDevice":"cpu", "executionContract":try JSONSerialization.jsonObject(with: JSONEncoder().encode(contract))]
        if let descriptor { raw["componentCapabilities"] = ["optimizer_adam":try JSONSerialization.jsonObject(with: JSONEncoder().encode(descriptor))] }
        return String(decoding: try JSONSerialization.data(withJSONObject: raw), as: UTF8.self)
    }
    func testHistoricalLessonDoesNotBorrowParentCapabilitiesOrLaunchARefresh() throws {
        _ = NSApplication.shared
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let registry = ProjectWindowRegistry(presentsWindows:false, makeWorkspace:{ WorkspaceStore(rootOverride:self.repository, storageOverride:storage, detectCapabilities:false) })
        let parent = try registry.newFlow(present:false)
        let descriptor = try XCTUnwrap(BundledOptimizerComponents.entries["optimizer_adam"]?.runtimeContract)
        parent.session.workspace.optimizerRuntimeCapabilities = .init(descriptors:["optimizer_adam":try JSONEncoder().encode(descriptor)])
        parent.session.workspace.capabilityStatus = "Parent runtime verified"
        XCTAssertTrue(parent.session.tensor.start(operation:OptimizerLearningTask.adam.rawValue))
        let child = try XCTUnwrap(registry.openControllers.first { $0.session.tensor.isActive })
        XCTAssertFalse(child.session.workspace === parent.session.workspace)
        XCTAssertFalse(child.session.workspace.optimizerRuntimeCapabilities.contains("optimizer_adam"))
        var launches=0
        child.session.workspace.capabilityLaunch = { _,_,_,_,_,_ in launches+=1;return .init(cancel:{}) }
        XCTAssertFalse(child.session.tensor.canRefreshLessonRuntime)
        XCTAssertFalse(child.session.tensor.refreshLessonRuntime())
        let reason=try XCTUnwrap(child.session.tensor.runtimeReadinessError)
        XCTAssertTrue(reason.contains("read-only"),reason)
        XCTAssertEqual(launches,0)
        XCTAssertTrue(parent.session.workspace.optimizerRuntimeCapabilities.contains("optimizer_adam"))
        for controller in registry.openControllers { XCTAssertTrue(registry.close(controller)) }
    }
    func testCurrentRuntimeDiscoveryKeepsDriftAndStaleExitGuardsWithoutRevivingHistoricalLesson() throws {
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let store = WorkspaceStore(rootOverride:repository, storageOverride:storage, detectCapabilities:false)
        let lesson = TensorLearningStore(workspace:store)
        XCTAssertTrue(lesson.start(operation:OptimizerLearningTask.adam.rawValue))
        let descriptor = try XCTUnwrap(BundledOptimizerComponents.entries["optimizer_adam"]?.runtimeContract)
        var lines:[(String)->Void]=[], exits:[(Int32,String)->Void]=[]
        store.capabilityLaunch = { _,_,_,_,line,exit in lines.append(line); exits.append(exit); return .init(cancel:{}) }
        store.refreshCapabilities()
        let valid = try event(store,descriptor:descriptor)
        lines[0](valid)
        XCTAssertFalse(store.optimizerRuntimeCapabilities.contains("optimizer_adam"))
        XCTAssertNotNil(lesson.runtimeReadinessError,"Payload alone is not a completed verified probe")
        store.refreshCapabilities()
        exits[0](0,"")
        XCTAssertFalse(store.optimizerRuntimeCapabilities.contains("optimizer_adam"))
        XCTAssertNotNil(lesson.runtimeReadinessError,"Cancelled stale exact result must not grant support")
        lines[1](try event(store,descriptor:nil)); exits[1](0,"")
        XCTAssertNotNil(lesson.runtimeReadinessError)
        XCTAssertTrue(try XCTUnwrap(lesson.runtimeReadinessError).contains("read-only"))
        XCTAssertFalse(store.optimizerRuntimeCapabilities.contains("optimizer_adam"))
        store.refreshCapabilities()
        lines[2](try event(store,descriptor:.object(["schemaVersion":.integer(999)]))); exits[2](0,"")
        XCTAssertFalse(store.optimizerRuntimeCapabilities.contains("optimizer_adam"))
        XCTAssertNotNil(lesson.runtimeReadinessError)
        store.refreshCapabilities()
        lines[3](valid); XCTAssertNotNil(lesson.runtimeReadinessError)
        exits[3](0,""); XCTAssertTrue(store.optimizerRuntimeCapabilities.contains("optimizer_adam"))
        XCTAssertTrue(lesson.runtimeReadinessError?.contains("read-only") == true)
        XCTAssertFalse(lesson.canSubmit);XCTAssertFalse(lesson.canRefreshLessonRuntime)
        store.isRunning=true
        let count=lines.count
        XCTAssertFalse(lesson.canRefreshLessonRuntime);XCTAssertFalse(lesson.refreshLessonRuntime());XCTAssertEqual(lines.count,count)
        store.isRunning=false
        XCTAssertTrue(lesson.start(operation:"reshape"))
        XCTAssertFalse(lesson.canRefreshLessonRuntime);XCTAssertFalse(lesson.refreshLessonRuntime());XCTAssertEqual(lines.count,count)
    }
}
