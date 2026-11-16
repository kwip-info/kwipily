import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SourceSeparatorTests:XCTestCase {
    @MainActor private func waitFor(_ predicate: () -> Bool) async throws {
        for _ in 0..<1000 {
            if predicate() { return }; try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("Inspection did not reach expected state")
    }
    @MainActor func testSeparatorInspectionUsesConfiguredValueAndDropsOldColumns() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let file=root.appendingPathComponent("source.csv");try "a;b\n1;2\n".write(to:file,atomically:true,encoding:.utf8)
        let store=DataFlowWorkspaceStore(workspace:WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false))
        store.selectNode("source");store.project.sourceBindings["source"]=file.path
        let index=try XCTUnwrap(store.project.graph.nodes.firstIndex{$0.id=="source"})
        store.project.graph.nodes[index].config["separator"] = .string(";")
        var separators: [String] = [], hash = ""
        var finish: ((Int32, String, SourceInspectionResult?) -> Void)?
        store.sourceInspectionExecute = { _, _, _, _, _, captured, separator, exit in
            separators.append(separator); hash = captured; finish = exit; return .init(cancel: {})
        }
        store.inspectSource(id: "source")
        try await waitFor { separators.count == 1 }
        XCTAssertEqual(separators[0], ";")
        finish?(0, "", .init(columns: ["a", "b"], rows: [["1", "2"]], rowCount: 1, sourceSHA256: hash))
        try await waitFor { !store.isInspectingSource }
        XCTAssertEqual(store.sourceColumns, ["a", "b"])
        store.configureSourceSeparator("\t", nodeID: "source")
        XCTAssertTrue(store.sourceColumns.isEmpty)
        try await waitFor { separators.count == 2 }
        XCTAssertEqual(separators[1], "\t")
        finish?(130, "", nil)
        try await waitFor { !store.isInspectingSource }
    }
    @MainActor func testCancelledInspectionHasTerminalStatusWithoutOverwritingReplacement() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let file=root.appendingPathComponent("source.csv");try "a,b\n1,2\n".write(to:file,atomically:true,encoding:.utf8)
        let store=DataFlowWorkspaceStore(workspace:WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false))
        store.project.sourceBindings["source"]=file.path
        var exits: [(Int32, String, SourceInspectionResult?) -> Void] = []
        store.sourceInspectionExecute = { _, _, _, _, _, _, _, exit in exits.append(exit); return .init(cancel: {}) }
        store.inspectSource(id: "source")
        try await waitFor { exits.count == 1 }
        store.cancelSourceInspection(); exits[0](130, "", nil)
        try await waitFor { !store.isInspectingSource }
        XCTAssertEqual(store.statusMessage, "Source inspection cancelled.")
        store.inspectSource(id: "source")
        try await waitFor { exits.count == 2 }
        store.inspectSource(id: "source"); exits[1](130, "", nil)
        try await waitFor { exits.count == 3 }
        XCTAssertEqual(store.statusMessage, "Inspecting source columns…"); XCTAssertTrue(store.isInspectingSource)
        store.closeSourceInspection(); store.statusMessage = "New context"; exits[2](130, "", nil)
        try await waitFor { !store.isInspectingSource }
        XCTAssertEqual(store.statusMessage, "New context")
    }
}
