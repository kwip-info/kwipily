import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class Data02SourceOwnerTests: XCTestCase {
    private func waitFor(_ predicate: () -> Bool) async throws {
        for _ in 0..<1000 {
            if predicate() { return }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("Managed source operation did not reach its expected state")
    }
    func testManagedPreviewPublishesAfterExitAndRetiresOnBindingOrSeparatorChange() async throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:directory) }
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let file=directory.appendingPathComponent("source.csv")
        try "a,b\n001,true\n".write(to:file,atomically:true,encoding:.utf8)
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:directory,detectCapabilities:false)
        let store=DataFlowWorkspaceStore(workspace:workspace)
        store.project.sourceBindings["source"]=file.path
        store.selectNode("source")
        var exit:((Int32,String,SourceInspectionResult?)->Void)?, hash=""
        store.sourceInspectionExecute={ _,_,_,_,_,captured,_,finish in
            hash=captured; exit=finish; return .init(cancel:{})
        }
        store.inspectSource(id:"source")
        try await waitFor { exit != nil }
        XCTAssertNil(store.sourceInspectionPreview(for:"source"))
        exit?(0,"",.init(columns:["a","b"],rows:[["001","true"]],rowCount:1,sourceSHA256:hash))
        try await waitFor { !store.isInspectingSource }
        let result=try XCTUnwrap(store.sourceInspectionPreview(for:"source"))
        XCTAssertEqual(result.rows,[["001","true"]]); XCTAssertEqual(result.sourceSHA256,hash)
        // Presentation reads use cached evidence; changing external bytes is discovered only by Refresh.
        try FileManager.default.removeItem(at:file)
        XCTAssertEqual(store.sourceInspectionPreview(for:"source"),result)
        let context=PreparationDataNodeContext(store:store,nodeID:"source")
        XCTAssertTrue(context.sourcePermissionSummary.contains("recorded snapshot"))
        let original=store.project.graph.nodes.first{$0.id == "source"}!.config
        let index=store.project.graph.nodes.firstIndex{$0.id == "source"}!
        store.project.graph.nodes[index].config["separator"] = .string(";")
        XCTAssertNil(store.sourceInspectionPreview(for:"source"))
        store.project.graph.nodes[index].config=original
        store.project.sourceBindings["source"] = directory.appendingPathComponent("different.csv").path
        XCTAssertNil(store.sourceInspectionPreview(for:"source"))
    }
}
