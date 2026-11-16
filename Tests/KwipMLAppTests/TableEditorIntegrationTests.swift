import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TableEditorIntegrationTests: XCTestCase {
    @MainActor func testSensorExampleSchemaEditingAndActualRowLossAreConsistent() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.loadExample()
        let model = workspace.graph
        let store = DataFlowWorkspaceStore(workspace: workspace)
        store.loadBundledSensorExample()
        let deadline = Date().addingTimeInterval(20)
        while store.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertEqual(store.project.graph.nodes.count, 10)
        XCTAssertTrue(store.fittedLessonText.lowercased().contains("filter"))
        XCTAssertEqual(store.availableColumnSchema(for: "range-filter").first { $0.name == "amount" }?.dtype, "float64")
        XCTAssertEqual(store.availableColumnSchema(for: "quality-filter").first { $0.name == "quality" }?.dtype, "string")
        store.selectNode("columns")
        let original=try XCTUnwrap(store.project.graph.nodes.first {$0.id=="columns"}).config
        guard case .array(var originalRows)=original["rows"],case .object(var reading)=originalRows.first,
              case .object(var temperature)=originalRows.dropFirst().first else {return XCTFail("Current Columns rules")}
        var collision=original
        reading["outputName"] = .string("temperature");originalRows[0] = .object(reading);collision["rows"] = .array(originalRows)
        XCTAssertFalse(store.applyTableConfiguration(collision,nodeID:"columns",expected:original))
        XCTAssertEqual(store.project.graph.nodes.first {$0.id=="columns"}?.config,original)
        var swap=original
        temperature["outputName"] = .string("reading_raw");originalRows[1] = .object(temperature);swap["rows"] = .array(originalRows)
        XCTAssertTrue(store.applyTableConfiguration(swap,nodeID:"columns",expected:original),store.statusMessage)
        XCTAssertFalse(store.applyTableConfiguration(original,nodeID:"columns",expected:original),"Stale expected configuration cannot overwrite the applied draft")
        store.undo();XCTAssertEqual(store.project.graph.nodes.first {$0.id=="columns"}?.config,original)
        store.selectNode("quality-filter")
        let quality=try XCTUnwrap(store.project.graph.nodes.first {$0.id=="quality-filter"}).config
        guard case .array(let conditions)=quality["conditions"],case .object(let condition)=conditions.first else {return XCTFail("Typed Filter conditions")}
        func withLiteral(_ value:JSONValue,comparison:String?=nil)->[String:JSONValue] {
            var next=quality,row=condition;row["value"]=value
            if let comparison {row["comparison"] = .string(comparison)}
            next["conditions"] = .array([.object(row)]);return next
        }
        XCTAssertFalse(store.applyTableConfiguration(withLiteral(.number(1)),nodeID:"quality-filter",expected:quality))
        XCTAssertFalse(store.applyTableConfiguration(withLiteral(.string(String(repeating:"x",count:2_000_001))),nodeID:"quality-filter",expected:quality))
        XCTAssertFalse(store.applyTableConfiguration(withLiteral(.string("null"),comparison:"isMissing"),nodeID:"quality-filter",expected:quality))
        XCTAssertNoThrow(try DataTableConfiguration.validate(withLiteral(.null,comparison:"isMissing"),operatorID:"kwip.data.filterRows",schema:store.availableColumnSchema(for:"quality-filter")),"Missing tests require a typed null, not the text null")
        XCTAssertEqual(store.project.graph.nodes.first {$0.id=="quality-filter"}?.config,quality)
        store.selectNode("range-filter")
        XCTAssertFalse(store.applyTableConfiguration(quality, nodeID: "quality-filter", expected: quality))
        store.runPreparation()
        while store.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertTrue(store.resultsCurrent, store.statusMessage)
        func rows(_ id:String,_ port:String="table")->Int? {((store.previews[id]?[port] as? [String:Any])?["preview"] as? [String:Any])?["rowCount"] as? Int}
        XCTAssertEqual(rows("source"),48);XCTAssertEqual(rows("quality-filter","matching"),40);XCTAssertEqual(rows("range-filter","matching"),36)
        // The 36 accepted rows contain 18 per label. Largest-remainder 60/20/20
        // allocates 11/4/3 within each stratum, including the validation/test tie.
        XCTAssertEqual(rows("training"),22);XCTAssertEqual(rows("validation"),8);XCTAssertEqual(rows("test"),6)
        XCTAssertEqual(workspace.graph, model); XCTAssertNil(workspace.execution.preparedTraining)
        store.save()
        let reopened = DataFlowWorkspaceStore(workspace: workspace)
        XCTAssertEqual(reopened.project, store.project)
        reopened.closeSourceInspection()
        while reopened.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
    }
}
