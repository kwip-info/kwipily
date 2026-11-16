import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FittedDataWorkspaceTests: XCTestCase {
    @MainActor func testCurrentToolsAreAvailableToSharedCanvasAndRetiredToolsStayUnavailable() throws {
        XCTAssertTrue(DataFlowWorkspaceStore.editableTypes.contains("kwip.data.preprocess"))
        let expected:Set<String>=["sourceCSV","sourceFiles","columns","filterRows","sampleRows","join","aggregate","view","formula","partition","preprocess","outputDelimited","textEncode","sequenceWindows","union","sort","unique","reshapeTable","splitText","prepareMedia","extractText","transcribeAudio","classifyAudio","gameRules","decide"]
        XCTAssertEqual(DataFlowWorkspaceStore.editableTypes,Set(expected.map {"kwip.data."+$0}))
        XCTAssertTrue(DataFlowWorkspaceStore.editableTypes.isDisjoint(with:RetiredDataOperatorMigration.retired))
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        XCTAssertEqual(DataFlowWorkspaceStore.editableTypes,Set(registry.operators.filter {$0.id != DataFlowSourceOperators.capturedTable}.map(\.id)))
    }
    @MainActor func testBundledFittedExampleExecutesAndRetainsMappingsExplicitly() async throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        workspace.loadExample()
        let model = workspace.graph
        let preparation = DataFlowWorkspaceStore(workspace: workspace)
        preparation.loadBundledFittedExample()
        XCTAssertEqual(preparation.project.graph.nodes.count, 7)
        XCTAssertEqual(workspace.graph, model)
        XCTAssertNil(workspace.execution.preparedTraining)
        XCTAssertTrue(preparation.fittedLessonText.contains("training"))
        preparation.runPreparation()
        let deadline = Date().addingTimeInterval(30)
        while preparation.isBusy && Date() < deadline { try await Task.sleep(nanoseconds: 25_000_000) }
        XCTAssertTrue(preparation.resultsCurrent, preparation.statusMessage)
        XCTAssertEqual(preparation.previews.count, 7)
        let training = try XCTUnwrap(preparation.previews["training"]?["table"] as? [String: Any])
        let validation = try XCTUnwrap(preparation.previews["validation"]?["table"] as? [String: Any])
        XCTAssertEqual((training["preview"] as? [String: Any])?["rowCount"] as? Int, 24)
        XCTAssertEqual((validation["preview"] as? [String: Any])?["rowCount"] as? Int, 8)
        let protected=try XCTUnwrap(preparation.previews["test"]?["table"] as? [String:Any])
        XCTAssertEqual((protected["preview"] as? [String:Any])?["rowCount"] as? Int,8)
        XCTAssertEqual(workspace.graph,model,"Preparing data does not edit its owning model canvas.")
        XCTAssertNil(workspace.execution.preparedTraining,"Preparation never silently selects a model mapping.")
        XCTAssertTrue(preparation.project.graph.nodes.allSatisfy {!RetiredDataOperatorMigration.retired.contains($0.operatorID)})
        preparation.save()
        let restored = DataFlowWorkspaceStore(workspace: workspace)
        XCTAssertEqual(restored.project, preparation.project)
    }

    func testCurrentPreprocessPoliciesPreserveExactColumnNamesAndTypedFallbacks() throws {
        var row=PreprocessConfiguration.Column(column:" x, coordinate ")
        row.missing="mostFrequent";row.tiePolicy="smallest";row.emptyPolicy="constant";row.emptyValue = .integer(0)
        let original=PreprocessConfiguration(columns:[row])
        let decoded=try PreprocessConfiguration(values:original.values)
        XCTAssertEqual(decoded,original)
        XCTAssertEqual(decoded.columns[0].column," x, coordinate ")
        XCTAssertEqual(decoded.columns[0].tiePolicy,"smallest")
        XCTAssertEqual(decoded.columns[0].emptyValue,.integer(0))
        XCTAssertThrowsError(try PreprocessConfiguration(mode:"table",columns:[row]).validate())
        row.missing="constant";row.value = .number(7)
        XCTAssertNoThrow(try PreprocessConfiguration(mode:"table",columns:[row]).validate())
    }
}
