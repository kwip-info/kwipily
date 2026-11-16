import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TrainDataAccessMigrationTests: XCTestCase {
    @MainActor func testOpeningMigrationPersistsOnceRetainsRecoveryAndRemovesRecipeCreationAndPorts() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let library = ProjectLibrary(directory: root)
        var source = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        source.schemaVersion = 13
        let loader = WorkflowNodeAddress(domain: .control, localID: "legacy-loader")
        source.nodes.append(.init(address: loader, payload: .batchRecipe(type: "data_loader", configurationVersion: 1, config: DataLoaderConfiguration.defaults)))
        source.edges.append(.init(from: .init(address: loader, port: "out"), to: .init(address: WorkflowExamples.trainAddress, port: "loader"), kind: .batchRecipe))
        let record = try library.saveSource(.workflow(source))
        let url = URL(fileURLWithPath: record.path), bytes = try Data(contentsOf: URL(fileURLWithPath: record.path))
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let models = try ComponentRegistry.builtIn(), operators = try DataFlowOperatorRegistry.load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
        let owner = try WorkflowAuthoringStore(library: library, projectID: source.projectID, modelRegistry: models, dataRegistry: operators)
        XCTAssertTrue(try owner.migrateDataLoadingForAuthoring())
        XCTAssertEqual(owner.document.revision, source.revision + 1)
        XCTAssertFalse(owner.canUndo)
        XCTAssertEqual(try Data(contentsOf: url.appendingPathExtension("before-retired-tools-r\(source.revision)")), bytes)
        XCTAssertEqual(try library.loadSource(id: source.projectID), .workflow(owner.document))
        XCTAssertFalse(try owner.migrateDataLoadingForAuthoring())
        let canvas = try WorkflowCanvasAdapter(owner: owner, modelRegistry: models, dataRegistry: operators)
        XCTAssertFalse(canvas.creationEntries.contains { ["dataset_source", "data_loader", "kwip.data.filter", "kwip.data.filterExpression", "kwip.data.regexMatch", "kwip.data.cast", "kwip.data.selectRename", "optimizer_sgd", "scheduler_step_lr", "loss_mse", "initialization", "gradient_policy", "trainability"].contains($0.iconType) })
        let inputs = canvas.projection.ports.filter { $0.endpoint.address == WorkflowExamples.trainAddress && $0.direction == .input }.map { $0.endpoint.port }
        XCTAssertEqual(inputs, ["model", "training", "validation"])
        let trainID = try XCTUnwrap(canvas.projection.displayIDByAddress[WorkflowExamples.trainAddress])
        XCTAssertNil(canvas.projection.portDescriptor(address: WorkflowExamples.trainAddress, port: "loader", direction: .input))
        XCTAssertNil(canvas.projection.portDescriptor(address: WorkflowExamples.trainAddress, port: "dataset", direction: .input))
        XCTAssertNotNil(canvas.graph.nodes.first { $0.id == trainID })
    }
}
