import XCTest
@testable import KwipMLCore

final class LossWorkflowExampleTests: XCTestCase {
    private let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    private func registry() throws -> ComponentRegistry {
        var registry = try ComponentRegistry.builtIn()
        let value = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: root.appendingPathComponent("components/losses/bundle.json")))
        guard case .object(let bundle) = value, case .array(let entries) = bundle["components"] else { throw GraphConfigurationError("Missing loss definitions") }
        for entry in entries {
            guard case .object(let fields) = entry, let definition = fields["definition"] else { throw GraphConfigurationError("Missing definition") }
            let decoded = try JSONDecoder().decode(ComponentDefinition.self, from: JSONEncoder().encode(definition))
            if !registry.components.contains(where: { $0.id == decoded.id }) { registry.components.append(decoded) }
        }
        return registry
    }
    func testEveryExampleCompilesItsRealTypedObjectiveAndRoundTrips() throws {
        for task in LossLearningTask.allCases {
            do {
                let doc = try WorkflowExamples.preparedLossObjective(task: task, device: "cpu")
                let plan = try WorkflowCompiler.compile(doc, train: WorkflowExamples.trainAddress, modelRegistry: registry(), dataRegistry: DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json")), contractVersion: 13)
                XCTAssertEqual(plan.lossRecipe?.configuration.componentID, task.componentID)
                XCTAssertEqual(plan.lossRecipe?.configuration, try LossConfiguration(componentID: task.componentID, values: task.configuration))
                XCTAssertNil(plan.targetColumn)
                XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(doc)), doc)
                XCTAssertEqual(doc.schemaVersion, 14)
                if let directory = ProcessInfo.processInfo.environment["KWIPML_LOSS_EXAMPLE_SNAPSHOTS"] {
                    let folder = URL(fileURLWithPath: directory).appendingPathComponent(task.componentID)
                    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                    let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
                    try encoder.encode(plan.model.graph).write(to: folder.appendingPathComponent("model.json"))
                    try DataFlowCodec.encodeGraph(plan.dataflow).write(to: folder.appendingPathComponent("dataflow.json"))
                    let recipe = try XCTUnwrap(plan.lossRecipe)
                    try encoder.encode(recipe.configuration.recipeValue).write(to: folder.appendingPathComponent("recipe.json"))
                    try encoder.encode(recipe.targetMapping).write(to: folder.appendingPathComponent("mapping.json"))
                }
                XCTAssertTrue(doc.nodes.contains { $0.address == WorkflowExamples.sourceAddress })
                XCTAssertFalse(doc.edges.contains { $0.kind == .objectiveRecipe })
                let train=try XCTUnwrap(doc.nodes.first {$0.address == WorkflowExamples.trainAddress})
                guard case .train(let settings)=train.payload else {return XCTFail("Missing Train")}
                XCTAssertEqual(settings["objective"],try LossConfiguration(componentID:task.componentID,values:task.configuration).recipeValue)
                XCTAssertEqual(plan.lossRecipe?.targetMapping.inputColumns.count, task == .triplet ? 6 : 2)
            } catch { XCTFail("\(task.rawValue): \(error)") }
        }
    }
    func testTripletUsesOneSharedEmbeddingRatherThanMixingExamples() throws {
        let doc = try WorkflowExamples.preparedLossObjective(task: .triplet, device: "cpu")
        let shape = try XCTUnwrap(doc.nodes.first { $0.address.localID == "shape" })
        guard case .modelPrimitive(let kind, _, let config) = shape.payload else { return XCTFail("Missing reshape") }
        XCTAssertEqual(kind, "reshape"); XCTAssertEqual(config["shape"], .array([3,2]))
        let head = try XCTUnwrap(doc.nodes.first { $0.address.localID == "head" })
        guard case .modelPrimitive(_, _, let headConfig) = head.payload else { return XCTFail("Missing shared linear") }
        XCTAssertEqual(headConfig["in_features"], 2); XCTAssertEqual(headConfig["out_features"], 2)
        XCTAssertTrue(doc.edges.contains { $0.from.address == shape.address && $0.to.address == head.address })
    }
    func testBadIdentityAndDeviceAreRefused() {
        XCTAssertThrowsError(try WorkflowExamples.preparedLossObjective(task: .mse, projectID: "bad"))
        XCTAssertThrowsError(try WorkflowExamples.preparedLossObjective(task: .mse, device: "cuda"))
    }
}
