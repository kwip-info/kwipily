import XCTest
@testable import KwipMLCore

final class RecurrentCanonicalExampleTests: XCTestCase {
    func testSixGenuinePreparedFactoriesCompileAndEmitCanonicalFixtures() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data = try DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        let factories: [(String, () throws -> WorkflowDocument)] = [
            ("rnn", { try WorkflowExamples.preparedRNNClassifier(device: "cpu") }),
            ("lstm", { try WorkflowExamples.preparedLSTMClassifier(device: "cpu") }),
            ("gru", { try WorkflowExamples.preparedGRUClassifier(device: "cpu") }),
            ("rnn_cell", { try WorkflowExamples.preparedRNNCellClassifier(device: "cpu") }),
            ("lstm_cell", { try WorkflowExamples.preparedLSTMCellClassifier(device: "cpu") }),
            ("gru_cell", { try WorkflowExamples.preparedGRUCellClassifier(device: "cpu") })
        ]
        XCTAssertEqual(Set(factories.map(\.0)), Set(RecurrentConfiguration.componentIDs))
        for (id, factory) in factories {
            let document = try factory()
            let train = try XCTUnwrap(document.nodes.first { if case .train = $0.payload { return true }; return false })
            let plan = try WorkflowCompiler.compile(document, train: train.address, modelRegistry: .builtIn(), dataRegistry: data, contractVersion: 6)
            let memory = try XCTUnwrap(plan.model.graph.nodes.first { $0.type == id })
            let configuration = try RecurrentConfiguration(componentID: id, values: memory.config)
            XCTAssertEqual(configuration.values["state_mode"], .string("zero"))
            XCTAssertTrue(configuration.supports(device: "cpu"))
            XCTAssertTrue(configuration.supports(device: "mps"))
            XCTAssertNil(plan.optimizerRecipe)
            XCTAssertNil(plan.schedulerRecipe)
            XCTAssertEqual(plan.model.graph.nodes.filter { RecurrentConfiguration.componentIDs.contains($0.type) }.count, 1)
            XCTAssertEqual(plan.settings.epochs, 3)
            XCTAssertEqual(plan.featureColumns.count, 2)
            if let path = ProcessInfo.processInfo.environment["KWIPML_RECURRENT_CANONICAL_FIXTURES"] {
                let directory = URL(fileURLWithPath: path).appendingPathComponent(id)
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys, .prettyPrinted]
                try encoder.encode(document).write(to: directory.appendingPathComponent("workflow.json"))
                try encoder.encode(plan.model.graph).write(to: directory.appendingPathComponent("model.json"))
                try encoder.encode(plan.dataflow).write(to: directory.appendingPathComponent("dataflow.json"))
                try encoder.encode(plan.settings).write(to: directory.appendingPathComponent("settings.json"))
            }
        }
    }
}
