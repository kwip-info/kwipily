import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RetiredToolAuthoringUpgradeTests: XCTestCase {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private func source(operatorID: String, config: [String: JSONValue]) throws -> WorkflowDocument {
        var document = try DataEngineeringPractice.document("data02-calculate")
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address.localID == "tool" })
        document.nodes[index].payload = .dataOperator(operatorID: operatorID, config: config)
        return document
    }
    @MainActor private func owner(_ document: WorkflowDocument, root: URL) throws -> (WorkflowAuthoringStore, URL, Data) {
        let library = ProjectLibrary(directory: root)
        let record = try library.saveSource(.workflow(document))
        let url = URL(fileURLWithPath: record.path)
        let models = try ComponentRegistry.builtIn()
        let data = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        return (try WorkflowAuthoringStore(library: library, projectID: document.projectID, modelRegistry: models, dataRegistry: data), url, try Data(contentsOf: url))
    }
    @MainActor func testExactUpgradePersistsOnceWithVerifiedOriginalAndNoUndoToRetiredTools() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try source(operatorID: "kwip.data.calculate", config: [
            "expression": .string("[quantity] * [unit_price]"), "outputColumn": .string("order_total"),
            "dtype": .string("float64"), "replaceExisting": .bool(false), "nullPolicy": .string("error")
        ])
        let (store, url, bytes) = try owner(original, root: root)
        XCTAssertTrue(try store.migrateRetiredToolsForAuthoring())
        XCTAssertEqual(store.document.revision, original.revision + 1)
        XCTAssertFalse(RetiredToolAuthoringUpgrade.requiresUpgrade(store.document))
        XCTAssertNil(store.retiredToolRepairMessage)
        XCTAssertFalse(store.canUndo)
        XCTAssertEqual(try Data(contentsOf: url.appendingPathExtension("before-retired-tools-r\(original.revision)")), bytes)
        let saved = try Data(contentsOf: url)
        XCTAssertFalse(try store.migrateRetiredToolsForAuthoring())
        XCTAssertEqual(try Data(contentsOf: url), saved)
    }
    @MainActor func testRunSetupUpgradeBacksUpOriginalAndPublishesCurrentTrainingTableInputs() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let original=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let (store,url,bytes)=try owner(original,root:root)
        XCTAssertTrue(try store.migrateRetiredToolsForAuthoring())
        XCTAssertEqual(store.document.schemaVersion,18)
        XCTAssertEqual(try Data(contentsOf:url.appendingPathExtension("before-retired-tools-r\(original.revision)")),bytes)
        XCTAssertFalse(store.document.nodes.contains(where:WorkflowTrainRunSetup.isLegacyNode))
        let train=try XCTUnwrap(store.document.nodes.first {$0.address==WorkflowExamples.trainAddress})
        let ports=try WorkflowPortContracts.resolve(train,document:store.document,modelRegistry:store.modelRegistry,dataRegistry:store.dataRegistry)
        XCTAssertEqual(ports.inputs.map(\.id),["model","training","validation"])
        let canvas=try WorkflowCanvasAdapter(owner:store,modelRegistry:store.modelRegistry,dataRegistry:store.dataRegistry)
        XCTAssertFalse(canvas.creationEntries.contains {$0.iconType=="workflow.runSetup"})
        XCTAssertEqual(try WorkflowTrainRunSetup.resolve(in:store.document,train:train.address).configuration,
                       try WorkflowTrainRunSetup.resolve(in:original,train:train.address).configuration)
    }
    @MainActor func testDifferentSplitAlgorithmLeavesOriginalEditableWithRepairNoticeAndNoWrite() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try source(operatorID: "kwip.data.split", config: ["validationFraction": .number(0.25), "seed": .integer(42)])
        let (store, url, bytes) = try owner(original, root: root)
        XCTAssertFalse(try store.migrateRetiredToolsForAuthoring())
        XCTAssertEqual(store.document, original)
        XCTAssertEqual(try Data(contentsOf: url), bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.appendingPathExtension("before-retired-tools-r\(original.revision)").path))
        XCTAssertNotNil(store.retiredToolRepairMessage)
        XCTAssertTrue(store.retiredToolRepairMessage?.contains("original project is unchanged") == true)
    }
    @MainActor func testRecoveryCopyCannotBeASymlinkToTheSourceBeingReplaced() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let original = try source(operatorID: "kwip.data.calculate", config: [
            "expression": .string("[quantity] * [unit_price]"), "outputColumn": .string("order_total"),
            "dtype": .string("float64"), "replaceExisting": .bool(false), "nullPolicy": .string("error")
        ])
        let (store, url, bytes) = try owner(original, root: root)
        let backup = url.appendingPathExtension("before-retired-tools-r\(original.revision)")
        try FileManager.default.createSymbolicLink(at: backup, withDestinationURL: url)
        XCTAssertThrowsError(try store.migrateRetiredToolsForAuthoring())
        XCTAssertEqual(try Data(contentsOf: url), bytes)
        XCTAssertEqual(store.document, original)
    }
}
