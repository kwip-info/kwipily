import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ColumnsOwnerTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func registry() throws -> DataFlowOperatorRegistry {
        try .load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
    }
    private func configuration() -> [String: JSONValue] {
        ["rows": .array([
            .object(["source": .string("amount"), "included": .bool(true), "outputName": .string("amount"), "targetType": .string("keep")]),
            .object(["source": .string("category"), "included": .bool(true), "outputName": .string("category"), "targetType": .string("keep")])
        ]), "castFailure": .string("error"), "trimText": .bool(false),
         "newColumns": .string("drop"), "missingPolicy": .string("preserve")]
    }

    func testGridApplyIsAtomicRejectsConflictsAndStaleExpectedAndUndoRestores() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory)
        let address = WorkflowNodeAddress(domain: .data, localID: "columns")
        let original = configuration()
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: address, name: "Choose columns", payload: .dataOperator(operatorID: "kwip.data.columns", config: original))
        ])
        _ = try library.saveSource(.workflow(document), id: document.projectID)
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID,
            modelRegistry: .builtIn(), dataRegistry: registry())
        XCTAssertTrue(owner.select(.init(nodeAddresses: [address]), lease: owner.lease()))
        let sources = ColumnsOwnerSources(owner: owner)
        let context = try WorkflowDataNodeContext(owner: owner, address: address, sources: sources)
        var draft = try ColumnsConfigurationDraft(configuration: original)
        draft.move("category", by: -1)
        draft.update("amount") { $0.outputName = "price"; $0.targetType = "float64" }
        let changed = try draft.validatedConfiguration(schema: context.upstreamSchema)
        let revision = owner.document.revision
        XCTAssertTrue(context.applyTableConfiguration(changed, expected: original), context.statusMessage)
        XCTAssertEqual(owner.document.revision, revision + 1, "The whole grid is one transaction")
        XCTAssertEqual(context.node?.config, changed)
        XCTAssertEqual(try library.loadSource(id: document.projectID), .workflow(owner.document))

        var conflicting = draft
        conflicting.update("category") { $0.outputName = "price" }
        let appliedRevision = owner.document.revision
        XCTAssertFalse(context.applyTableConfiguration(conflicting.configuration, expected: changed))
        XCTAssertEqual(context.node?.config, changed)
        XCTAssertEqual(owner.document.revision, appliedRevision)
        XCTAssertFalse(context.applyTableConfiguration(original, expected: original), "The old expected configuration cannot overwrite the applied grid")
        XCTAssertEqual(context.node?.config, changed)
        XCTAssertEqual(owner.document.revision, appliedRevision)

        XCTAssertTrue(owner.undo(lease: owner.lease()))
        XCTAssertFalse(context.isCurrent)
        let restored = try WorkflowDataNodeContext(owner: owner, address: address, sources: sources)
        XCTAssertEqual(restored.node?.config, original, "One Undo restores selection, order, names and type together")
        XCTAssertFalse(restored.applyTableConfiguration(changed, expected: changed))
        XCTAssertEqual(try library.loadSource(id: document.projectID), .workflow(owner.document))
    }

    func testSavedLegacySelectAndCastRoundTripRetainsExactGraphAndConfigurations() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory)
        func address(_ id: String) -> WorkflowNodeAddress { .init(domain: .data, localID: id) }
        let selection: [String: JSONValue] = ["columns": .array([.string("amount"), .string("category")]),
            "rename": .object(["amount": .string("price")])]
        let cast: [String: JSONValue] = ["columns": .object(["price": .string("float64")]), "missingPolicy": .string("preserve")]
        let document = WorkflowDocument(projectID: UUID().uuidString, name: "Saved legacy preparation", nodes: [
            .init(address: address("source"), name: "Old source", payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["separator": .string(",")])),
            .init(address: address("choose"), name: "Old selection", x: 240, y: 80, payload: .dataOperator(operatorID: "kwip.data.selectRename", config: selection)),
            .init(address: address("cast"), name: "Old conversion", x: 480, y: 80, payload: .dataOperator(operatorID: "kwip.data.cast", config: cast))
        ], edges: [
            .init(id: "source-select", from: .init(address: address("source"), port: "table"), to: .init(address: address("choose"), port: "table"), kind: .table),
            .init(id: "select-cast", from: .init(address: address("choose"), port: "table"), to: .init(address: address("cast"), port: "table"), kind: .table)
        ])
        let bytes = try WorkflowCodec.encode(document)
        XCTAssertEqual(try WorkflowCodec.decode(bytes), document)
        _ = try library.saveSource(.workflow(document), id: document.projectID)
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID,
            modelRegistry: .builtIn(), dataRegistry: registry())
        XCTAssertEqual(owner.document, document)
        XCTAssertEqual(try WorkflowCodec.encode(owner.document), bytes, "Opening with the new registry cannot rewrite old IDs or configuration")
        // Original source remains byte-exact; captured descriptors are for
        // historical shape validation only, never new execution authoring.
        var captured=try registry()
        let retired=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/retired-data-operators-v1.json"))
        XCTAssertTrue(Set(captured.operators.map(\.id)).isDisjoint(with:retired.operators.map(\.id)))
        captured.operators += retired.operators
        XCTAssertThrowsError(try WorkflowDataPreparationPlan.compile(owner.document, registry: captured))
        let plan = try WorkflowDataPreparationPlan.compileHistoricalEvidence(owner.document, registry: captured)
        XCTAssertTrue(plan.nodes.contains { $0.operatorID == "kwip.data.selectRename" })
        XCTAssertTrue(plan.nodes.contains { $0.operatorID == "kwip.data.cast" })
        XCTAssertFalse(plan.nodes.contains { $0.operatorID == "kwip.data.columns" })
    }

    func testColumnsInPreparedClassifierPreservesTrainingLineageCompilation() throws {
        var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        let castAddress = WorkflowNodeAddress(domain: .data, localID: "cast")
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address == castAddress })
        var config = configuration()
        config["rows"] = .array([("signal", "float64"), ("context", "float64"), ("label", "int64")].map { name, type in
            .object(["source": .string(name), "included": .bool(true), "outputName": .string(name), "targetType": .string(type)])
        })
        config["missingPolicy"] = .string("error")
        document.nodes[index].payload = .dataOperator(operatorID: "kwip.data.columns", config: config)
        let plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
            modelRegistry: .builtIn(), dataRegistry: registry(), contractVersion: 12)
        XCTAssertEqual(plan.featureColumns, ["signal", "context"])
        XCTAssertEqual(plan.targetColumn, "label")
        XCTAssertEqual(plan.training, .init(node: "fit", port: "training"))
        XCTAssertEqual(plan.validation, .init(node: "fit", port: "validation"))
        XCTAssertEqual(plan.dataflow.nodes.first { $0.id == "cast" }?.operatorID, "kwip.data.columns")
        let fit=try XCTUnwrap(plan.dataflow.nodes.first {$0.operatorID == "kwip.data.preprocess"})
        XCTAssertEqual(Set(plan.dataflow.edges.filter {$0.to.node == fit.id}.map(\.from)),Set(["training","validation","test"].map {DataFlowEndpoint(node:"split",port:$0)}))
        let options=try PreprocessConfiguration(values:fit.config)
        XCTAssertEqual(options.mode,"partitions")
        XCTAssertTrue(options.columns.allSatisfy {$0.scaling == "standard" && $0.missing == "error"})
        XCTAssertTrue(document.edges.contains {
            $0.from == .init(address:.init(domain:.data,localID:"fit"),port:"validation") &&
            $0.to == .init(address:.init(domain:.data,localID:"validation"),port:"table")
        })
        XCTAssertTrue(Set(plan.dataflow.nodes.map(\.id)).isDisjoint(with:["training","validation"]))
    }
}

@MainActor private final class ColumnsOwnerSources: WorkflowDataSources {
    let projectID: String
    let sessionID: UUID
    var isBusy = false
    var statusMessage = "Recorded source"
    var changes: AnyPublisher<Void, Never> { Empty(completeImmediately: false).eraseToAnyPublisher() }
    init(owner: WorkflowAuthoringStore) { projectID = owner.projectID; sessionID = owner.sessionID }
    func recordedSchema(for address: WorkflowNodeAddress, in document: WorkflowDocument) -> [DataColumnSchema] {
        [.init(name: "amount", dtype: "string"), .init(name: "category", dtype: "string")]
    }
    func bindingDisplayName(for address: WorkflowNodeAddress) -> String? { nil }
    func chooseSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool { false }
    func inspectSource(for address: WorkflowNodeAddress, lease: WorkflowEditorLease) -> Bool { false }
}
