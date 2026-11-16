import XCTest
@testable import KwipMLCore

final class WorkflowModuleCompositionTests: XCTestCase {
    private func dataRegistry() throws -> DataFlowOperatorRegistry {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try .load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
    }
    private func legacy() throws -> WorkflowDocument {
        var source = try WorkflowExamples.preparedParallelClassifier()
        source.schemaVersion = 10
        return source
    }
    private func selected() throws -> WorkflowDocument {
        var source = try legacy(); source.schemaVersion = 11
        source.definitions[0].composition = .init(kind: .graph, children:
            source.definitions[0].body.nodes.map { .init(localID: $0.id) })
        return source
    }
    func testExplicitMigrationPreservesWholeSourceAndLegacyBytes() throws {
        let old = try legacy(), before = try WorkflowCodec.encode(old)
        let migrated = try WorkflowCodec.migrateToVersion11(old)
        XCTAssertEqual(migrated.schemaVersion, 11)
        var restored = migrated; restored.schemaVersion = 10
        XCTAssertEqual(restored, old)
        XCTAssertEqual(try WorkflowCodec.encode(restored), before)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(migrated)), migrated)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion11(migrated), migrated)
    }
    func testSelectedPolicyRoundTripAndCompilerTenPreserveActualPlan() throws {
        let source = try selected()
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(source)), source)
        let plan = try WorkflowCompiler.compile(source, train: WorkflowExamples.trainAddress,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry(), contractVersion: 10)
        XCTAssertNotNil(plan.model.graph.compositionPlan)
        XCTAssertEqual(plan.model.graph.schemaVersion, 2)
        XCTAssertEqual(plan.model.provenance?.nativeExpansionContractVersion, 2)
        XCTAssertEqual(plan.model.provenance?.sourceCompositeDocument.schemaVersion, 3)
        XCTAssertEqual(plan.settings.epochs, 3)
        XCTAssertEqual(plan.featureColumns, ["signal", "context"])
    }
    func testOldWorkflowAndCompilerCannotSilentlyDropPolicy() throws {
        var old = try selected(); old.schemaVersion = 10
        XCTAssertThrowsError(try WorkflowCodec.encode(old))
        let selected = try selected()
        XCTAssertThrowsError(try WorkflowCompiler.compile(selected, train: WorkflowExamples.trainAddress,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry(), contractVersion: 9)) { error in
            XCTAssertEqual((error as? WorkflowCompilerError)?.code, .configuration)
        }
    }
    func testLegacyCompilerNineStillProducesAbsentPlanAndOriginalIdentity() throws {
        let source = try legacy()
        let before = try WorkflowCodec.encode(source)
        let plan = try WorkflowCompiler.compile(source, train: WorkflowExamples.trainAddress,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry(), contractVersion: 9)
        XCTAssertNil(plan.model.graph.compositionPlan)
        XCTAssertEqual(plan.model.provenance?.nativeExpansionContractVersion, 1)
        XCTAssertEqual(plan.model.provenance?.sourceCompositeDocument.schemaVersion, 2)
        XCTAssertEqual(try WorkflowCodec.encode(source), before)
    }
}
