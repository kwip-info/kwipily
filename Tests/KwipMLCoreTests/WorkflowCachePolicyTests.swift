import XCTest
@testable import KwipMLCore

final class WorkflowCachePolicyTests: XCTestCase {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func sample() throws -> WorkflowDocument { try WorkflowExamples.preparedParallelClassifier(device: "cpu") }
    private func compile(_ document: WorkflowDocument) throws -> WorkflowExecutionPlan {
        try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress, modelRegistry: .builtIn(),
            dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func set(_ value: JSONValue?, in document: inout WorkflowDocument, key: String = "preparationCachePolicy") throws {
        let index = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
        guard case .runSetup(var config) = document.nodes[index].payload else { return XCTFail("Missing setup") }
        config[key] = value; document.nodes[index].payload = .runSetup(config: config)
    }
    func testLegacyRemainsOffAndRejectsNewKeyEvenExplicitOff() throws {
        var old = try sample(); old.schemaVersion = 1
        XCTAssertEqual(try compile(old).preparationCachePolicy, .off)
        let bytes = try WorkflowCodec.encode(old)
        XCTAssertEqual(try WorkflowCodec.decode(bytes), old)
        try set(.string("off"), in: &old)
        XCTAssertThrowsError(try WorkflowCodec.encode(old))
        XCTAssertThrowsError(try compile(old))
    }
    func testVersionTwoPoliciesRoundTripWithoutChangingExecutableGraphsOrSettings() throws {
        let legacy = try sample(), baseline = try compile(legacy)
        for policy in WorkflowPreparationCachePolicy.allCases {
            var current = try WorkflowCodec.migrateToVersion2(legacy)
            try set(.string(policy.rawValue), in: &current)
            XCTAssertEqual(current.schemaVersion, 2)
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(current)), current)
            let plan = try compile(current)
            XCTAssertEqual(plan.preparationCachePolicy, policy)
            XCTAssertEqual(plan.dataflow, baseline.dataflow)
            XCTAssertEqual(plan.settings, baseline.settings)
            XCTAssertEqual(plan.model.graph, baseline.model.graph)
            XCTAssertEqual(plan.model.provenance, baseline.model.provenance)
            XCTAssertEqual(plan.model.semanticSHA256, baseline.model.semanticSHA256)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            XCTAssertEqual(try encoder.encode(plan.model.graph), try encoder.encode(baseline.model.graph), "Export input identity must not absorb preparation policy")
        }
    }
    func testVersionTwoMissingMeansOffButMalformedValuesNeverCoerce() throws {
        var current = try WorkflowCodec.migrateToVersion2(sample())
        try set(nil, in: &current)
        XCTAssertEqual(try compile(current).preparationCachePolicy, .off)
        for value: JSONValue in [.null, .bool(true), .integer(1), .string("automatic"), .object([:]), .array([])] {
            var invalid = current; try set(value, in: &invalid)
            XCTAssertThrowsError(try WorkflowCodec.encode(invalid), "Invalid policy \(value)")
            XCTAssertThrowsError(try compile(invalid), "Invalid policy \(value)")
        }
    }
    func testExplicitMigrationPreservesSourceAndIsIdempotentWithoutRevisionMutation() throws {
        var old = try sample(); old.revision = 17
        let migrated = try WorkflowCodec.migrateToVersion2(old)
        var expected = old; expected.schemaVersion = 2; try set(.string("off"), in: &expected)
        XCTAssertEqual(migrated, expected)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion2(migrated), migrated)
        XCTAssertEqual(old.schemaVersion, 1); XCTAssertEqual(migrated.revision, 17)
        var reuse = migrated; try set(.string("reuse"), in: &reuse)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion2(reuse), reuse)
        var unknown = old; unknown.schemaVersion = 999
        XCTAssertThrowsError(try WorkflowCodec.migrateToVersion2(unknown))
    }
    func testPortableConfigurationRejectsPrivateReferenceAndPolicyPathKeys() throws {
        let current = try WorkflowCodec.migrateToVersion2(sample())
        for key in ["reuseManifest", "reuseManifestSHA256", "cachePath", "sourceBindings"] {
            var invalid = current; try set(.string("/private/cache/manifest.json"), in: &invalid, key: key)
            XCTAssertThrowsError(try WorkflowCodec.encode(invalid), key)
        }
        let text = String(decoding: try WorkflowCodec.encode(current), as: UTF8.self)
        XCTAssertFalse(text.contains("/private/")); XCTAssertFalse(text.contains("reuseManifest"))
    }
}
