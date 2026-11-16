import XCTest
@testable import KwipMLCore

final class ModuleCompositionAuthoringTests: XCTestCase {
    private func model() -> ModelAuthoringDocument {
        var source = CompositeExamples.parallelClassifier
        let first = source.root.instances[0]
        source.root.instances.append(.init(id: "private-twin", definition: first.definition, name: "Other private instance", x: 900, y: 500))
        return .init(projectID: "11111111-1111-1111-1111-111111111111", revision: 4, source: source)
    }
    private func selected(_ definition: CompositeDefinition) -> CompositeDefinition {
        var copy = definition
        copy.composition = .init(kind: .graph, children: (definition.body.nodes.map(\.id) + definition.body.instances.map(\.id)).map { .init(localID: $0) })
        return copy
    }
    func testSelectedDraftAtomicallyMigratesEnvelopeAndApplyCreatesOnlyPrivatePin() throws {
        let original = model(), path = ["features"]
        let begun = try ModelBlockEditing.beginDraft(original, path: path)
        let proposed = selected(begun.drafts[0].proposed)
        let updated = try ModelBlockEditing.updateDraft(begun, proposed: proposed, expectedBaseRevision: 4, path: path)
        XCTAssertEqual(updated.schemaVersion, 2); XCTAssertEqual(updated.source.schemaVersion, 3)
        XCTAssertEqual(updated.projectID, original.projectID); XCTAssertEqual(updated.revision, 4)
        XCTAssertEqual(updated.source.root, original.source.root); XCTAssertEqual(updated.source.definitions, original.source.definitions)
        XCTAssertEqual(updated.drafts[0].proposed, proposed)
        let applied = try ModelBlockEditing.applyDraft(updated, path: path, expectedBaseRevision: 4, idProvider: { "private-composition" })
        XCTAssertEqual(applied.schemaVersion, 2); XCTAssertEqual(applied.source.schemaVersion, 3); XCTAssertEqual(applied.revision, 5)
        XCTAssertTrue(applied.drafts.isEmpty)
        XCTAssertEqual(applied.source.root.instances.first { $0.id == "features" }?.definition.id, "private-composition")
        XCTAssertEqual(applied.source.root.instances.first { $0.id == "private-twin" }, original.source.root.instances.first { $0.id == "private-twin" })
        XCTAssertEqual(applied.source.definitions.first, original.source.definitions.first)
        XCTAssertEqual(applied.source.definitions.last?.composition, proposed.composition)
        XCTAssertEqual(try ModelProjectCodec.decode(ModelProjectCodec.encode(.blockModel(applied))), .blockModel(applied))
    }
    func testEphemeralScopeProjectionPreservesCompositionLibraryInsteadOfLegacyEnvelopeRefusal() throws {
        var source = model().source; source.schemaVersion = 3
        source.definitions[0] = selected(source.definitions[0])
        let projection = try ModelScopeEditing.project(source.root, definitions: source.definitions, registry: .builtIn())
        XCTAssertTrue(projection.graph.nodes.contains { $0.id == "features" })
        XCTAssertTrue(projection.graph.nodes.contains { $0.id == "private-twin" })
        XCTAssertEqual(source.definitions[0].composition?.children.count, 5)
    }
    func testWorkflowSelectedDraftMigratesToElevenAndPreservesMixedSourceAndPrivateCopies() throws {
        let original = try WorkflowCodec.migrateModel(.blockModel(model()), projectID: model().projectID).document
        let root = WorkflowNodeAddress(domain: .model, localID: "features")
        let begun = try WorkflowBlockEditing.beginDraft(original, root: root)
        let proposed = selected(begun.drafts[0].proposed)
        let updated = try WorkflowBlockEditing.updateDraft(begun, proposed: proposed, expectedBaseRevision: 4, root: root)
        XCTAssertEqual(updated.schemaVersion, 11); XCTAssertEqual(updated.nodes, original.nodes); XCTAssertEqual(updated.edges, original.edges)
        XCTAssertEqual(updated.definitions, original.definitions); XCTAssertEqual(updated.revision, 4); XCTAssertEqual(updated.projectID, original.projectID)
        let applied = try WorkflowBlockEditing.applyDraft(updated, root: root, expectedBaseRevision: 4, idProvider: { "workflow-private" })
        XCTAssertEqual(applied.schemaVersion, 11); XCTAssertEqual(applied.revision, 5)
        XCTAssertEqual(applied.nodes.first { $0.address.localID == "private-twin" }, original.nodes.first { $0.address.localID == "private-twin" })
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(applied)), applied)
    }
    func testOrdinaryAbsentPolicyDraftsRetainLegacyEnvelopeAndDiscardExactSource() throws {
        let original = model(), path = ["features"]
        let begun = try ModelBlockEditing.beginDraft(original, path: path)
        var proposed = begun.drafts[0].proposed; proposed.title = "A different display title"
        let updated = try ModelBlockEditing.updateDraft(begun, proposed: proposed, expectedBaseRevision: 4, path: path)
        XCTAssertEqual(updated.schemaVersion, 1); XCTAssertEqual(updated.source.schemaVersion, 2)
        XCTAssertEqual(try ModelBlockEditing.discardDraft(updated, path: path, expectedBaseRevision: 4), original)
        let workflow = try WorkflowCodec.migrateModel(.blockModel(original), projectID: original.projectID).document
        let root = WorkflowNodeAddress(domain: .model, localID: "features")
        let draft = try WorkflowBlockEditing.beginDraft(workflow, root: root)
        let changed = try WorkflowBlockEditing.updateDraft(draft, proposed: proposed, expectedBaseRevision: 4, root: root)
        XCTAssertEqual(changed.schemaVersion, workflow.schemaVersion)
        XCTAssertEqual(try WorkflowBlockEditing.discardDraft(changed, root: root, expectedBaseRevision: 4), workflow)
    }
    func testInvalidPolicyGrammarAndStaleContextNeverBecomeASelectedDraft() throws {
        let begun = try ModelBlockEditing.beginDraft(model(), path: ["features"])
        var proposed = selected(begun.drafts[0].proposed)
        proposed.composition = .init(kind: .namedMap, children: [.init(localID: "branch-a", key: "forward")])
        XCTAssertThrowsError(try ModelBlockEditing.updateDraft(begun, proposed: proposed, expectedBaseRevision: 4, path: ["features"]))
        XCTAssertThrowsError(try ModelBlockEditing.updateDraft(begun, proposed: selected(begun.drafts[0].proposed), expectedBaseRevision: 3, path: ["features"]))
        XCTAssertEqual(begun.schemaVersion, 1); XCTAssertEqual(begun.source.schemaVersion, 2)
    }
    private func rawMigratedModel(origin: [String: Any]) throws -> Data {
        let original = model()
        var migrated = try ModelProjectCodec.enablingComposition(ModelBlockEditing.beginDraft(original, path: ["features"]))
        migrated.drafts[0].proposed = selected(migrated.drafts[0].proposed)
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: ModelProjectCodec.encode(.blockModel(migrated))) as? [String: Any])
        var drafts = try XCTUnwrap(raw["drafts"] as? [[String: Any]])
        drafts[0]["migrationOrigin"] = origin; raw["drafts"] = drafts
        return try JSONSerialization.data(withJSONObject: raw, options: [.sortedKeys])
    }
    func testColdModelDraftRetainsOriginAndDiscardRestoresOriginalEnvelopeExactly() throws {
        let bytes = try rawMigratedModel(origin: ["kind": "model", "modelSchemaVersion": 1, "sourceSchemaVersion": 2])
        guard case .blockModel(let cold) = try ModelProjectCodec.decode(bytes) else { return XCTFail("Expected model owner") }
        let roundtrip = try JSONDecoder().decode(JSONValue.self, from: ModelProjectCodec.encode(.blockModel(cold)))
        guard case .object(let root) = roundtrip, case .array(let drafts) = root["drafts"], case .object(let draft) = drafts[0] else { return XCTFail("Expected retained draft") }
        XCTAssertEqual(draft["migrationOrigin"], .object(["kind": .string("model"), "modelSchemaVersion": 1, "sourceSchemaVersion": 2]))
        XCTAssertEqual(try ModelBlockEditing.discardDraft(cold, path: ["features"], expectedBaseRevision: 4), model())
    }
    func testTamperedModelMigrationOriginsRefuseInsteadOfBeingSilentlyDropped() throws {
        for origin in [
            ["kind": "model", "modelSchemaVersion": 2, "sourceSchemaVersion": 2],
            ["kind": "model", "modelSchemaVersion": 1, "sourceSchemaVersion": 3],
            ["kind": "workflow", "workflowSchemaVersion": 10],
            ["kind": "model", "modelSchemaVersion": 1, "sourceSchemaVersion": 2, "extra": true],
            ["kind": "model", "modelSchemaVersion": true, "sourceSchemaVersion": 2]
        ] as [[String: Any]] { XCTAssertThrowsError(try ModelProjectCodec.decode(rawMigratedModel(origin: origin))) }
    }

    func testColdWorkflowDraftRestoresItsOriginalWorkflowVersion() throws {
        let original = try WorkflowCodec.migrateModel(.blockModel(model()), projectID: model().projectID).document
        let address = WorkflowNodeAddress(domain: .model, localID: "features")
        var migrated = try WorkflowBlockEditing.beginDraft(original, root: address)
        migrated.schemaVersion = 11; migrated.drafts[0].proposed = selected(migrated.drafts[0].proposed)
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: WorkflowCodec.encode(migrated)) as? [String: Any])
        var drafts = try XCTUnwrap(raw["drafts"] as? [[String: Any]])
        drafts[0]["migrationOrigin"] = ["kind": "workflow", "workflowSchemaVersion": original.schemaVersion]
        raw["drafts"] = drafts
        let cold = try WorkflowCodec.decode(JSONSerialization.data(withJSONObject: raw, options: [.sortedKeys]))
        XCTAssertEqual(try WorkflowBlockEditing.discardDraft(cold, root: address, expectedBaseRevision: 4), original)
    }

}
