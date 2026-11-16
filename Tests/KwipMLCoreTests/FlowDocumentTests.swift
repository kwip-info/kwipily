import XCTest
@testable import KwipMLCore

final class FlowDocumentTests: XCTestCase {
    let rootID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!
    let childID = UUID(uuidString: "00000000-0000-0000-0000-000000000002")!
    func child() -> FlowDocument {
        .init(id: childID, payload: .modelBlock(CompositeExamples.parallelClassifier.definitions[0]))
    }
    func parent(child: FlowDocument) throws -> FlowDocument {
        let definition = CompositeExamples.parallelClassifier.definitions[0]
        let workflow = WorkflowDocument(projectID: rootID.uuidString, nodes: [
            .init(address: .init(domain: .model, localID: "features"), name: "My features",
                  payload: .modelBlock(reference: definition.reference, overrides: ["branchWidth": 4]))
        ])
        return .init(id: rootID, payload: .workflow(workflow), dependencies: [
            .init(definition: definition.reference, revision: try FlowDocumentCodec.reference(child), relativePath: "subflows/features.kwipflow")
        ])
    }
    func testRoundTripPreservesAuthoredPayloadAndPinnedResolution() throws {
        let child = child(), parent = try parent(child: child)
        XCTAssertEqual(try FlowDocumentCodec.decode(FlowDocumentCodec.encode(parent)), parent)
        let snapshot = try FlowDocumentResolver.resolve(parent) { reference in
            XCTAssertEqual(reference.documentID, self.childID); return child
        }
        let workflow = try snapshot.workflow()
        XCTAssertEqual(workflow.definitions, [CompositeExamples.parallelClassifier.definitions[0]])
        XCTAssertEqual(workflow.nodes.first?.address.localID, "features")
        guard case .modelBlock(_, let overrides) = workflow.nodes.first?.payload else { return XCTFail() }
        XCTAssertEqual(overrides["branchWidth"], 4)
    }
    func testChangedRevisionOrSameRevisionDifferentContentRefuses() throws {
        let original = child(), parent = try parent(child: original)
        var changed = original; changed.revision = 1
        XCTAssertThrowsError(try FlowDocumentResolver.resolve(parent) { _ in changed })
        changed = original
        if case .modelBlock(var definition) = changed.payload {
            definition.title = "Changed externally"; changed.payload = .modelBlock(definition)
        }
        XCTAssertThrowsError(try FlowDocumentResolver.resolve(parent) { _ in changed })
        // Resolving the archived revision is still valid after a newer document exists.
        XCTAssertNoThrow(try FlowDocumentResolver.resolve(parent) { _ in original })
    }
    func testMissingDependencyAndCycleHaveExplicitErrors() throws {
        var child = child()
        let parent = try parent(child: child)
        XCTAssertThrowsError(try FlowDocumentResolver.resolve(parent) { _ in throw FlowDocumentError.missingDependency })
        let reference = CompositeDefinitionReference("back-to-root")
        child.dependencies = [.init(definition: reference, revision: try FlowDocumentCodec.reference(parent), relativePath: "parent.kwipflow")]
        var cyclic = try self.parent(child: child)
        // ID cycle is rejected before a lookup can substitute a different root revision.
        cyclic.dependencies[0].revision = try FlowDocumentCodec.reference(child)
        XCTAssertThrowsError(try FlowDocumentResolver.resolve(cyclic) { _ in child }) { error in
            XCTAssertEqual(error as? FlowDocumentError, .cycle)
        }
    }
    func testPathsVersionsAndUnknownFieldsAreRejected() throws {
        let child = child(); var parent = try parent(child: child)
        for path in ["../outside.kwipflow", "/tmp/external.kwipflow", "a//b", "a/./b", "https://example.com/file", "a\\b"] {
            parent.dependencies[0].relativePath = path
            XCTAssertThrowsError(try FlowDocumentCodec.encode(parent), path)
        }
        parent = try self.parent(child: child); parent.schemaVersion = 99
        XCTAssertThrowsError(try FlowDocumentCodec.encode(parent))
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: FlowDocumentCodec.encode(child)) as? [String:Any])
        raw["unknown"] = true
        XCTAssertThrowsError(try FlowDocumentCodec.decode(JSONSerialization.data(withJSONObject: raw)))
    }
    func testDuplicateBindingsAndResourceLimitsRefuse() throws {
        let child = child(); var parent = try parent(child: child)
        parent.dependencies.append(parent.dependencies[0])
        XCTAssertThrowsError(try FlowDocumentCodec.encode(parent))
        parent = try self.parent(child: child)
        XCTAssertThrowsError(try FlowDocumentResolver.resolve(parent, maximumDocuments: 1) { _ in child })
        XCTAssertThrowsError(try FlowDocumentResolver.resolve(parent, maximumDepth: 0) { _ in child })
    }
    func testPortableSnapshotDoesNotReadMutableSources() throws {
        let child = child(), parent = try parent(child: child)
        let snapshot = try FlowDocumentResolver.resolve(parent) { _ in child }
        let data = try JSONEncoder().encode(snapshot)
        let restored = try JSONDecoder().decode(FlowDocumentSnapshot.self, from: data)
        XCTAssertEqual(try restored.workflow(), try snapshot.workflow())
        var tampered = restored
        tampered.documents.removeAll { $0.id == child.id }
        XCTAssertThrowsError(try tampered.workflow())
    }
    func testSchemaOneBytesDoNotGainDraftFields() throws {
        let document = child()
        let bytes = try FlowDocumentCodec.encode(document)
        let raw = try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        XCTAssertNil(raw["draftDependencies"])
        XCTAssertEqual(try FlowDocumentCodec.encode(FlowDocumentCodec.decode(bytes)),bytes)
    }
    func testDraftPayloadStrictVersionMetadataAndLinks() throws {
        let definition = CompositeExamples.parallelClassifier.definitions[0]
        let draft = ModelBlockDraft(projectID:rootID.uuidString,instancePath:["block"],
            baseReference:definition.reference,baseRevision:17,proposed:definition)
        var document = FlowDocument(payload:.modelBlockDraft(draft))
        XCTAssertEqual(document.schemaVersion,2)
        XCTAssertEqual(try FlowDocumentCodec.decode(FlowDocumentCodec.encode(document)),document)
        document.schemaVersion = 1
        XCTAssertThrowsError(try FlowDocumentCodec.encode(document))
        document.schemaVersion = 2
        var invalid = draft; invalid.instancePath = []
        document.payload = .modelBlockDraft(invalid)
        XCTAssertThrowsError(try FlowDocumentCodec.encode(document))
        document.payload = .modelBlockDraft(draft)
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with:FlowDocumentCodec.encode(document)) as? [String:Any])
        var payload = try XCTUnwrap(raw["payload"] as? [String:Any])
        var content = try XCTUnwrap(payload["content"] as? [String:Any])
        content["unknown"] = true; payload["content"] = content; raw["payload"] = payload
        XCTAssertThrowsError(try FlowDocumentCodec.decode(JSONSerialization.data(withJSONObject:raw)))
    }

}
