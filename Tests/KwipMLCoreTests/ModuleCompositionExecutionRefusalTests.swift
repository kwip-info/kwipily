import XCTest
@testable import KwipMLCore

/// New authoring intent must never silently run through the old flat executor.
final class ModuleCompositionExecutionRefusalTests: XCTestCase {
    private func selected(_ kind: CompositionPolicy.Kind) -> CompositeProjectDocument {
        var source = CompositeExamples.parallelClassifier
        source.definitions[0].composition = CompositionPolicy(kind: kind, children:
            source.definitions[0].body.nodes.enumerated().map { index, node in
                CompositionPolicy.Child(localID: node.id, key: kind == .namedMap ? "stage_\(index)" : nil)
            })
        return source
    }

    func testLegacyExpansionRefusesEveryExplicitContainerKind() throws {
        for kind in [CompositionPolicy.Kind.graph, .sequential, .orderedList, .namedMap] {
            for version in [1, 2] {
                var source = selected(kind); source.schemaVersion = version
                XCTAssertThrowsError(try CompositeExpander.expand(source, registry: .builtIn())) { error in
                    XCTAssertEqual((error as? CompositeContractError)?.code, .unsupportedSchema)
                }
            }
        }
    }

    func testLegacyModelCodecCannotPersistAndExecuteUnadmittedOwnership() throws {
        let projectID = UUID().uuidString
        for kind in [CompositionPolicy.Kind.graph, .orderedList, .namedMap] {
            let document = ModelProjectDocument.blockModel(ModelAuthoringDocument(projectID: projectID, source: selected(kind)))
            XCTAssertThrowsError(try ModelProjectCodec.encode(document))
            XCTAssertThrowsError(try ModelExecutionSnapshotBuilder.build(document, registry: .builtIn()))
        }
    }

    func testAbsentPolicyPreservesExistingModelAndExpansion() throws {
        let source = CompositeExamples.parallelClassifier
        let expected = try CompositeExpander.expand(source, registry: .builtIn())
        let document = ModelProjectDocument.blockModel(ModelAuthoringDocument(projectID: UUID().uuidString, source: source))
        let decoded = try ModelProjectCodec.decode(ModelProjectCodec.encode(document))
        let snapshot = try ModelExecutionSnapshotBuilder.build(decoded, registry: .builtIn())
        XCTAssertEqual(snapshot.graph, expected.graph)
        XCTAssertEqual(snapshot.origins, expected.nodeOrigins)
        XCTAssertEqual(snapshot.provenance?.nativeExpansionContractVersion, 1)
    }
}
