import XCTest
@testable import KwipMLCore

final class WorkflowInlineContainerProjectionTests: XCTestCase {
    private let root = WorkflowNodeAddress(domain: .model, localID: "features")
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func fixture() throws -> WorkflowDocument {
        let source = CompositeExamples.parallelClassifier
        let model = ModelAuthoringDocument(projectID: UUID().uuidString, revision: 4, source: source)
        var document = try WorkflowCodec.migrateModel(.blockModel(model), projectID: model.projectID).document
        document.schemaVersion = 11
        document.definitions[0].composition = .init(kind: .graph, children: document.definitions[0].body.nodes.map { .init(localID: $0.id) })
        return document
    }
    private func registries() throws -> (ComponentRegistry, DataFlowOperatorRegistry) {
        (try .builtIn(), try .load(from: repository.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func project(_ document: WorkflowDocument) throws -> WorkflowCanvasProjection {
        let (model, data) = try registries()
        return try WorkflowCanvasProjector.project(document, modelRegistry: model, dataRegistry: data, inlineTrainControls: true)
    }
    private func expand(_ base: WorkflowCanvasProjection, _ document: WorkflowDocument,
                        roots: Set<WorkflowNodeAddress>? = nil, positions: [WorkflowNodeAddress: CGPoint] = [:]) throws -> WorkflowCanvasProjection {
        let (model, data) = try registries()
        return try base.addingInlineMembers(document, expandedRoots: roots ?? [root], positions: positions,
            modelRegistry: model, dataRegistry: data)
    }
    func testParallelExpansionAddsAllFiveActualMembersWhilePreservingRootProjectionAndSource() throws {
        var document = try fixture()
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address == root })
        guard case .modelBlock(let pin, _) = document.nodes[index].payload else { return XCTFail("Missing graph Block") }
        document.nodes[index].payload = .modelBlock(reference: pin, overrides: ["branchWidth": 12])
        let original = try WorkflowCodec.encode(document), base = try project(document)
        let child = WorkflowInlineContainer.address(root: root, childID: "branch-a")
        let expanded = try expand(base, document, positions: [child: .init(x: 123, y: 456)])
        let definition = try WorkflowInlineContainer.definition(in: document, root: root)
        XCTAssertEqual(definition.body.nodes.count, 5)
        XCTAssertEqual(expanded.graph.nodes.count, base.graph.nodes.count + 5)
        XCTAssertEqual(expanded.graph.edges.count, base.graph.edges.count + definition.body.edges.count)
        XCTAssertEqual(try WorkflowCodec.encode(document), original, "Projection cannot mutate authoring or worker source")
        for node in base.graph.nodes { XCTAssertEqual(expanded.graph.nodes.first { $0.id == node.id }, node) }
        for edge in base.graph.edges { XCTAssertEqual(expanded.graph.edges.first { $0.id == edge.id }, edge) }
        for (id, address) in base.addressByDisplayID { XCTAssertEqual(expanded.addressByDisplayID[id], address) }
        for (address, id) in base.displayIDByAddress { XCTAssertEqual(expanded.displayIDByAddress[address], id) }
        for (id, edge) in base.edgeIDByDisplayID { XCTAssertEqual(expanded.edgeIDByDisplayID[id], edge) }
        for (id, type) in base.typeByDisplayID { XCTAssertEqual(expanded.typeByDisplayID[id], type) }
        for port in base.ports {
            XCTAssertEqual(expanded.portDescriptor(address: port.endpoint.address, port: port.endpoint.port, direction: port.direction), port)
        }
        let memberID = try XCTUnwrap(expanded.displayIDByAddress[child])
        let displayed = try XCTUnwrap(expanded.graph.nodes.first { $0.id == memberID })
        XCTAssertEqual(displayed.config["out_features"], 12)
        XCTAssertEqual(displayed.x, 123); XCTAssertEqual(displayed.y, 456)
        let metadata = try XCTUnwrap(expanded.typeByDisplayID[displayed.type])
        XCTAssertEqual(metadata.address, child); XCTAssertEqual(metadata.identifier, "linear")
        XCTAssertNotNil(expanded.portDescriptor(address: child, port: "in", direction: .input)?.contract)
        XCTAssertNotNil(expanded.portDescriptor(address: child, port: "out", direction: .output)?.contract)
        XCTAssertEqual(expanded.registry.component(type: displayed.type)?.title, "Linear")
    }
    func testInternalWireIdentitiesResolveCurrentChildrenAndNeverAddBoundaryLinks() throws {
        let document = try fixture(), base = try project(document), expanded = try expand(base, document)
        let definition = try WorkflowInlineContainer.definition(in: document, root: root)
        let added = expanded.graph.edges.filter { base.edgeIDByDisplayID[$0.id] == nil }
        XCTAssertEqual(added.count, definition.body.edges.count)
        for edge in definition.body.edges {
            let source = WorkflowInlineContainerProjection.internalEdgeID(root: root, edgeID: edge.id)
            let display = try XCTUnwrap(expanded.displayIDByEdgeID[source])
            XCTAssertEqual(expanded.edgeIDByDisplayID[display], source)
            XCTAssertEqual(WorkflowInlineContainerProjection.internalEdge(in: document, displayID: display)?.root, root)
            XCTAssertEqual(WorkflowInlineContainerProjection.internalEdge(in: document, displayID: source)?.edgeID, edge.id)
            let visual = try XCTUnwrap(added.first { $0.id == display })
            XCTAssertEqual(expanded.addressByDisplayID[visual.source], WorkflowInlineContainer.address(root: root, childID: edge.source))
            XCTAssertEqual(expanded.addressByDisplayID[visual.target], WorkflowInlineContainer.address(root: root, childID: edge.target))
        }
        XCTAssertTrue(added.allSatisfy { expanded.addressByDisplayID[$0.source] != root && expanded.addressByDisplayID[$0.target] != root })
        var changed = document
        let removed = changed.definitions[0].body.edges.removeFirst()
        XCTAssertNil(WorkflowInlineContainerProjection.internalEdge(in: changed,
            displayID: WorkflowInlineContainerProjection.internalEdgeID(root: root, edgeID: removed.id)))
        XCTAssertNil(WorkflowInlineContainerProjection.internalEdge(in: document, displayID: "not-a-wire"))
    }
    func testSharedDefinitionInstancesGetDistinctQualifiedMembersAndWires() throws {
        var document = try fixture()
        let other = WorkflowNodeAddress(domain: .model, localID: "other-features")
        let original = try XCTUnwrap(document.nodes.first { $0.address == root })
        document.nodes.append(.init(address: other, name: "Another instance", payload: original.payload))
        let base = try project(document), expanded = try expand(base, document, roots: [root, other])
        let a = WorkflowInlineContainer.address(root: root, childID: "branch-a")
        let b = WorkflowInlineContainer.address(root: other, childID: "branch-a")
        XCTAssertNotEqual(a, b)
        XCTAssertNotEqual(expanded.displayIDByAddress[a], expanded.displayIDByAddress[b])
        XCTAssertEqual(expanded.graph.nodes.count, base.graph.nodes.count + 10)
        let edge = try XCTUnwrap(document.definitions[0].body.edges.first)
        XCTAssertNotEqual(WorkflowInlineContainerProjection.internalEdgeID(root: root, edgeID: edge.id),
                          WorkflowInlineContainerProjection.internalEdgeID(root: other, edgeID: edge.id))
    }
    func testCollapsedUnknownAndUnsupportedRootsDoNotInventMembers() throws {
        let document = try fixture(), base = try project(document)
        XCTAssertEqual(try expand(base, document, roots: []).graph, base.graph)
        XCTAssertEqual(try expand(base, document, roots: [.init(domain: .model, localID: "absent")]).graph, base.graph)
        let primitive = try XCTUnwrap(document.nodes.first { if case .modelPrimitive = $0.payload { return true }; return false })
        XCTAssertEqual(try expand(base, document, roots: [primitive.address]).graph, base.graph)
    }
    func testNonfinitePositionsAndSavedEdgeIdentityConflictsRefuseProjection() throws {
        var document = try fixture()
        let base = try project(document)
        let child = WorkflowInlineContainer.address(root: root, childID: "branch-a")
        XCTAssertThrowsError(try expand(base, document, positions: [child: .init(x: Double.infinity, y: 0)]))
        let edge = try XCTUnwrap(document.definitions[0].body.edges.first)
        document.edges[0].id = WorkflowInlineContainerProjection.internalEdgeID(root: root, edgeID: edge.id)
        XCTAssertThrowsError(try expand(try project(document), document))
        let original = try fixture(), originalBase = try project(original)
        let alreadyExpanded = try expand(originalBase, original)
        XCTAssertThrowsError(try expand(alreadyExpanded, original), "Augmenting an already expanded projection cannot duplicate its members")
        var unrelated = original; unrelated.projectID = UUID().uuidString
        XCTAssertThrowsError(try expand(originalBase, unrelated), "A source from another project cannot augment this projection")
    }
}
