import XCTest
@testable import KwipMLCore

final class WorkflowEditingTests: XCTestCase {
    private func address(_ domain: WorkflowDomain, _ id: String) -> WorkflowNodeAddress { .init(domain: domain, localID: id) }
    private func node(_ id: String, _ op: String = "kwip.data.columns") -> WorkflowNode {
        .init(address: address(.data, id), name: id, payload: .dataOperator(operatorID: op, config: [:]))
    }
    private func apply(_ command: WorkflowCommand, _ document: WorkflowDocument, selection: WorkflowSelection = .init(), makeID: () -> String = { UUID().uuidString }) throws -> WorkflowEditResult {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowEditing.apply(command, to: document, selection: selection, modelRegistry: .builtIn(), dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")), makeID: makeID)
    }
    func testAPIOutputMutationRefusesSecretAndRepairsExistingDraft()throws {
        let id=address(.data,"publish")
        guard case .object(let variant)=APITableSettings.discoveryVariant(direction:"output"),case .object(let valid)=variant["exampleConfiguration"],case .object(var api)=valid["api"] else {return XCTFail("Missing example")}
        var invalid=valid;api["headers"] = .object(["Authorization":.string("PRIVATE-SENTINEL")]);invalid["api"] = .object(api)
        var old=WorkflowDocument(projectID:UUID().uuidString)
        old.nodes=[.init(address:id,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:invalid))]
        XCTAssertNoThrow(try WorkflowCodec.encode(old))
        XCTAssertThrowsError(try apply(.configure(address:id,config:invalid),old)){error in
            XCTAssertEqual((error as? DataFlowContractError)?.apiIssue,.requestFields)
            XCTAssertFalse(error.localizedDescription.contains("PRIVATE-SENTINEL"))
        }
        var wrongShape=valid;wrongShape["api"] = .array([])
        XCTAssertThrowsError(try apply(.configure(address:id,config:wrongShape),old)){error in XCTAssertEqual((error as? DataFlowContractError)?.apiIssue,.outputEnvelope)}
        let repaired=try apply(.configure(address:id,config:valid),old)
        XCTAssertEqual(repaired.document.revision,old.revision+1)
        XCTAssertEqual(old.nodes[0].payload,.dataOperator(operatorID:"kwip.data.outputDelimited",config:invalid))
    }
    func testPrepareMediaMutationValidatesWithoutBlockingOldDocumentRepair() throws {
        let id=address(.data,"media")
        var valid=try PrepareMediaConfiguration.defaults()
        var invalid=valid;invalid["matte"] = .array([255,255,255])
        var old=WorkflowDocument(projectID:UUID().uuidString)
        old.nodes=[.init(address:id,payload:.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:invalid))]
        XCTAssertNoThrow(try WorkflowCodec.encode(old))
        XCTAssertThrowsError(try apply(.configure(address:id,config:invalid),old))
        valid["matte"] = .array([0,0.5,1])
        let repaired=try apply(.configure(address:id,config:valid),old)
        XCTAssertEqual(repaired.document.revision,old.revision+1)
        guard case .dataOperator(_,let actual)=repaired.document.nodes[0].payload else {return XCTFail("Missing media")}
        XCTAssertEqual(actual,valid)
        XCTAssertEqual(old.nodes[0].payload,.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:invalid))
    }
    func testInsertRenameConfigureMovePreserveIncompleteRawSourceAndNoOpRevision() throws {
        let id = address(.model, "affine")
        let source = WorkflowDocument(projectID: UUID().uuidString)
        let item = WorkflowNode(address: id, name: "Original", x: 7, y: 8, payload: .modelPrimitive(type: "linear", configurationVersion: 1, config: [:]))
        let inserted = try apply(.insert(item), source)
        XCTAssertEqual(inserted.document.revision, 1)
        XCTAssertEqual(inserted.document.nodes, [item])
        let renamed = try apply(.rename(address: id, name: "User label"), inserted.document)
        let configured = try apply(.configure(address: id, config: ["in_features": 2]), renamed.document)
        let moved = try apply(.move(addresses: [id], offset: .init(x: 10, y: -3)), configured.document)
        XCTAssertEqual(moved.document.revision, 4)
        XCTAssertEqual(moved.document.nodes[0].x, 17); XCTAssertEqual(moved.document.nodes[0].y, 5)
        XCTAssertEqual(moved.document.nodes[0].payload, .modelPrimitive(type: "linear", configurationVersion: 1, config: ["in_features": 2]))
        XCTAssertEqual(try apply(.configure(address: id, config: ["in_features": 2]), moved.document).document, moved.document)
        XCTAssertTrue(source.nodes.isEmpty)
    }
    func testConnectionsUseActualKindsCardinalityCyclesAndReconnectIdentity() throws {
        let doc = WorkflowDocument(projectID: UUID().uuidString, nodes: [node("source", "kwip.data.sourceCSV"), node("other", "kwip.data.sourceCSV"), node("a"), node("b"), .init(address: address(.model, "linear"), payload: .modelPrimitive(type: "linear", configurationVersion: 2, config: [:]))])
        func ep(_ id: String) -> WorkflowEndpoint { .init(address: address(.data, id), port: "table") }
        XCTAssertThrowsError(try apply(.connect(from: ep("source"), to: .init(address: address(.model, "linear"), port: "in")), doc))
        let first = try apply(.connect(from: ep("source"), to: ep("a")), doc, makeID: { "wire" })
        XCTAssertEqual(first.document.edges[0].kind, .table)
        XCTAssertThrowsError(try apply(.connect(from: ep("other"), to: ep("a")), first.document))
        let reconnected = try apply(.reconnect(edgeID: "wire", from: ep("other"), to: ep("a")), first.document)
        XCTAssertEqual(reconnected.document.edges[0].id, "wire")
        XCTAssertEqual(reconnected.document.edges[0].from, ep("other"))
        let chain = try apply(.connect(from: ep("a"), to: ep("b")), reconnected.document)
        XCTAssertThrowsError(try apply(.reconnect(edgeID: "wire", from: ep("b"), to: ep("a")), chain.document))
        XCTAssertEqual(doc.edges, [])
        XCTAssertEqual(chain.document.edges[0].from, ep("other"))
    }
    func testDuplicateMixedDomainsPreservesPinsRawConfigAndOnlyInternalEdges() throws {
        let definition = CompositeExamples.parallelClassifier.definitions[0]
        let block = address(.model, "block"), primitive = address(.model, "shared"), data = address(.data, "shared")
        let internalEdge = WorkflowEdge(id: "inside", from: .init(address: block, port: "out"), to: .init(address: primitive, port: "in"), kind: .tensor)
        let doc = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: block, name: "B", x: 1, y: 2, payload: .modelBlock(reference: definition.reference, overrides: ["branchWidth": 3])),
            .init(address: primitive, name: "P", x: 5, y: 6, payload: .modelPrimitive(type: "linear", configurationVersion: 1, config: ["in_features": 6, "out_features": 2])),
            node("shared", "kwip.data.sourceCSV"), node("outside")],
            edges: [internalEdge, .init(id: "boundary", from: .init(address: data, port: "table"), to: .init(address: address(.data, "outside"), port: "table"), kind: .table)], definitions: [definition])
        var next = 0
        let result = try apply(.duplicate(addresses: [block, primitive, data], offset: .init(x: 20, y: 30)), doc, makeID: { next += 1; return "copy-\(next)" })
        XCTAssertEqual(result.document.nodes.count, 7); XCTAssertEqual(result.document.edges.count, 3)
        for original in doc.nodes.prefix(3) {
            let copied = try XCTUnwrap(result.copies.nodes[original.address])
            let node = try XCTUnwrap(result.document.nodes.first { $0.address == copied })
            XCTAssertEqual(node.payload, original.payload); XCTAssertEqual(node.name, original.name)
            XCTAssertEqual(node.x, original.x + 20); XCTAssertEqual(node.y, original.y + 30)
        }
        XCTAssertEqual(result.copies.edges.count, 1)
        XCTAssertNil(result.copies.edges["boundary"])
        XCTAssertEqual(result.document.definitions, doc.definitions)
        XCTAssertEqual(result.document.revision, 1)
        XCTAssertThrowsError(try apply(.duplicate(addresses: [data], offset: .init()), doc, makeID: { "shared" }))
    }
    func testDeletePrunesSelectionAndAtomicInvalidMutationLeavesSourceUntouched() throws {
        let a = address(.data, "a"), b = address(.data, "b")
        let doc = WorkflowDocument(projectID: UUID().uuidString, nodes: [node("a"), node("b")], edges: [.init(id: "edge", from: .init(address: a, port: "table"), to: .init(address: b, port: "table"), kind: .table)])
        let result = try apply(.delete(.init(nodeAddresses: [a])), doc, selection: .init(nodeAddresses: [a, b], edgeIDs: ["edge"]))
        XCTAssertEqual(result.document.nodes.map(\.address), [b]); XCTAssertTrue(result.document.edges.isEmpty)
        XCTAssertEqual(result.selection.nodeAddresses, [b]); XCTAssertTrue(result.selection.edgeIDs.isEmpty)
        XCTAssertThrowsError(try apply(.move(addresses: [a], offset: .init(x: .infinity, y: 0)), doc))
        XCTAssertThrowsError(try apply(.configure(address: a, config: ["bad": .number(.nan)]), doc))
        XCTAssertEqual(doc.nodes.count, 2); XCTAssertEqual(doc.edges.count, 1)
    }
    func testPendingDraftMissingPinAndRevisionOverflowFailWithoutPromotionOrMutation() throws {
        var source = ModelAuthoringDocument(projectID: UUID().uuidString, source: CompositeExamples.parallelClassifier)
        source = try ModelBlockEditing.beginDraft(source, path: ["features"])
        let document = try WorkflowCodec.migrateModel(.blockModel(source), projectID: source.projectID).document
        XCTAssertThrowsError(try apply(.move(addresses: [address(.model, "input")], offset: .init()), document))
        let empty = WorkflowDocument(projectID: UUID().uuidString)
        XCTAssertThrowsError(try apply(.insert(.init(address: address(.model, "block"), payload: .modelBlock(reference: .init("absent"), overrides: [:]))), empty))
        var exhausted = WorkflowDocument(projectID: UUID().uuidString, revision: .max, nodes: [node("a")])
        XCTAssertEqual(try apply(.rename(address: address(.data, "a"), name: "a"), exhausted).document, exhausted)
        XCTAssertThrowsError(try apply(.rename(address: address(.data, "a"), name: "changed"), exhausted))
        exhausted.nodes[0].address.domain = .model
        XCTAssertThrowsError(try apply(.insert(node("b")), exhausted))
    }
}
