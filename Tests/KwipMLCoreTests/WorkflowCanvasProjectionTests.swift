import XCTest
@testable import KwipMLCore

final class WorkflowCanvasProjectionTests: XCTestCase {
    func testMaximumLengthUnicodeEdgeIdentityRoundTripsThroughProjection() throws {
        let source = WorkflowNodeAddress(domain: .data, localID: "source")
        let target = WorkflowNodeAddress(domain: .data, localID: "target")
        let identity = String(repeating: "🧬", count: 1024)
        let edge = WorkflowEdge(id: identity, from: .init(address: source, port: "table"), to: .init(address: target, port: "table"), kind: .table)
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: source, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: [:])),
            .init(address: target, payload: .dataOperator(operatorID: "kwip.data.cast", config: [:]))], edges: [edge])
        let projection = try project(document)
        let displayID = try XCTUnwrap(projection.graph.edges.first?.id)
        XCTAssertEqual(try WorkflowCanvasIDs.edgeID(from: displayID), identity)
        XCTAssertEqual(projection.edgeIDByDisplayID[displayID], identity)
        XCTAssertEqual(projection.displayIDByEdgeID[identity], displayID)
        XCTAssertThrowsError(try WorkflowCanvasIDs.edgeID(from: WorkflowCanvasIDs.displayEdgeID(identity + "x")))
    }
    func testMaximumLengthEscapedEdgeIdentityRoundTripsWithinBoundedDisplayToken() throws {
        let identity = String(repeating: "\0", count: 1024)
        let displayID = try WorkflowCanvasIDs.displayEdgeID(identity)
        XCTAssertGreaterThan(displayID.utf8.count, 8192)
        XCTAssertLessThan(displayID.utf8.count, 16384)
        XCTAssertEqual(try WorkflowCanvasIDs.edgeID(from: displayID), identity)
        XCTAssertThrowsError(try WorkflowCanvasIDs.edgeID(from: "wf-e-" + String(repeating: "a", count: 16384)))
    }
    func testPortIndexPreservesQualifiedIdentityDirectionAndFirstDescriptor() throws {
        let data = WorkflowEndpoint(address: .init(domain: .data, localID: "same"), port: "value")
        let model = WorkflowEndpoint(address: .init(domain: .model, localID: "same"), port: "value")
        let descriptors: [WorkflowCanvasPortDescriptor] = [
            .init(endpoint: data, direction: .input, contract: nil, unresolvedReason: "saved input"),
            .init(endpoint: data, direction: .output, contract: nil, unresolvedReason: "saved output"),
            .init(endpoint: model, direction: .input, contract: nil, unresolvedReason: "model input"),
            .init(endpoint: data, direction: .input, contract: nil, unresolvedReason: "duplicate")]
        let index = WorkflowCanvasPortIndex(descriptors)
        for descriptor in descriptors {
            XCTAssertEqual(index.descriptor(endpoint: descriptor.endpoint, direction: descriptor.direction),
                           descriptors.first { $0.endpoint == descriptor.endpoint && $0.direction == descriptor.direction })
        }
        XCTAssertNil(index.descriptor(endpoint: model, direction: .output))
        let updated = WorkflowCanvasPortIndex(Array(descriptors.dropFirst()))
        XCTAssertEqual(updated.descriptor(endpoint: data, direction: .input)?.unresolvedReason, "duplicate")
        XCTAssertEqual(index.descriptor(endpoint: data, direction: .input)?.unresolvedReason, "saved input")
    }

    func testPortLookupDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["KWIPML_CANVAS_BENCHMARK"] == "1" else {
            throw XCTSkip("Opt-in timing diagnostic; no wall-clock acceptance threshold")
        }
        let nodes: [WorkflowNode] = (0..<256).map {
            .init(address: .init(domain: $0 < 128 ? .model : .data, localID: "node-\($0)"),
                  payload: $0 < 128 ? .modelPrimitive(type: "linear", configurationVersion: 2, config: [:])
                    : .dataOperator(operatorID: "kwip.data.cast", config: [:]))
        }
        let projection = try project(.init(projectID: UUID().uuidString, nodes: nodes))
        let start = Date()
        var matches = 0
        for _ in 0..<8 {
            for descriptor in projection.ports {
                if projection.portDescriptor(address: descriptor.endpoint.address, port: descriptor.endpoint.port,
                                             direction: descriptor.direction) == descriptor { matches += 1 }
            }
        }
        let elapsed = Date().timeIntervalSince(start) * 1000
        XCTAssertEqual(matches, projection.ports.count * 8)
        print("PORT_LOOKUP descriptors=\(projection.ports.count) reads=\(matches) elapsed_ms=\(elapsed)")
    }

    func testDisabledPoolPortHidesUntilEnabledAndPreservesBrokenSavedWire() throws {
        let source = WorkflowNodeAddress(domain:.model,localID:"pool")
        let target = WorkflowNodeAddress(domain:.model,localID:"target")
        var document = WorkflowDocument(projectID:UUID().uuidString,nodes:[
            .init(address:source,payload:.modelPrimitive(type:"max_pool1d",configurationVersion:2,config:["return_indices":false])),
            .init(address:target,payload:.modelPrimitive(type:"flatten",configurationVersion:2,config:[:]))])
        let hidden = try project(document)
        XCTAssertFalse(hidden.ports.contains { $0.endpoint.address == source && $0.endpoint.port == "indices" })
        document.nodes[0].payload = .modelPrimitive(type:"max_pool1d",configurationVersion:2,config:["return_indices":true])
        let enabled = try project(document)
        XCTAssertNotNil(enabled.ports.first { $0.endpoint.address == source && $0.endpoint.port == "indices" }?.contract)
        document.nodes[0].payload = .modelPrimitive(type:"max_pool1d",configurationVersion:2,config:["return_indices":false])
        document.edges = [.init(from:.init(address:source,port:"indices"),to:.init(address:target,port:"in"),kind:.tensor)]
        let broken = try project(document)
        let endpoint = try XCTUnwrap(broken.ports.first { $0.endpoint.address == source && $0.endpoint.port == "indices" })
        XCTAssertNil(endpoint.contract)
        XCTAssertNotNil(endpoint.unresolvedReason)
        XCTAssertEqual(broken.graph.edges.count,1)
    }
    private func project(_ document: WorkflowDocument) throws -> WorkflowCanvasProjection {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try WorkflowCanvasProjector.project(document, modelRegistry: .builtIn(), dataRegistry: .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    func testQualifiedIdentitiesRawConfigsAndGeometryAreReversibleWithoutSourceChanges() throws {
        let data = WorkflowNodeAddress(domain: .data, localID: "same/a:b.λ")
        let model = WorkflowNodeAddress(domain: .model, localID: data.localID)
        let control = WorkflowNodeAddress(domain: .control, localID: data.localID)
        let config: [String: JSONValue] = ["columns": .object([" a,b ": .string("float64")]), "missingPolicy": .string("preserve")]
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: data, name: "My columns", x: -12, y: 900, payload: .dataOperator(operatorID: "kwip.data.cast", config: config)),
            .init(address: model, name: "My affine", x: 123, y: 45, payload: .modelPrimitive(type: "linear", configurationVersion: 1, config: ["in_features": 2, "out_features": 2])),
            .init(address: control, name: "My recipe", x: 640, y: 50, payload: .runSetup(config: ["epochs": 5]))])
        let original = document, projection = try project(document)
        XCTAssertEqual(document, original)
        XCTAssertEqual(Set(projection.graph.nodes.map(\.id)).count, 3)
        for node in document.nodes {
            let id = try XCTUnwrap(projection.displayIDByAddress[node.address])
            XCTAssertEqual(projection.addressByDisplayID[id], node.address)
            XCTAssertEqual(try WorkflowCanvasIDs.address(from: id), node.address)
            let displayed = try XCTUnwrap(projection.graph.nodes.first { $0.id == id })
            XCTAssertEqual(displayed.name, node.name); XCTAssertEqual(displayed.x, node.x); XCTAssertEqual(displayed.y, node.y)
            XCTAssertEqual(try WorkflowCanvasIDs.type(from: displayed.type), projection.typeByDisplayID[displayed.type])
        }
        XCTAssertEqual(projection.graph.nodes[0].config, config)
        XCTAssertNil(projection.graph.nodes[1].config["bias"])
        var reordered = document; reordered.nodes.reverse()
        XCTAssertEqual(try project(reordered).displayIDByAddress, projection.displayIDByAddress)
        XCTAssertEqual(try project(reordered).typeByDisplayID, projection.typeByDisplayID)
        XCTAssertThrowsError(try WorkflowCanvasIDs.address(from: "model:same/a:b.λ"))
        XCTAssertThrowsError(try WorkflowCanvasIDs.address(from: projection.graph.nodes[0].id + "="))
    }
    func testUnwiredNodesExposeAuthoritativePortsAndTypedControlDescriptors() throws {
        let definitions = CompositeExamples.parallelClassifier.definitions
        let items: [(String, WorkflowNodePayload)] = [
            ("linear", .modelPrimitive(type: "linear", configurationVersion: 2, config: [:])),
            ("output", .modelPrimitive(type: "output", configurationVersion: 2, config: [:])),
            ("block", .modelBlock(reference: definitions[0].reference, overrides: [:])),
            ("definition", .modelDefinition), ("mapping", .trainingData(config: [:])),
            ("setup", .runSetup(config: [:])), ("train", .train(config: [:]))]
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: items.map { .init(address: .init(domain: $0.1.domain, localID: $0.0), payload: $0.1) }, definitions: definitions)
        let projection = try project(document)
        XCTAssertTrue(projection.issues.isEmpty)
        let linear = WorkflowNodeAddress(domain: .model, localID: "linear")
        XCTAssertEqual(projection.portDescriptor(address: linear, port: "in", direction: .input)?.contract?.maximumConnections, 1)
        XCTAssertEqual(projection.portDescriptor(address: linear, port: "out", direction: .output)?.semanticKind, .tensor)
        XCTAssertEqual(projection.portDescriptor(address: .init(domain: .model, localID: "output"), port: "definition", direction: .output)?.semanticKind, .modelDefinition)
        XCTAssertNil(projection.portDescriptor(address: .init(domain: .model, localID: "output"), port: "out", direction: .output))
        XCTAssertEqual(projection.portDescriptor(address: .init(domain: .control, localID: "mapping"), port: "data", direction: .output)?.semanticKind, .trainingData)
        XCTAssertEqual(projection.portDescriptor(address: .init(domain: .control, localID: "setup"), port: "configuration", direction: .output)?.semanticKind, .runConfiguration)
        XCTAssertEqual(projection.graph.nodes.first { projection.addressByDisplayID[$0.id]?.localID == "block" }?.config, [:])
    }
    func testUnknownOperatorAndUndeclaredWirePortsStayVisibleWithoutInventingContracts() throws {
        let unknown = WorkflowNodeAddress(domain: .data, localID: "unknown")
        let known = WorkflowNodeAddress(domain: .data, localID: "cast")
        let edge = WorkflowEdge(id: "wire:1/a", from: .init(address: unknown, port: "custom"), to: .init(address: known, port: "wrong"), kind: .table)
        let document = WorkflowDocument(projectID: UUID().uuidString, nodes: [
            .init(address: unknown, name: "Unavailable but mine", payload: .dataOperator(operatorID: "third.party.missing", config: ["keep": .bool(true)])),
            .init(address: known, payload: .dataOperator(operatorID: "kwip.data.cast", config: [:]))], edges: [edge])
        let projection = try project(document)
        XCTAssertEqual(projection.graph.nodes.count, 2); XCTAssertEqual(projection.graph.edges.count, 1)
        XCTAssertEqual(try WorkflowCanvasIDs.edgeID(from: projection.graph.edges[0].id), edge.id)
        XCTAssertEqual(projection.edgeIDByDisplayID[projection.graph.edges[0].id], edge.id)
        XCTAssertTrue(projection.issues.contains { $0.address == unknown })
        XCTAssertTrue(projection.issues.contains { $0.address == known && $0.port == "wrong" })
        let descriptor = try XCTUnwrap(projection.portDescriptor(address: unknown, port: "custom", direction: .output))
        XCTAssertNil(descriptor.semanticKind); XCTAssertNil(descriptor.contract)
        XCTAssertNotNil(descriptor.unresolvedReason)
        let node = projection.graph.nodes[0]
        XCTAssertEqual(projection.registry.component(type: node.type)?.outputs, ["custom"])
        XCTAssertEqual(node.config, ["keep": .bool(true)])
    }
}
