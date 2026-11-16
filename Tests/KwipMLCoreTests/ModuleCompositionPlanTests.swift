import Foundation
import CryptoKit
import XCTest
@testable import KwipMLCore

final class ModuleCompositionPlanTests: XCTestCase {
    private func source(selected: Bool = true, nested: Bool = false) -> CompositeProjectDocument {
        let inner = CompositeDefinition(reference: .init("inner"), title: "Two steps", body: .init(nodes: [
            .init(id: "a", type: "linear", config: ["in_features": 2, "out_features": 2]), .init(id: "b", type: "relu")
        ], edges: [.init(id: "ab", source: "a", target: "b")]), inputs: [.init("in", targets: [.init("a", "in")])], outputs: [.init("out", source: .init("b", "out"))], composition: selected ? .init(kind: .sequential, children: [.init(localID: "a"), .init(localID: "b")]) : nil)
        let outer = CompositeDefinition(reference: .init("outer"), title: "Wrapper", body: .init(instances: [.init(id: "inner-use", definition: inner.reference)]), inputs: [.init("in", targets: [.init("inner-use", "in")])], outputs: [.init("out", source: .init("inner-use", "out"))])
        return .init(schemaVersion: selected ? 3 : 2, root: .init(nodes: [.init(id: "input", type: "input", config: ["features": 2]), .init(id: "output", type: "output")], instances: [.init(id: "block", definition: nested ? outer.reference : inner.reference)], edges: [.init(id: "enter", source: "input", target: "block"), .init(id: "leave", source: "block", target: "output")]), definitions: nested ? [inner, outer] : [inner])
    }
    private func cid(_ path: [String]) -> String {
        let bytes = path.reduce(into: Data()) { data, part in data.append(Data("\(part.utf8.count):".utf8)); data.append(Data(part.utf8)) }
        return "c-" + SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
    }
    private func plainGraph() -> GraphDocument {
        .init(nodes: [.init(id: "input", type: "input", config: ["features": 2]), .init(id: "a", type: "linear", config: ["in_features": 2, "out_features": 2]), .init(id: "b", type: "relu"), .init(id: "output", type: "output")], edges: [.init(id: "ia", source: "input", target: "a"), .init(id: "ab", source: "a", target: "b"), .init(id: "bo", source: "b", target: "output")])
    }
    private func rootPlan(children: [CompositionPlan.Child]? = nil) -> CompositionPlan {
        .init(containers: [.init(id: "root", kind: .graph, path: [], children: children ?? [.init(kind: .node, id: "a", key: "0"), .init(kind: .node, id: "b", key: "1")], inputs: [.init(id: "input", targets: [.init(node: "a", port: "in")])], outputs: [.init(id: "output", source: .init(node: "b", port: "out"))])])
    }
    func testSelectedExpansionKeepsLeafIDsAndBuildsPreorderResolvedBoundaries() throws {
        let registry = try ComponentRegistry.builtIn()
        let old = try CompositeExpander.expand(source(selected: false, nested: true), registry: registry)
        let actual = try CompositeExpander.expand(source(nested: true), registry: registry)
        XCTAssertEqual(actual.graph.nodes, old.graph.nodes); XCTAssertEqual(actual.graph.edges, old.graph.edges)
        XCTAssertEqual(actual.nodeOrigins, old.nodeOrigins)
        let plan = try XCTUnwrap(actual.graph.compositionPlan)
        XCTAssertEqual(plan.containers.map(\.id), ["root", cid(["block"]), cid(["block", "inner-use"])])
        XCTAssertEqual(plan.containers.map(\.kind), [.graph, .graph, .sequential])
        XCTAssertEqual(plan.containers.map(\.path), [[], ["block"], ["block", "inner-use"]])
        XCTAssertEqual(plan.containers[0].children, [.init(kind: .container, id: cid(["block"]), key: "0")])
        XCTAssertEqual(plan.containers[2].children.map(\.id), actual.graph.nodes.filter { !["input", "output"].contains($0.type) }.map(\.id))
        XCTAssertEqual(plan.containers[0].inputs[0].id, "input")
        XCTAssertEqual(plan.containers[0].inputs[0].targets, plan.containers[2].inputs[0].targets)
        XCTAssertEqual(plan.containers[0].outputs[0].id, "output")
        try plan.validate(graph: actual.graph, registry: registry)
    }
    func testRegistrationOrderAffectsIdentityWithoutChangingFlatDag() throws {
        let registry = try ComponentRegistry.builtIn()
        var a = source(); a.definitions[0].composition = .init(kind: .orderedList, children: [.init(localID: "a"), .init(localID: "b")])
        var b = a; b.definitions[0].composition = .init(kind: .orderedList, children: [.init(localID: "b"), .init(localID: "a")])
        let one = try CompositeExpander.expand(a, registry: registry).graph, two = try CompositeExpander.expand(b, registry: registry).graph
        XCTAssertEqual(one.nodes, two.nodes); XCTAssertEqual(one.edges, two.edges)
        XCTAssertNotEqual(try GraphSemantics.canonicalData(one, registry: registry), try GraphSemantics.canonicalData(two, registry: registry))
        var moved = one; moved.nodes[0].x += 17; moved.nodes[0].name = "Changed label"
        XCTAssertEqual(try GraphSemantics.canonicalData(one, registry: registry), try GraphSemantics.canonicalData(moved, registry: registry))
    }
    func testNestedInstanceRegistrationOrderControlsPreorderWithoutRekeyingLeaves() throws {
        var project = source(nested: true)
        let inner = project.definitions[0].reference
        project.definitions[1].body = .init(instances: [.init(id: "left", definition: inner), .init(id: "right", definition: inner)], edges: [.init(id: "lr", source: "left", target: "right")])
        project.definitions[1].inputs = [.init("in", targets: [.init("left", "in")])]
        project.definitions[1].outputs = [.init("out", source: .init("right", "out"))]
        project.definitions[1].composition = .init(kind: .orderedList, children: [.init(localID: "right"), .init(localID: "left")])
        let actual = try CompositeExpander.expand(project, registry: .builtIn())
        let plan = try XCTUnwrap(actual.graph.compositionPlan)
        XCTAssertEqual(plan.containers.map(\.path), [[], ["block"], ["block", "right"], ["block", "left"]])
        XCTAssertEqual(Set(plan.containers.flatMap(\.children).filter { $0.kind == .node }.map(\.id)).count, 4)
        var legacy = project; legacy.schemaVersion = 2
        for i in legacy.definitions.indices { legacy.definitions[i].composition = nil }
        let prior = try CompositeExpander.expand(legacy, registry: .builtIn())
        XCTAssertEqual(actual.graph.nodes, prior.graph.nodes); XCTAssertEqual(actual.graph.edges, prior.graph.edges)
    }
    func testPlanCoverageKeysBoundariesAndTypedNamespacesRefuseTampering() throws {
        let graph = plainGraph(), registry = try ComponentRegistry.builtIn()
        try rootPlan().validate(graph: graph, registry: registry)
        let invalidChildren: [[CompositionPlan.Child]] = [[], [.init(kind: .node, id: "a", key: "0")], [.init(kind: .node, id: "a", key: "0"), .init(kind: .node, id: "a", key: "1")], [.init(kind: .container, id: "a", key: "0"), .init(kind: .node, id: "b", key: "1")], [.init(kind: .node, id: "a", key: "wrong"), .init(kind: .node, id: "b", key: "1")]]
        for children in invalidChildren {
            XCTAssertThrowsError(try rootPlan(children: children).validate(graph: graph, registry: registry))
        }
        let root = rootPlan().containers[0]
        let foreign = CompositionPlan.Container(id: root.id, kind: root.kind, path: [], children: root.children, inputs: [.init(id: "input", targets: [.init(node: "a", port: "hidden")])], outputs: root.outputs)
        XCTAssertThrowsError(try CompositionPlan(containers: [foreign]).validate(graph: graph, registry: registry))
        XCTAssertThrowsError(try CompositionPlan(schemaVersion: 2, containers: [root]).validate(graph: graph, registry: registry))
        XCTAssertThrowsError(try CompositionPlan(containers: Array(repeating: root, count: 130)).validate(graph: graph, registry: registry))
    }
    func testTypedContainerNodeIDOverlapAndPreorderAreUnambiguous() throws {
        let id = cid(["nested"])
        var graph = plainGraph()
        graph.nodes[1].id = id
        graph.edges[0].target = id; graph.edges[1].source = id
        let nested = CompositionPlan.Container(id: id, kind: .graph, path: ["nested"], children: [.init(kind: .node, id: "b", key: "0")], inputs: [.init(id: "in", targets: [.init(node: "b", port: "in")])], outputs: [.init(id: "out", source: .init(node: "b", port: "out"))])
        let root = CompositionPlan.Container(id: "root", kind: .graph, path: [], children: [.init(kind: .node, id: id, key: "0"), .init(kind: .container, id: id, key: "1")], inputs: [.init(id: "input", targets: [.init(node: id, port: "in")])], outputs: [.init(id: "output", source: .init(node: "b", port: "out"))])
        let plan = CompositionPlan(containers: [root, nested])
        try plan.validate(graph: graph, registry: .builtIn())
        XCTAssertThrowsError(try CompositionPlan(containers: [nested, root]).validate(graph: graph, registry: .builtIn()))
        XCTAssertThrowsError(try ModelProjectCodec.encode(.legacyFlat(GraphDocument(nodes: graph.nodes, edges: graph.edges, compositionPlan: plan))))
    }
    func testStrictPlanWireAndExplicitNullGraphFieldRefuse() throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let plan = rootPlan(); let good = try encoder.encode(plan)
        XCTAssertEqual(try JSONDecoder().decode(CompositionPlan.self, from: good), plan)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: good) as? [String: Any]); object["nativePath"] = "untrusted"
        XCTAssertThrowsError(try JSONDecoder().decode(CompositionPlan.self, from: JSONSerialization.data(withJSONObject: object)))
        object.removeValue(forKey: "nativePath"); object["schemaVersion"] = true
        XCTAssertThrowsError(try JSONDecoder().decode(CompositionPlan.self, from: JSONSerialization.data(withJSONObject: object)))
        var graphObject = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(plainGraph())) as? [String: Any]); graphObject["compositionPlan"] = NSNull()
        XCTAssertThrowsError(try JSONDecoder().decode(GraphDocument.self, from: JSONSerialization.data(withJSONObject: graphObject)))
    }
    func testExplicitAuthoringMigrationSourceReexpansionAndForeignProofRefuse() throws {
        let registry = try ComponentRegistry.builtIn()
        let old = ModelAuthoringDocument(projectID: UUID().uuidString, source: source(selected: false))
        let upgraded = try ModelProjectCodec.enablingComposition(old)
        XCTAssertEqual(upgraded.schemaVersion, 2); XCTAssertEqual(upgraded.source.schemaVersion, 3)
        XCTAssertEqual(upgraded.projectID, old.projectID); XCTAssertEqual(upgraded.revision, old.revision)
        var selected = upgraded; selected.source = source()
        let encoded = try ModelProjectCodec.encode(.blockModel(selected))
        XCTAssertEqual(try ModelProjectCodec.decode(encoded), .blockModel(selected))
        let snapshot = try ModelExecutionSnapshotBuilder.build(.blockModel(selected), registry: registry)
        var provenance = try XCTUnwrap(snapshot.provenance)
        XCTAssertEqual(provenance.nativeExpansionContractVersion, 2)
        XCTAssertEqual(try provenance.verifiedExpansion(registry: registry).graph, snapshot.graph)
        provenance.nodeOrigins.removeValue(forKey: snapshot.graph.nodes[1].id)
        XCTAssertThrowsError(try provenance.verifiedExpansion(registry: registry))
        var wrongPair = selected; wrongPair.schemaVersion = 1
        XCTAssertThrowsError(try ModelProjectCodec.encode(.blockModel(wrongPair)))
    }
    func testUnreachablePolicyDoesNotCreatePlanAndLegacyCanonicalBytesRemain() throws {
        let registry = try ComponentRegistry.builtIn(), old = source(selected: false)
        let original = try CompositeExpander.expand(old, registry: registry)
        var unused = old; unused.schemaVersion = 3
        var definition = old.definitions[0]; definition.reference = .init("unused"); definition.composition = .init(kind: .sequential, children: [.init(localID: "a"), .init(localID: "b")]); unused.definitions.append(definition)
        let expanded = try CompositeExpander.expand(unused, registry: registry)
        let captured = CompositeSourceSnapshot(project: unused, expansion: expanded)
        XCTAssertEqual(captured.nativeExpansionContractVersion, 2, "Source3 uses independently verified expansion2 even without a reachable native container.")
        _ = try captured.verifiedExpansion(registry: registry)
        XCTAssertNil(expanded.graph.compositionPlan)
        XCTAssertEqual(try GraphSemantics.canonicalData(expanded.graph, registry: registry), try GraphSemantics.canonicalData(original.graph, registry: registry))
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        XCTAssertEqual(try encoder.encode(expanded.graph), try encoder.encode(original.graph))
        XCTAssertNil((try JSONSerialization.jsonObject(with: encoder.encode(original.graph)) as? [String: Any])?["compositionPlan"])
    }
}
