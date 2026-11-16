import XCTest
@testable import KwipMLCore

final class ModuleCompositionEditingTests: XCTestCase {
    private let ports: [String: CompositionPolicy.ChildPorts] = ["a": .init(inputs: ["in"], outputs: ["out"]), "b": .init(inputs: ["in"], outputs: ["out"]), "c": .init(inputs: ["in"], outputs: ["out"])]
    private func fixture(_ kind: CompositionPolicy.Kind? = nil) -> CompositeDefinition {
        .init(reference: .init("editable-block"), title: "My named design",
            body: .init(nodes: ["c", "a", "b"].enumerated().map { .init(id: $0.element, type: "relu", name: "Display " + $0.element, x: Double($0.offset * 90), y: 17) },
                edges: [.init(id: "wire-b-c", source: "b", target: "c"), .init(id: "wire-a-b", source: "a", target: "b")]),
            inputs: [.init("observations", targets: [.init("a", "in")])], outputs: [.init("answers", source: .init("c", "out"))],
            composition: kind.map { .init(kind: $0, children: ["a", "b", "c"].map { .init(localID: $0, key: kind == .namedMap ? "key_" + $0 : nil) }) })
    }
    func testSelectingSequentialDerivesWireChainNotBodyArrayOrVisualOrder() throws {
        let original = fixture(); var checked = false
        let edited = try ModuleCompositionEditing.selecting(kind: .sequential, in: original, childPorts: ports) { _ in checked = true }
        XCTAssertTrue(checked); XCTAssertNil(original.composition)
        XCTAssertEqual(edited.composition?.kind, .sequential); XCTAssertEqual(edited.composition?.children.map(\.localID), ["a", "b", "c"])
        XCTAssertEqual(edited.body, original.body); XCTAssertEqual(edited.inputs, original.inputs); XCTAssertEqual(edited.outputs, original.outputs)
        XCTAssertEqual(edited.title, original.title); XCTAssertEqual(edited.parameters, original.parameters)
        var broken = original; broken.body.edges.removeLast()
        XCTAssertThrowsError(try ModuleCompositionEditing.selecting(kind: .sequential, in: broken, childPorts: ports, validate: { _ in }))
    }
    func testSequentialMoveAtomicallyRewiresUsingOldChainPositionWireIdentities() throws {
        let original = fixture(.sequential)
        let edited = try ModuleCompositionEditing.movingChild(localID: "c", to: 0, in: original, childPorts: ports, validate: { _ in })
        XCTAssertEqual(edited.composition?.children.map(\.localID), ["c", "a", "b"])
        let wireAB = try XCTUnwrap(edited.body.edges.first { $0.id == "wire-a-b" })
        let wireBC = try XCTUnwrap(edited.body.edges.first { $0.id == "wire-b-c" })
        XCTAssertEqual(wireAB.source, "c"); XCTAssertEqual(wireAB.target, "a")
        XCTAssertEqual(wireBC.source, "a"); XCTAssertEqual(wireBC.target, "b")
        XCTAssertEqual(edited.inputs[0].targets, [.init("c", "in")]); XCTAssertEqual(edited.outputs[0].source, .init("b", "out"))
        XCTAssertEqual(edited.body.nodes, original.body.nodes); XCTAssertEqual(edited.body.edges.map(\.id), original.body.edges.map(\.id))
        XCTAssertEqual(original.inputs[0].targets, [.init("a", "in")])
        XCTAssertThrowsError(try ModuleCompositionEditing.movingChild(localID: "c", to: 0, in: original, childPorts: ports) { _ in throw GraphConfigurationError("Actual expanded shape is incompatible") })
    }
    func testGraphListAndMapMoveStorageOnlyAndNamedRenameKeepsLocalInvocation() throws {
        for kind in [CompositionPolicy.Kind.graph, .orderedList, .namedMap] {
            let original = fixture(kind)
            let edited = try ModuleCompositionEditing.movingChild(localID: "c", to: 0, in: original, childPorts: ports, validate: { _ in })
            XCTAssertEqual(edited.composition?.children.map(\.localID), ["c", "a", "b"])
            XCTAssertEqual(edited.body, original.body); XCTAssertEqual(edited.inputs, original.inputs); XCTAssertEqual(edited.outputs, original.outputs)
        }
        let original = fixture(.namedMap)
        let renamed = try ModuleCompositionEditing.renamingKey(localID: "a", key: "prepare", in: original, childPorts: ports, validate: { _ in })
        XCTAssertEqual(renamed.composition?.children.first?.key, "prepare")
        XCTAssertEqual(renamed.body, original.body); XCTAssertEqual(renamed.inputs, original.inputs); XCTAssertEqual(renamed.outputs, original.outputs)
        for key in ["key_b", "forward", "bad.key", "", "1start"] {
            XCTAssertThrowsError(try ModuleCompositionEditing.renamingKey(localID: "a", key: key, in: original, childPorts: ports, validate: { _ in }))
        }
    }
    func testMembershipRefreshIsExplicitPreservesSurvivorOrderAndKeysThenAppends() throws {
        var stale = fixture(.namedMap)
        stale.body.nodes.removeAll { $0.id == "b" }; stale.body.nodes.append(.init(id: "new child", type: "relu"))
        stale.body.edges = []; stale.inputs = [.init("observations", targets: [.init("a", "in")])]
        let actual = ["a": ports["a"]!, "c": ports["c"]!, "new child": CompositionPolicy.ChildPorts(inputs: ["in"], outputs: ["out"])]
        XCTAssertThrowsError(try ModuleCompositionEditing.movingChild(localID: "a", to: 1, in: stale, childPorts: actual, validate: { _ in }))
        let refreshed = try ModuleCompositionEditing.refreshingMembership(in: stale, childPorts: actual, validate: { _ in })
        XCTAssertEqual(refreshed.composition?.children.map(\.localID), ["a", "c", "new child"])
        XCTAssertEqual(refreshed.composition?.children.prefix(2).map(\.key), ["key_a", "key_c"])
        let key = try XCTUnwrap(refreshed.composition?.children.last?.key)
        XCTAssertFalse(key.isEmpty); XCTAssertFalse(key.contains(" "))
        XCTAssertEqual(refreshed.body, stale.body); XCTAssertEqual(stale.composition?.children.map(\.localID), ["a", "b", "c"])
    }
    func testUnknownChildInvalidIndexAndNonUnarySequentialRefuseWithoutMutating() throws {
        let original = fixture(.orderedList)
        XCTAssertThrowsError(try ModuleCompositionEditing.movingChild(localID: "missing", to: 0, in: original, childPorts: ports, validate: { _ in }))
        XCTAssertThrowsError(try ModuleCompositionEditing.movingChild(localID: "a", to: 3, in: original, childPorts: ports, validate: { _ in }))
        XCTAssertThrowsError(try ModuleCompositionEditing.renamingKey(localID: "a", key: "start", in: original, childPorts: ports, validate: { _ in }))
        var multi = ports; multi["b"] = .init(inputs: ["left", "right"], outputs: ["out"])
        XCTAssertThrowsError(try ModuleCompositionEditing.selecting(kind: .sequential, in: original, childPorts: multi, validate: { _ in }))
        XCTAssertEqual(original, fixture(.orderedList))
    }
}
