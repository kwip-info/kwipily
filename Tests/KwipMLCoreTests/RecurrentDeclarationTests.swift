import XCTest
@testable import KwipMLCore

final class RecurrentDeclarationTests: XCTestCase {
    func testSixNamedStateDeclarationsRemainObservationFirstAndCarryIsExplicit() throws {
        let factories: [(String, (ComponentDefinition) throws -> CapabilityDescriptor)] = [
            ("rnn", RecurrentCapabilityBridge.rnnDeclaration), ("lstm", RecurrentCapabilityBridge.lstmDeclaration),
            ("gru", RecurrentCapabilityBridge.gruDeclaration), ("rnn_cell", RecurrentCapabilityBridge.rnnCellDeclaration),
            ("lstm_cell", RecurrentCapabilityBridge.lstmCellDeclaration), ("gru_cell", RecurrentCapabilityBridge.gruCellDeclaration)]
        var lessons = Set<String>()
        for (id, factory) in factories {
            let definition = BundledRecurrentComponents.definition(id)
            let capability = try factory(definition)
            XCTAssertTrue(CapabilityContractValidator.validate(capability).isValid)
            XCTAssertEqual(capability.ports.filter { $0.direction == .input }.map(\.id), definition.inputs)
            XCTAssertEqual(capability.ports.filter { $0.direction == .output }.map(\.id), definition.outputs)
            XCTAssertEqual(capability.fields, definition.fields)
            XCTAssertEqual(capability.lifecycle.resume, .weightsOnly)
            XCTAssertEqual(capability.devices, [.cpu, .mps])
            XCTAssertFalse(capability.lifecycle.state.contains { $0.kind == .recurrent }, "Tensor connections, never mutable implicit carry")
            XCTAssertEqual(capability.lifecycle.state.map(\.kind), id.hasSuffix("_cell") ? [.parameters] : [.parameters, .randomGenerator])
            XCTAssertEqual(capability.ports.filter { $0.direction == .input && $0.required }.map(\.id), ["in"])
            for port in capability.ports {
                XCTAssertEqual(port.schema.tensor?.dtype, .float32)
                XCTAssertEqual(port.schema.tensor?.leadingAxes, TensorLeadingAxes(minimum: 1, maximum: 1))
                XCTAssertEqual(port.schema.tensor?.dimensions.count, id.hasSuffix("_cell") ? 1 : 2)
            }
            XCTAssertEqual(BundledRecurrentComponents.entries[id]?.inputConditions, Dictionary(uniqueKeysWithValues: definition.inputs.filter { $0 != "in" }.map { ($0, "state_mode=supplied") }))
            XCTAssertEqual(capability.learning.lessonIDs.count, 1); lessons.formUnion(capability.learning.lessonIDs)
            var altered = definition; altered.outputs.removeAll { $0 == "hidden" }
            XCTAssertThrowsError(try factory(altered))
        }
        XCTAssertEqual(lessons.count, 6)
    }
    func testExactOptionalDiscoveryRefusesForeignStateAndProjectionPolicies() throws {
        XCTAssertTrue(RecurrentRuntimeCapabilities().componentIDs.isEmpty)
        var descriptors: [String: Data] = [:]
        for id in RecurrentRuntimeCapabilities.operations { descriptors[id] = try JSONEncoder().encode(BundledRecurrentComponents.entries[id]!.runtimeContract) }
        XCTAssertEqual(RecurrentRuntimeCapabilities(descriptors: descriptors).componentIDs, RecurrentRuntimeCapabilities.operations)
        for field in ["statePersistence", "projectionPolicy", "nativeStateLayout", "maximumParameters"] {
            let id = "lstm"
            var object = try XCTUnwrap(JSONSerialization.jsonObject(with: descriptors[id]!) as? [String: Any])
            if field == "maximumParameters" { object[field] = 2_000_001 } else { object[field] = "unreviewed" }
            var changed = descriptors; changed[id] = try JSONSerialization.data(withJSONObject: object)
            XCTAssertFalse(RecurrentRuntimeCapabilities(descriptors: changed).contains(id))
            XCTAssertThrowsError(try RecurrentRuntimeCapabilities.verify(changed[id]!, componentID: id))
        }
        XCTAssertThrowsError(try RecurrentRuntimeCapabilities.verify(Data(repeating: 32, count: 4097), componentID: "lstm"))
        XCTAssertThrowsError(try RecurrentRuntimeCapabilities.verify(descriptors["rnn"]!, componentID: "attention"))
    }
}
