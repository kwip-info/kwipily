import XCTest
@testable import KwipMLCore

final class CompositeIntegerProductTests: XCTestCase {
    private func fixture() -> CompositeDefinition {
        .init(reference: .init("product-block"), title: "Product block",
            body: .init(nodes: [.init(id: "first", type: "linear", config: ["in_features": 2, "out_features": 3, "bias": true]),
                                .init(id: "second", type: "linear", config: ["in_features": 6, "out_features": 2, "bias": true])]),
            inputs: [], outputs: [], parameters: [
                .init(.init(key: "channels", title: "Channels", defaultValue: 2), targets: [.init("first", "in_features")]),
                .init(.init(key: "multiplier", title: "Multiplier", defaultValue: 3), targets: [.init("first", "out_features")])],
            integerProducts: [.init(lhs: "channels", rhs: "multiplier", targets: [.init("second", "in_features")])])
    }
    func testDefaultsOverridesAndOriginalDefinitionRemainIndependent() throws {
        let definition = fixture(), registry = try ComponentRegistry.builtIn()
        let defaults = try CompositeConfigurationResolver.resolve(definition, definitions: [], registry: registry)
        XCTAssertEqual(defaults.nodes[1].config["in_features"], 6)
        let overridden = try CompositeConfigurationResolver.resolve(definition, overrides: ["channels": 4, "multiplier": 5], definitions: [], registry: registry)
        XCTAssertEqual(overridden.nodes[0].config["in_features"], 4)
        XCTAssertEqual(overridden.nodes[0].config["out_features"], 5)
        XCTAssertEqual(overridden.nodes[1].config["in_features"], 20)
        XCTAssertEqual(definition, fixture())
        XCTAssertEqual(try JSONDecoder().decode(CompositeDefinition.self, from: JSONEncoder().encode(definition)), definition)
    }
    func testOldDefinitionEncodingOmitsOptionalBindings() throws {
        var definition = fixture(); definition.integerProducts = nil
        let data = try JSONEncoder().encode(definition)
        XCTAssertNil(try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])["integerProducts"])
        let decoded = try JSONDecoder().decode(CompositeDefinition.self, from: data)
        XCTAssertNil(decoded.integerProducts)
        XCTAssertEqual(try CompositeConfigurationResolver.resolve(decoded, definitions: [], registry: .builtIn()), decoded.body)
    }
    func testInvalidOperandsTargetsAndCollisionsRefuseWithoutMutation() throws {
        let changes: [(inout CompositeDefinition) -> Void] = [
            { $0.integerProducts![0].lhs = "missing" },
            { $0.parameters[0].field.allowsNull = true },
            { $0.parameters[0].field.kind = .number },
            { $0.integerProducts![0].targets = [.init("absent", "in_features")] },
            { $0.integerProducts![0].targets = [.init("second", "bias")] },
            { $0.integerProducts![0].targets = [.init("first", "in_features")] },
            { $0.integerProducts!.append($0.integerProducts![0]) },
            { $0.integerProducts![0].targets.append($0.integerProducts![0].targets[0]) },
            { $0.integerProducts![0].targets = [] },
            { $0.integerProducts = Array(repeating: $0.integerProducts![0], count: 129) },
            { $0.integerProducts![0].targets = Array(repeating: .init("second", "in_features"), count: 257) }
        ]
        for change in changes {
            var definition = fixture(); change(&definition); let snapshot = definition
            XCTAssertThrowsError(try CompositeConfigurationResolver.resolve(definition, definitions: [], registry: .builtIn()))
            XCTAssertEqual(definition, snapshot)
        }
        XCTAssertThrowsError(try CompositeConfigurationResolver.resolve(fixture(), overrides: ["channels": 512, "multiplier": 2], definitions: [], registry: .builtIn()))
        XCTAssertThrowsError(try CompositeConfigurationResolver.resolve(fixture(), overrides: ["unknown": 2], definitions: [], registry: .builtIn()))
    }
    func testZeroNegativeAndOverflowUseTargetIntegerContract() throws {
        var registry = try ComponentRegistry.builtIn()
        let index = try XCTUnwrap(registry.components.firstIndex { $0.id == "linear" })
        for field in registry.components[index].fields.indices where registry.components[index].fields[field].kind == .integer {
            registry.components[index].fields[field].min = -10
            registry.components[index].fields[field].max = 5_000_000_000
        }
        var definition = fixture()
        for index in definition.parameters.indices {
            definition.parameters[index].field.min = -10
            definition.parameters[index].field.max = 5_000_000_000
        }
        for (lhs, expected) in [(0, 0), (-2, -6)] {
            let resolved = try CompositeConfigurationResolver.resolve(definition, overrides: ["channels": .integer(Int64(lhs))], definitions: [], registry: registry)
            XCTAssertEqual(resolved.nodes[1].config["in_features"], .integer(Int64(expected)))
        }
        XCTAssertThrowsError(try CompositeConfigurationResolver.resolve(definition, overrides: ["channels": 4_000_000_000, "multiplier": 4_000_000_000], definitions: [], registry: registry))
    }
    func testNestedExposedIntegerTargetReceivesDerivedOverride() throws {
        let nested = CompositeDefinition(reference: .init("nested-product-target"), title: "Nested", body: .init(nodes: [.init(id: "internal", type: "linear", config: ["in_features": 6, "out_features": 2, "bias": true])]), inputs: [], outputs: [],
            parameters: [.init(.init(key: "width", title: "Width", defaultValue: 6), targets: [.init("internal", "in_features")])])
        var definition = fixture()
        definition.body.nodes.removeLast()
        definition.body.instances = [.init(id: "nested", definition: nested.reference)]
        definition.integerProducts![0].targets = [.init("nested", "width")]
        let result = try CompositeConfigurationResolver.resolve(definition, overrides: ["channels": 4], definitions: [nested], registry: .builtIn())
        XCTAssertEqual(result.instances[0].overrides["width"], 12)
        XCTAssertTrue(definition.body.instances[0].overrides.isEmpty)
    }
    func testWorkflowCodecRetainsProductsAndFullExposedFieldKinds() throws {
        var definition = fixture()
        definition.parameters += [
            .init(.init(key: "shape", title: "Shape", kind: .integerList, defaultValue: .array([1, 2]), min: 1, max: 512, minItems: 2, maxItems: 2), targets: [.init("first", "shape")]),
            .init(.init(key: "mode", title: "Mode", kind: .choice, defaultValue: .string("same"), min: nil, max: nil,
                choices: [.init(value: "same", title: "Same")]), targets: [.init("first", "mode")]),
            .init(.init(key: "optional", title: "Optional", defaultValue: .null, min: 0, max: 512, allowsNull: true), targets: [.init("first", "optional")])]
        let document = WorkflowDocument(projectID: UUID().uuidString, definitions: [definition])
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: WorkflowCodec.encode(document)) as? [String: Any])
        var definitions = try XCTUnwrap(raw["definitions"] as? [[String: Any]])
        var products = try XCTUnwrap(definitions[0]["integerProducts"] as? [[String: Any]])
        products[0]["code"] = "channels * multiplier"
        definitions[0]["integerProducts"] = products; raw["definitions"] = definitions
        XCTAssertThrowsError(try WorkflowCodec.decode(JSONSerialization.data(withJSONObject: raw)))
    }

}
