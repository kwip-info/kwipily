import XCTest
@testable import KwipMLCore

final class AdaptivePoolingConfigurationTests: XCTestCase {
    func testOptionalSignedFieldUsesDeclaredEnablementValueAndRoundTrips() throws {
        let data = Data("{\"key\":\"divisor\",\"title\":\"Divisor\",\"kind\":\"integer\",\"defaultValue\":null,\"min\":-10,\"max\":10,\"allowsNull\":true,\"enabledValue\":1}".utf8)
        let field = try JSONDecoder().decode(ComponentField.self, from: data)
        XCTAssertTrue(field.isValidDefinition)
        XCTAssertEqual(field.valueWhenEnabled, 1)
        let encoded = try JSONEncoder().encode(field)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoded) as? [String:Any])
        XCTAssertEqual(object["enabledValue"] as? Int, 1)
    }
    func testEnablementDeclarationRefusesNullWrongTypeAndOutOfRange() throws {
        let base = ComponentField(key: "divisor", title: "Divisor", kind: .integer, defaultValue: .null, min: -10, max: 10, allowsNull: true)
        for invalid: JSONValue in [.null, true, .number(1.5), 11] {
            var field = base
            field.enabledValue = invalid
            XCTAssertFalse(field.isValidDefinition)
            XCTAssertEqual(field.valueWhenEnabled, .null)
        }
        var required = base
        required.allowsNull = false
        required.defaultValue = 2
        required.enabledValue = 1
        XCTAssertFalse(required.isValidDefinition)
        let legacy = ComponentField(key: "stride", title: "Stride", kind: .integer, defaultValue: .null, min: 1, max: 4096, allowsNull: true)
        XCTAssertEqual(legacy.valueWhenEnabled, 1)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(legacy)) as? [String:Any])
        XCTAssertNil(object["enabledValue"])
    }
    func testAdaptiveAxesPreserveChannelsAndResolveIndependentNulls() throws {
        let c = try AdaptivePoolingConfiguration(componentID: "adaptive_avg_pool2d", values: ["output_height": .null, "output_width": 5])
        XCTAssertEqual(try c.resolved(input: .init(sample: [2,3,4])).outputLayout.sample.dimensions, [2,3,5])
        XCTAssertEqual(c.values, ["output_height": .null, "output_width": 5])
        XCTAssertNil(try c.resolved(input: .init(sample: [2,3,4])).indicesLayout)
    }
    func testAdaptiveMaxOptionalIndicesAndOneDimensionalOutput() throws {
        let max = try AdaptivePoolingConfiguration(componentID: "adaptive_max_pool2d", values: ["output_height": 2, "output_width": .null, "return_indices": true])
        let result = try max.resolved(input: .init(sample: [3,5,7]))
        XCTAssertEqual(result.outputLayout.sample.dimensions, [3,2,7])
        XCTAssertEqual(result.indicesLayout, result.outputLayout)
        let average = try AdaptivePoolingConfiguration(componentID: "adaptive_avg_pool1d", values: ["output_size": 7])
        XCTAssertEqual(try average.resolved(input: .init(sample: [2,3])).outputLayout.sample.dimensions, [2,7])
    }
    func testStrictIdentityFieldsBoundsAndAllocationBudget() throws {
        for values: [String:JSONValue] in [["output_size": .null], ["output_size": true], ["output_size": 0], ["output_size": 4097], ["output_size": 2, "return_indices": false]] {
            XCTAssertThrowsError(try AdaptivePoolingConfiguration(componentID: "adaptive_avg_pool1d", values: values))
        }
        XCTAssertThrowsError(try AdaptivePoolingConfiguration(componentID: "adaptive_max_pool1d", values: ["output_size": 2]))
        let c = try AdaptivePoolingConfiguration(componentID: "adaptive_avg_pool2d", values: ["output_height": 4096, "output_width": 4096])
        XCTAssertThrowsError(try c.resolved(input: .init(sample: [512,2,2])))
        for shape in [[2,3], [513,2,2], [2,4097,2]] {
            XCTAssertThrowsError(try c.resolved(input: .init(sample: shape)))
        }
    }
    func testAverageTwoDimensionalGeometryAndSignedDivisor() throws {
        let values: [String:JSONValue] = ["kernel_size": .array([3,2]), "stride": .null, "padding": .array([1,0]), "ceil_mode": true, "count_include_pad": false, "divisor_override": -2]
        let c = try AvgPool2dConfiguration(values: values)
        XCTAssertEqual(c.divisorOverride, -2)
        XCTAssertEqual(c.values, values)
        let r = try c.resolved(input: .init(sample: [2,5,7]))
        XCTAssertEqual(r.outputLayout.sample.dimensions, [2,2,4])
        XCTAssertNil(r.indicesLayout)
        for invalid: JSONValue in [0, true, .number(1.5), 2147483648] {
            XCTAssertThrowsError(try AvgPool2dConfiguration(values: values.merging(["divisor_override": invalid]) { _,new in new }))
        }
    }
}
