import XCTest
@testable import KwipMLCore
final class APITableConfigurationSchemaTests:XCTestCase {
    func testSchemasCoverNativeFieldsAndBoundedChoicesAndBothExamplesValidate()throws {
        for direction in ["input","output"] {
            guard case .object(let schema)=APITableSettings.configurationSchema(direction:direction),case .object(let properties)=schema["properties"] else {return XCTFail("Missing properties")}
            XCTAssertEqual(Set(properties.keys),Set(APITableSettings.defaults.keys))
            for (key,low,high) in APITableSettings.limits {
                guard case .object(let field)=properties[key] else {return XCTFail("Missing limit")}
                XCTAssertEqual(field["minimum"],.integer(Int64(low)));XCTAssertEqual(field["maximum"],.integer(Int64(high)))
            }
            guard case .object(let variant)=APITableSettings.discoveryVariant(direction:direction),case .object(let example)=variant["exampleConfiguration"],case .object(let api)=example[direction=="input" ? "options":"api"] else {return XCTFail("Missing example")}
            XCTAssertNoThrow(try APITableSettings.normalized(api))
            if direction=="input" {XCTAssertNoThrow(try TableInputConfiguration.validate(example));XCTAssertEqual(Set(example.keys),["format","options"])}
            XCTAssertNil(properties["password"]);XCTAssertNil(properties["token"]);XCTAssertNil(properties["profileID"])
            for mode in ["none","cursor","page","offset","nextLink"] {var value=api;value["pagination"] = .string(mode);XCTAssertNoThrow(try APITableSettings.normalized(value))}
            var invalid=api;invalid["mapping"] = .array([])
            XCTAssertThrowsError(try APITableSettings.normalized(invalid)){error in XCTAssertEqual((error as? DataFlowContractError)?.apiIssue,.mapping)}
            invalid=api;invalid["password"] = .string("PRIVATE-CREDENTIAL")
            XCTAssertThrowsError(try APITableSettings.normalized(invalid)){error in XCTAssertEqual((error as? DataFlowContractError)?.apiIssue,.fields);XCTAssertFalse(error.localizedDescription.contains("PRIVATE-CREDENTIAL"))}
        }
    }
}
