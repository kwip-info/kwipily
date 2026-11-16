import XCTest
@testable import KwipMLCore

final class ConfigurableActivationBundleTests: XCTestCase {
    func testExactDefinitionsOwnRuntimeAndAssessedLearningContracts() throws {
        let registry = try ComponentRegistry.builtIn()
        let records = [SiLUComponent.metadata, LeakyReLUComponent.metadata, ELUComponent.metadata]
        for (i, value) in records.enumerated() {
            let id = ["silu", "leaky_relu", "elu"][i]
            let entry = try XCTUnwrap(value)
            XCTAssertEqual(entry.definition, registry.component(type: id))
            XCTAssertEqual(entry.presentation.theme, "amber")
            XCTAssertEqual(entry.presentation.icon, id + "-v2")
            let data = try JSONEncoder().encode(entry.runtimeContract)
            XCTAssertNoThrow(try ConfigurableActivationRuntimeCapabilities.verify(data, componentID: id))
            XCTAssertTrue(ConfigurableActivationRuntimeCapabilities(descriptors: [id: data]).contains(id))
            XCTAssertFalse(ConfigurableActivationRuntimeCapabilities().contains(id))
            XCTAssertEqual(entry.definition.fields.map(\.key), i == 0 ? [] : [i == 1 ? "negative_slope" : "alpha"])
            var wrong = entry.runtimeContract
            if case .object(var fields) = wrong { fields["inplace"] = .bool(true); wrong = .object(fields) }
            XCTAssertThrowsError(try ConfigurableActivationRuntimeCapabilities.verify(JSONEncoder().encode(wrong), componentID: id))
        }
    }
}
