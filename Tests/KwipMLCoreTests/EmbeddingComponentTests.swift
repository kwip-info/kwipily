import XCTest
@testable import KwipMLCore

final class EmbeddingComponentTests:XCTestCase {
    func testProductionRegistryShipsReviewedLookupDefinitions() throws {
        let registry = try ComponentRegistry.builtIn()
        XCTAssertEqual(registry.component(type: "index_input"), EmbeddingComponent.indexInputDefinition)
        XCTAssertEqual(registry.component(type: "embedding"), EmbeddingComponent.definition)
    }
    func testSharedDefinitionsHaveValidDefaultsAndOptionalControls()throws {
        let definition=EmbeddingComponent.definition
        XCTAssertTrue(definition.fields.allSatisfy(\.isValidDefinition))
        let values=Dictionary(uniqueKeysWithValues:definition.fields.map {($0.key,$0.defaultValue)})
        let config=try EmbeddingConfiguration(values:values)
        XCTAssertEqual(config.numberOfEmbeddings,16)
        XCTAssertEqual(config.embeddingDimension,8)
        XCTAssertNil(config.paddingIndex);XCTAssertNil(config.maximumNorm)
        XCTAssertEqual(Set(definition.fields.filter(\.allowsNull).map(\.key)),["padding_idx","max_norm"])
        let source=EmbeddingComponent.indexInputDefinition
        XCTAssertTrue(source.fields.allSatisfy(\.isValidDefinition))
        XCTAssertEqual(try IndexInputConfiguration(values:Dictionary(uniqueKeysWithValues:source.fields.map {($0.key,$0.defaultValue)})).tensor,try TypedTensorContract(sample:[2],dtype:.int64))
        XCTAssertEqual(try JSONDecoder().decode(ComponentDefinition.self,from:JSONEncoder().encode(definition)),definition)
    }
}
