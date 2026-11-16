import XCTest
@testable import KwipMLCore

final class DatasetConfigurationTests:XCTestCase {
    func testFourNativeSourceModesRetainOrderedTrainingViewAndRecipe()throws {
        for kind in ["indexed","boundedIterable","tensor","subset"] {
            var values=DatasetConfiguration.defaults;values["kind"] = .string(kind)
            if kind == "subset" {values["indices"] = .array([4,1,0])}
            if kind == "boundedIterable" {values["max_rows"]=7}
            let configuration=try DatasetConfiguration(values:values)
            XCTAssertEqual(configuration.values,values)
            XCTAssertEqual(configuration.kind,kind)
            XCTAssertEqual(configuration.indices,kind == "subset" ? [4,1,0]:[])
            XCTAssertEqual(configuration.maximumRows,kind == "boundedIterable" ? 7:100000)
            XCTAssertEqual(try DatasetConfiguration.decodeRecipe(configuration.recipeValue),configuration)
        }
    }
    func testUniqueSubsetAndFiniteIterableBoundariesRefuseUnsupportedPolicies()throws {
        for (key,value) in [("kind",JSONValue.string("arbitraryGenerator")),("indices",.array([1])),("max_rows",.integer(3)),("max_rows",.bool(true)),("extra",.integer(1))] {
            var values=DatasetConfiguration.defaults;values[key]=value
            XCTAssertThrowsError(try DatasetConfiguration(values:values),key)
        }
        for indices in ([[],[1,1],[-1],[100000],[.bool(true)],[.number(1.5)]] as [[JSONValue]]) {
            var values=DatasetConfiguration.defaults;values["kind"] = .string("subset");values["indices"] = .array(indices)
            XCTAssertThrowsError(try DatasetConfiguration(values:values),indices.description)
        }
        for limit in ([0,100001,.bool(true),.number(1.5)] as [JSONValue]) {
            var values=DatasetConfiguration.defaults;values["kind"] = .string("boundedIterable");values["max_rows"]=limit
            XCTAssertThrowsError(try DatasetConfiguration(values:values))
        }
        for key in DatasetConfiguration.defaults.keys {var values=DatasetConfiguration.defaults;values.removeValue(forKey:key);XCTAssertThrowsError(try DatasetConfiguration(values:values),key)}
    }
    func testRecipeVersionsAndQualifiedControlOriginCannotBeForged()throws {
        let configuration=try DatasetConfiguration(values:DatasetConfiguration.defaults)
        XCTAssertNoThrow(try DatasetRecipe(configuration:configuration,source:.init(domain:.control,localID:"training-view")).validate())
        for source in [WorkflowNodeAddress(domain:.model,localID:"foreign"),.init(domain:.control,localID:""),.init(domain:.control,localID:String(repeating:"x",count:129))] {XCTAssertThrowsError(try DatasetRecipe(configuration:configuration,source:source).validate())}
        guard case .object(let original)=configuration.recipeValue else {return XCTFail("Recipe object")}
        for (key,value) in [("schemaVersion",JSONValue.bool(true)),("configurationVersion",.integer(2)),("componentID",.string("data_loader")),("extra",.null)] {
            var recipe=original;recipe[key]=value;XCTAssertThrowsError(try DatasetConfiguration.decodeRecipe(.object(recipe)))
        }
        for key in original.keys {var recipe=original;recipe.removeValue(forKey:key);XCTAssertThrowsError(try DatasetConfiguration.decodeRecipe(.object(recipe)))}
    }
}
