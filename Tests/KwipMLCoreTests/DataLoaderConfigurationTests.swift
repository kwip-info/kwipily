import XCTest
@testable import KwipMLCore

final class DataLoaderConfigurationTests:XCTestCase {
    func testExactDefaultsAndNonzeroWorkerControlsRoundTrip()throws {
        let defaults=try DataLoaderConfiguration(values:DataLoaderConfiguration.defaults)
        XCTAssertEqual(defaults.batchSize,32);XCTAssertFalse(defaults.shuffle);XCTAssertFalse(defaults.dropLast);XCTAssertEqual(defaults.seed,42)
        XCTAssertEqual(defaults.numberOfWorkers,0);XCTAssertNil(defaults.prefetchFactor);XCTAssertEqual(defaults.timeoutSeconds,0)
        for workers in [1,2] {
            var values=DataLoaderConfiguration.defaults
            values["batch_size"]=1024;values["shuffle"]=true;values["drop_last"]=true;values["seed"] = .integer(Int64.max)
            values["num_workers"] = .integer(Int64(workers));values["prefetch_factor"]=2;values["timeout"] = .number(30)
            let configuration=try DataLoaderConfiguration(values:values)
            XCTAssertEqual(configuration.values,values)
            XCTAssertEqual(try DataLoaderConfiguration.decodeRecipe(configuration.recipeValue),configuration)
            let decoded=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(configuration.recipeValue))
            XCTAssertEqual(try DataLoaderConfiguration.decodeRecipe(decoded),configuration,"Integral timeout tokens canonicalize through JSON")
        }
    }
    func testEveryNamedCollationAndNonnegativeSeedRangeAreExplicit()throws {
        for name in ["auto","fixed_stack","padded_sequence","whole_bags"] {
            var values=DataLoaderConfiguration.defaults;values["collation"] = .string(name);values["seed"] = .integer(Int64.max)
            XCTAssertEqual(try DataLoaderConfiguration(values:values).collation,name)
        }
    }
    func testTypeBoundsWorkerConditionsAndUnsupportedExecutionFlagsRefuse()throws {
        for (key,value) in [("batch_size",JSONValue.bool(true)),("batch_size",.integer(0)),("batch_size",.integer(1025)),("shuffle",.integer(1)),("drop_last",.string("false")),("seed",.number(42)),("seed",.integer(-1)),("num_workers",.integer(3)),("prefetch_factor",.integer(1)),("timeout",.number(1)),("collation",.string("python_callback")),("persistent_workers",.bool(true)),("pin_memory",.bool(true)),("in_order",.bool(false))] {
            var values=DataLoaderConfiguration.defaults;values[key]=value;XCTAssertThrowsError(try DataLoaderConfiguration(values:values),key)
        }
        var worker=DataLoaderConfiguration.defaults;worker["num_workers"]=1
        XCTAssertThrowsError(try DataLoaderConfiguration(values:worker))
        worker["prefetch_factor"]=1;XCTAssertThrowsError(try DataLoaderConfiguration(values:worker))
        worker["timeout"] = .number(0.5);XCTAssertNoThrow(try DataLoaderConfiguration(values:worker))
        for value in ([0,.number(-1),.number(.infinity),.number(30.01),.bool(true)] as [JSONValue]) {worker["timeout"]=value;XCTAssertThrowsError(try DataLoaderConfiguration(values:worker))}
        for key in DataLoaderConfiguration.defaults.keys {var values=DataLoaderConfiguration.defaults;values.removeValue(forKey:key);XCTAssertThrowsError(try DataLoaderConfiguration(values:values),key)}
    }
    func testQualifiedRecipeAndExactEnvelopeRefuseForeignSourceOrVersions()throws {
        let configuration=try DataLoaderConfiguration(values:DataLoaderConfiguration.defaults)
        XCTAssertNoThrow(try DataLoaderRecipe(configuration:configuration,source:.init(domain:.control,localID:"batches")).validate())
        XCTAssertThrowsError(try DataLoaderRecipe(configuration:configuration,source:.init(domain:.data,localID:"foreign")).validate())
        guard case .object(let original)=configuration.recipeValue else {return XCTFail("Recipe object")}
        for (key,value) in [("schemaVersion",JSONValue.bool(true)),("configurationVersion",.integer(2)),("componentID",.string("dataset_source")),("extra",.null)] {
            var recipe=original;recipe[key]=value;XCTAssertThrowsError(try DataLoaderConfiguration.decodeRecipe(.object(recipe)))
        }
        var recipe=original;recipe["config"] = .object([:]);XCTAssertThrowsError(try DataLoaderConfiguration.decodeRecipe(.object(recipe)))
    }
}
