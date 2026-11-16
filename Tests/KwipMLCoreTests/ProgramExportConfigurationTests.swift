import XCTest
@testable import KwipMLCore

final class ProgramExportConfigurationTests:XCTestCase {
    func testExactSevenControlsAndStaticBoundariesRoundTrip()throws {
        let configuration=try ProgramExportConfiguration(values:ProgramExportConfiguration.defaults)
        XCTAssertEqual(configuration.values,ProgramExportConfiguration.defaults)
        XCTAssertEqual(try ProgramExportConfiguration.decodeRecipe(configuration.recipeValue),configuration)
        for size in [1,128] {
            var values=ProgramExportConfiguration.defaults
            for key in ["example_batch_size","minimum_batch_size","maximum_batch_size"] {values[key] = .integer(Int64(size))}
            XCTAssertEqual(try ProgramExportConfiguration(values:values).values,values)
        }
    }
    func testBoundedBatchKeepsSampleAxesAndExplicitDeviceControls()throws {
        for device in ["cpu","mps"] {
            var values=ProgramExportConfiguration.defaults
            values["batch_mode"] = .string("bounded");values["minimum_batch_size"]=1;values["maximum_batch_size"]=4096
            values["device"] = .string(device);values["strict"]=false;values["prefer_deferred_runtime_asserts"]=true
            XCTAssertEqual(try ProgramExportConfiguration(values:values).values,values)
        }
    }
    func testMissingUnknownWrongTypesAndContradictoryBoundsRefuse()throws {
        for key in ProgramExportConfiguration.defaults.keys {
            var missing=ProgramExportConfiguration.defaults;missing.removeValue(forKey:key)
            XCTAssertThrowsError(try ProgramExportConfiguration(values:missing),key)
            var invalid=ProgramExportConfiguration.defaults;invalid[key] = .null
            XCTAssertThrowsError(try ProgramExportConfiguration(values:invalid),key)
        }
        for (key,value) in [("example_batch_size",JSONValue.bool(true)),("minimum_batch_size",.number(1.5)),("maximum_batch_size",.integer(4097)),("strict",.integer(1)),("device",.string("cuda")),("batch_mode",.string("dynamic_samples")),("prefer_deferred_runtime_asserts",.string("false")),("unknown",.integer(1)),("example_batch_size",.integer(129))] {
            var values=ProgramExportConfiguration.defaults;values[key]=value
            XCTAssertThrowsError(try ProgramExportConfiguration(values:values),key)
        }
        var staticMismatch=ProgramExportConfiguration.defaults;staticMismatch["maximum_batch_size"]=3
        XCTAssertThrowsError(try ProgramExportConfiguration(values:staticMismatch))
        var bounded=ProgramExportConfiguration.defaults;bounded["batch_mode"] = .string("bounded")
        XCTAssertThrowsError(try ProgramExportConfiguration(values:bounded),"A range needs distinct bounds")
        bounded["minimum_batch_size"]=1;bounded["maximum_batch_size"]=3;bounded["example_batch_size"]=1
        XCTAssertThrowsError(try ProgramExportConfiguration(values:bounded),"Native bounded export needs nonspecialized example size at least2")
        bounded["example_batch_size"]=4
        XCTAssertThrowsError(try ProgramExportConfiguration(values:bounded))
    }
    func testRecipeEnvelopeCannotAcquireUnknownVersionOrOmittedControls()throws {
        let valid=try ProgramExportConfiguration(values:ProgramExportConfiguration.defaults)
        guard case .object(let original)=valid.recipeValue else {return XCTFail("Recipe object")}
        for key in original.keys {
            var value=original;value.removeValue(forKey:key)
            XCTAssertThrowsError(try ProgramExportConfiguration.decodeRecipe(.object(value)),key)
        }
        for (key,value) in [("schemaVersion",JSONValue.bool(true)),("configurationVersion",.integer(2)),("componentID",.string("checkpoint_lifecycle")),("extra",.bool(true))] {
            var next=original;next[key]=value
            XCTAssertThrowsError(try ProgramExportConfiguration.decodeRecipe(.object(next)),key)
        }
        var next=original;next["config"] = .object([:])
        XCTAssertThrowsError(try ProgramExportConfiguration.decodeRecipe(.object(next)))
    }
}
