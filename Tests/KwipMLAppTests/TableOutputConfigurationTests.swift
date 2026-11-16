import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TableOutputConfigurationTests:XCTestCase {
    private func definition()throws->DataFlowOperatorDefinition {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try XCTUnwrap(DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")).operatorDefinition(id:"kwip.data.outputDelimited"))
    }
    func testLegacyConfigAndEveryFormatInventory()throws {
        let definition=try definition()
        let legacy:[String:JSONValue]=["filename":.string("old.csv"),"separator":.string(";"),"columns":.array([]),"overwrite":.bool(false)]
        XCTAssertEqual(try TableOutputConfiguration.validate(legacy,definition:definition,schema:[]),legacy)
        for (format,_) in TableOutputConfiguration.formats {
            var config=legacy;config["format"] = .string(format);config["options"] = .object(TableOutputConfiguration.defaults(format))
            XCTAssertNoThrow(try TableOutputConfiguration.validate(config,definition:definition,schema:[]),format)
        }
        XCTAssertFalse(TableOutputConfiguration.fields("ndjson").contains {$0.id == "pretty" || $0.id == "shape"})
    }
    func testRejectUnsafeOrContradictoryOptions()throws {
        let definition=try definition()
        var config=definition.defaultConfig
        config["filename"] = .string("../private.csv")
        XCTAssertThrowsError(try TableOutputConfiguration.validate(config,definition:definition,schema:[]))
        config["filename"] = .string("safe.csv");config["format"] = .string("csv");config["options"] = .object(["quoting":.string("none"),"escaping":.string("none")])
        XCTAssertThrowsError(try TableOutputConfiguration.validate(config,definition:definition,schema:[]))
        config["format"] = .string("parquet");config["options"] = .object(["compression":.string("snappy"),"compressionLevel":.integer(3)])
        XCTAssertThrowsError(try TableOutputConfiguration.validate(config,definition:definition,schema:[]))
        config["options"] = .object([:]);config["imagePolicy"] = .string("copyAssets");config["overwrite"] = .bool(true)
        XCTAssertThrowsError(try TableOutputConfiguration.validate(config,definition:definition,schema:[]))
    }
    func testConditionalFieldsFollowChosenBehavior() {
        let csv=TableOutputConfiguration.fields("csv")
        let defaults=TableOutputConfiguration.defaults("csv")
        XCTAssertFalse(TableOutputConfiguration.visible(csv.first {$0.id == "escapeChar"}!,format:"csv",options:defaults))
        XCTAssertFalse(TableOutputConfiguration.visible(csv.first {$0.id == "numericPattern"}!,format:"csv",options:defaults))
        var escaping=defaults;escaping["escaping"] = .string("escape")
        XCTAssertTrue(TableOutputConfiguration.visible(csv.first {$0.id == "escapeChar"}!,format:"csv",options:escaping))
    }
}
