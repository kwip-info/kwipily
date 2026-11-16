import XCTest
import KwipMLCore
@testable import KwipMLApp

final class Data02OperatorConfigurationTests:XCTestCase {
    private var registry:DataFlowOperatorRegistry { get throws {try .load(from:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("runner/dataflow-operators.json"))} }
    private func replace(_ column:String="message",output:String="cleaned",write:String="new")->[String:JSONValue] {
        let guided=FormulaGuidedRegex(kind:"replace",column:column)
        return ["operations":.array([.object(["id":.string("replace"),"expression":.string(guided.expression),"nullPolicy":.string("preserve"),"outputs":.array([.object(["name":.string(output),"dtype":.string("string"),"write":.string(write)])])])])]
    }
    func testUnknownFieldsTypesMissingColumnsAndReplacementCollisionsRefuse()throws {
        let schema=[DataColumnSchema(name:"message",dtype:"string",nullable:false)]
        let config=replace()
        XCTAssertEqual(try FormulaDraft(configuration:config).validated(schema:schema),config)
        var bad=config;bad["surprise"] = .bool(true)
        XCTAssertThrowsError(try FormulaDraft(configuration:bad))
        bad=config;bad["operations"] = .string("true")
        XCTAssertThrowsError(try FormulaDraft(configuration:bad))
        XCTAssertThrowsError(try FormulaDraft(configuration:replace("missing")).validated(schema:schema))
        XCTAssertThrowsError(try FormulaDraft(configuration:replace(output:"message")).validated(schema:schema))
        XCTAssertNoThrow(try FormulaDraft(configuration:replace(output:"message",write:"replace")).validated(schema:schema))
    }
    func testExtractOutputNamesAndFilenameAreValidatedWithoutTouchingFiles()throws {
        var guided=FormulaGuidedRegex(kind:"extract",column:"message");guided.groups=["1","2"]
        let output:JSONValue = .object(["name":.string("part"),"dtype":.string("string"),"write":.string("new")])
        let config:[String:JSONValue] = ["operations":.array([.object(["id":.string("extract"),"expression":.string(guided.expression),"nullPolicy":.string("preserve"),"outputs":.array([output,output])])])]
        XCTAssertThrowsError(try FormulaDraft(configuration:config))
        let sink=try XCTUnwrap(try registry.operatorDefinition(id:"kwip.data.outputDelimited"));var outputConfig=sink.defaultConfig
        outputConfig["filename"] = .string("../outside.csv")
        XCTAssertThrowsError(try DataEngineeringConfiguration.validate(outputConfig,definition:sink,schema:[]))
        XCTAssertEqual(try DataEngineeringConfiguration.validate(sink.defaultConfig,definition:sink,schema:[]),sink.defaultConfig)
    }
}
