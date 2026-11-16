import XCTest
import KwipMLCore
@testable import KwipMLApp
final class PreprocessSchemaTests:XCTestCase {
    func testImagePreservationTypedConstantsAndUnknownVocabulary()throws {
        let schema=[DataColumnSchema(name:"x",dtype:"int64",nullable:true),.init(name:"city",dtype:"string",nullable:true),.init(name:"pixels",dtype:"image",nullable:false)]
        var x=PreprocessConfiguration.Column(column:"x");x.missing="mean"
        var config=PreprocessConfiguration(columns:[x])
        let projected=try config.outputSchema(schema)
        XCTAssertEqual(projected[0].dtype,"float64");XCTAssertEqual(projected[0].nullable,false);XCTAssertEqual(projected[2],schema[2])
        var city=PreprocessConfiguration.Column(column:"city");city.encoding="oneHot";config.columns.append(city)
        XCTAssertEqual(try config.outputSchema(schema),[],"No invented categories before fitting.")
        config.columns=[.init(column:"pixels")];XCTAssertNoThrow(try config.validate(schema:schema),"Preserve is supported for image columns.")
        config.columns[0].whitespace="trim";XCTAssertThrowsError(try config.validate(schema:schema),"Text cleanup is refused for images.")
        x.missing="constant";x.value = .string("not integer");config = .init(mode:"table",columns:[x]);XCTAssertThrowsError(try config.validate(schema:schema))
        x.value=7;config.columns=[x];XCTAssertNoThrow(try config.validate(schema:schema))
    }
    func testSchemaUsesTrainingInputWhenAllPartitionsAreConnected()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let graph=try WorkflowDataPreparationPlan.compile(PreprocessPracticeExample.document(),registry:registry)
        let schema=[DataColumnSchema(name:"x",dtype:"float64",nullable:true)]
        XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"preprocess",in:graph,sourceSchema:{_ in schema}),schema)
    }
    func testTableCleanupEmptyToMissingProjectionAndTypeRefusals()throws {
        let schema=[DataColumnSchema(name:"name",dtype:"string",nullable:false),.init(name:"count",dtype:"int64",nullable:false),.init(name:"pixels",dtype:"image",nullable:true)]
        var rule=PreprocessConfiguration.Column(column:"name");rule.emptyAfterCleanup="missing"
        var config=PreprocessConfiguration(mode:"table",columns:[rule])
        XCTAssertEqual(try config.outputSchema(schema)[0].nullable,true)
        rule.column="count";config.columns=[rule];XCTAssertThrowsError(try config.validate(schema:schema))
        rule = .init(column:"pixels");rule.missing="drop";config.columns=[rule];XCTAssertNoThrow(try config.validate(schema:schema))
        rule.missing="constant";config.columns=[rule];XCTAssertThrowsError(try config.validate(schema:schema))
    }

}
