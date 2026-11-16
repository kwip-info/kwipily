import XCTest
import KwipMLCore
@testable import KwipMLApp

final class AggregateConfigurationTests:XCTestCase {
    let schema:[DataColumnSchema]=[.init(name:" entity, ",dtype:"string",nullable:true),.init(name:"amount",dtype:"int64",nullable:true),.init(name:"flag",dtype:"bool",nullable:false)]
    var config:[String:JSONValue] {["groupBy":.array([.string(" entity, ")]),"nullKeys":.string("group"),"aggregations":.array([.object(["output":.string("My total"),"op":.string("sum"),"column":.string("amount"),"nullPolicy":.string("skip")])]),"maxRows":.integer(100)]}
    func testExactLabelsAndDeclaredTypes()throws {
        XCTAssertEqual(try AggregateTableConfiguration.validate(config,schema:schema),config)
        let output=try AggregateTableConfiguration.outputSchema(config,schema:schema)
        XCTAssertEqual(output.map(\.name),[" entity, ","My total"])
        XCTAssertEqual(output.map(\.dtype),["string","int64"])
        XCTAssertEqual(output.map(\.nullable),[true,true])
        for policy in ["drop","error"] {
            var next=config;next["nullKeys"] = .string(policy)
            XCTAssertEqual(try AggregateTableConfiguration.outputSchema(next,schema:schema).first?.nullable,false)
        }
    }
    func testReducerShapesTypesAndNullability()throws {
        for op in ["countRows","countNonNull","sum","mean","min","max"] {
            var reducer:[String:JSONValue] = ["output":.string("Chosen label"),"op":.string(op)]
            if op != "countRows" {reducer["column"] = .string("amount")}
            if !["countRows","countNonNull"].contains(op) {reducer["nullPolicy"] = .string("error")}
            var next=config;next["groupBy"] = .array([]);next["aggregations"] = .array([.object(reducer)])
            let output=try AggregateTableConfiguration.outputSchema(next,schema:schema)
            XCTAssertEqual(output.count,1)
            XCTAssertEqual(output[0].dtype,op == "mean" ? "float64":"int64")
            XCTAssertEqual(output[0].nullable,false)
            reducer["unexpected"] = .string("x");next["aggregations"] = .array([.object(reducer)])
            XCTAssertThrowsError(try AggregateTableConfiguration.validate(next,schema:schema),op)
        }
    }
    func testInvalidDraftsDoNotInventLabelsOrColumns()throws {
        for (key,value) in [("groupBy",JSONValue.array([.string("amount"),.string("amount")])),("groupBy",.array([.string(" ")])),("nullKeys",.string("guess")),("aggregations",.array([])),("maxRows",.integer(0)),("maxRows",.number(2.5))] {
            var next=config;next[key]=value
            XCTAssertThrowsError(try AggregateTableConfiguration.validate(next,schema:schema),key)
        }
        for reducer:[String:JSONValue] in [
            ["output":.string(" "),"op":.string("countRows")],
            ["output":.string(" entity, "),"op":.string("countRows")],
            ["output":.string("X"),"op":.string("sum"),"column":.string("flag"),"nullPolicy":.string("skip")],
            ["output":.string("X"),"op":.string("countRows"),"column":.string("amount")],
            ["output":.string("X"),"op":.string("countNonNull"),"column":.string("amount"),"nullPolicy":.string("skip")],
            ["output":.string("X"),"op":.string("max"),"column":.string("missing"),"nullPolicy":.string("skip")]
        ] {
            var next=config;next["aggregations"] = .array([.object(reducer)])
            XCTAssertThrowsError(try AggregateTableConfiguration.validate(next,schema:schema))
        }
        XCTAssertEqual(try AggregateTableConfiguration.validate(config,schema:[]),config)
        XCTAssertTrue(try AggregateTableConfiguration.outputSchema(config,schema:[]).isEmpty)
    }
    func testProjectedAggregationFeedsDownstream()throws {
        let graph=DataFlowDocument(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV",config:[:]),.init(id:"aggregate",operatorID:"kwip.data.aggregate",config:config),.init(id:"next",operatorID:"kwip.data.filter",config:[:])],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"aggregate",port:"table")),.init(from:.init(node:"aggregate",port:"table"),to:.init(node:"next",port:"table"))])
        XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"next",in:graph,sourceSchema:{_ in self.schema}).map(\.name),[" entity, ","My total"])
    }
    func testTextJoinDraftRoundTripAndSchema()throws {
        var row=SummaryReducerDraft();row.output="notes";row.column=" entity, ";row.setOperation("joinText",schema:schema)
        row.setSeparatorChoice("newline");row.distinct=true;row.skipEmpty=true
        let decoded=try SummaryReducerDraft.decode(row.configuration)
        XCTAssertEqual(decoded.configuration,row.configuration)
        XCTAssertEqual(decoded.separator,"\n")
        XCTAssertEqual(decoded.eligible(schema).map(\.name),[" entity, "])
        var next=config;next["groupBy"] = .array([]);next["aggregations"] = .array([row.configuration])
        let output=try AggregateTableConfiguration.outputSchema(next,schema:schema)
        XCTAssertEqual(output.map(\.dtype),["string"]);XCTAssertEqual(output.first?.nullable,true)
        row.column="amount";next["aggregations"] = .array([row.configuration])
        XCTAssertThrowsError(try AggregateTableConfiguration.validate(next,schema:schema))
        row.setOperation("sum",schema:schema)
        guard case .object(let numeric)=row.configuration else {return XCTFail()}
        XCTAssertNil(numeric["separator"]);XCTAssertNil(numeric["distinct"])
    }
    func testSeparatorPresetsCustomAndOperationChanges()throws {
        var row=SummaryReducerDraft();row.column="amount"
        row.setOperation("joinText",schema:schema);XCTAssertEqual(row.column,"")
        row.setSeparatorChoice("tab");XCTAssertEqual(row.separator,"\t")
        row.setSeparatorChoice("none");XCTAssertEqual(row.separator,"")
        row.setSeparatorChoice("custom");row.separator=" / "
        row.output="combined";row.column=" entity, "
        XCTAssertEqual(try SummaryReducerDraft.decode(row.configuration).separatorChoice,"custom")
        row.separator=String(repeating:"x",count:1025)
        var next=config;next["aggregations"] = .array([row.configuration])
        XCTAssertThrowsError(try AggregateTableConfiguration.validate(next,schema:schema))
    }

}
