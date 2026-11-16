import XCTest
import KwipMLCore
@testable import KwipMLApp

final class UnionConfigurationTests:XCTestCase {
    let a:[DataColumnSchema]=[.init(name:"id",dtype:"int64",nullable:false),.init(name:"notes",dtype:"string",nullable:false)]
    let b:[DataColumnSchema]=[.init(name:"notes",dtype:"string",nullable:true),.init(name:"id",dtype:"float64",nullable:false),.init(name:"extra",dtype:"image",nullable:false)]
    func testAlignmentCoverageTypesAndOrigins()throws {
        var c=UnionConfiguration.defaults
        XCTAssertThrowsError(try UnionTableConfiguration.alignment(c,schemas:["table1":a,"table2":b]))
        c["typePolicy"] = .string("losslessNumeric")
        var rows=try UnionTableConfiguration.alignment(c,schemas:["table1":a,"table2":b])
        XCTAssertEqual(rows.map(\.output.name),["id","notes","extra"])
        XCTAssertEqual(rows.map(\.output.dtype),["float64","string","image"])
        XCTAssertEqual(rows.map(\.output.nullable),[false,true,true])
        XCTAssertEqual(rows[0].sources,["id","id"])
        c["coverage"] = .string("shared")
        XCTAssertEqual(try UnionTableConfiguration.alignment(c,schemas:["table1":a,"table2":b]).map(\.output.name),["id","notes"])
        c["coverage"] = .string("exact")
        XCTAssertThrowsError(try UnionTableConfiguration.alignment(c,schemas:["table1":a,"table2":b]))
        c["coverage"] = .string("all");c["sourceColumn"] = .string("origin")
        rows=try UnionTableConfiguration.alignment(c,schemas:["table1":a,"table2":b])
        XCTAssertEqual(rows.last?.sources,["table1","table2"])
        c["sourceColumn"] = .string("id")
        XCTAssertThrowsError(try UnionTableConfiguration.alignment(c,schemas:["table1":a,"table2":b]))
    }
    func testPositionNamesAndSafePortRemoval()throws {
        var c=UnionConfiguration.defaults;c["match"] = .string("position")
        let second:[DataColumnSchema]=[.init(name:"other",dtype:"int64",nullable:false)]
        let rows=try UnionTableConfiguration.alignment(c,schemas:["table1":a,"table2":second])
        XCTAssertEqual(rows.map(\.output.name),["id","notes"])
        XCTAssertEqual(rows[0].sources,["id","other"])
        XCTAssertEqual(rows[1].sources,["notes",nil])
        XCTAssertThrowsError(try UnionTableConfiguration.validate(c,schemas:[:],connected:["table3"]))
        XCTAssertEqual(try UnionTableConfiguration.validate(c,schemas:[:],connected:["table1"]),c)
    }
    func testSchemaProjectsThroughMultipleInputs()throws {
        var c=UnionConfiguration.defaults;c["typePolicy"] = .string("losslessNumeric")
        let graph=DataFlowDocument(nodes:[.init(id:"a",operatorID:"kwip.data.sourceCSV",config:[:]),.init(id:"b",operatorID:"kwip.data.sourceCSV",config:[:]),.init(id:"union",operatorID:"kwip.data.union",config:c),.init(id:"view",operatorID:"kwip.data.view",config:[:])],edges:[.init(from:.init(node:"a",port:"table"),to:.init(node:"union",port:"table1")),.init(from:.init(node:"b",port:"table"),to:.init(node:"union",port:"table2")),.init(from:.init(node:"union",port:"table"),to:.init(node:"view",port:"table"))])
        XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"view",in:graph,sourceSchema:{$0 == "a" ? self.a:self.b}).map(\.name),["id","notes","extra"])
    }
}
