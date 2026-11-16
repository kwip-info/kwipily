import XCTest
import KwipMLCore
@testable import KwipMLApp

final class JoinConfigurationTests:XCTestCase {
    var config:[String:JSONValue] { ["leftKeys":.array([.string(" id, ")]),"rightKeys":.array([.string("key")]),"mode":.string("left"),"nullPolicy":.string("neverMatch"),"cardinality":.string("manyToOne"),"leftPrefix":.string("l_"),"rightPrefix":.string("r_"),"maxRows":.integer(100)] }
    let left:[DataColumnSchema] = [.init(name:" id, ",dtype:"int64"),.init(name:"value",dtype:"float64",nullable:false)]
    let right:[DataColumnSchema] = [.init(name:"key",dtype:"int64"),.init(name:"value",dtype:"string",nullable:false)]
    func testProjectionSettingsMustBePairedEvenBeforeSchemasAreKnown()throws {
        var next=config;next["newColumns"] = .string("keep")
        XCTAssertThrowsError(try JoinTableConfiguration.validate(next,left:[],right:[]))
        next=config;next["outputColumns"] = .array([])
        XCTAssertThrowsError(try JoinTableConfiguration.validate(next,left:[],right:[]))
        next["newColumns"] = .string("keep")
        XCTAssertNoThrow(try JoinTableConfiguration.validate(next,left:[],right:[]))
        XCTAssertNoThrow(try JoinTableConfiguration.validate(config,left:[],right:[]))
    }
    func testExactPairsAndOuterSchema()throws {
        XCTAssertEqual(try JoinTableConfiguration.validate(config,left:left,right:right),config)
        let output=try JoinTableConfiguration.outputSchema(config,left:left,right:right)
        XCTAssertEqual(output.map(\.name),[" id, ","l_value","key","r_value"])
        XCTAssertEqual(output.last?.nullable,true)
        XCTAssertEqual(output[1].nullable,false)
    }
    func testInvalidDraftsAndDtypeCollisionAreRejected()throws {
        for (key,value) in [("leftKeys",JSONValue.array([.string("  ")])),("rightKeys",JSONValue.array([])),("mode",.string("guess")),("maxRows",.integer(0)),("leftPrefix",.string(" ")),("rightPrefix",.string("l_")),("leftKeys",.array([.string("missing")]))] {
            var invalid=config;invalid[key]=value
            XCTAssertThrowsError(try JoinTableConfiguration.validate(invalid,left:left,right:right),key)
        }
        XCTAssertThrowsError(try JoinTableConfiguration.validate(config,left:left,right:[.init(name:"key",dtype:"string")]))
        XCTAssertThrowsError(try JoinTableConfiguration.outputSchema(config,left:left + [.init(name:"r_value",dtype:"string")],right:right))
    }
    func testAllModesAndUnknownSchemas()throws {
        for (mode,l,r) in [("inner",false,false),("left",false,true),("right",true,false),("full",true,true)] {
            var next=config;next["mode"] = .string(mode)
            let result=try JoinTableConfiguration.outputSchema(next,left:left,right:right)
            XCTAssertEqual(result[1].nullable,l);XCTAssertEqual(result[3].nullable,r)
        }
        XCTAssertEqual(try JoinTableConfiguration.validate(config,left:[],right:[]),config)
        XCTAssertTrue(try JoinTableConfiguration.outputSchema(config,left:[],right:right).isEmpty)
        var next=config;next["leftKeys"] = .array([.string(" id, "),.string(" id, ")]);next["rightKeys"] = .array([.string("key"),.string("value")])
        XCTAssertThrowsError(try JoinTableConfiguration.validate(next,left:left,right:right))
        next=config;next["maxRows"] = .number(1.5)
        XCTAssertThrowsError(try JoinTableConfiguration.validate(next,left:left,right:right))
    }
    func testPortProjectionAndJoinedDownstreamSchema()throws {
        var graph=DataFlowDocument(nodes:[.init(id:"a",operatorID:"kwip.data.sourceCSV",config:[:]),.init(id:"b",operatorID:"kwip.data.sourceCSV",config:[:]),.init(id:"j",operatorID:"kwip.data.join",config:config),.init(id:"next",operatorID:"kwip.data.filter",config:[:])],edges:[.init(from:.init(node:"b",port:"table"),to:.init(node:"j",port:"right")),.init(from:.init(node:"a",port:"table"),to:.init(node:"j",port:"left")),.init(from:.init(node:"j",port:"table"),to:.init(node:"next",port:"table"))])
        let schema:(String)->[DataColumnSchema] = {$0 == "a" ? self.left:self.right}
        XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"j",in:graph,inputPort:"left",sourceSchema:schema),left)
        XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"j",in:graph,inputPort:"right",sourceSchema:schema),right)
        XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"next",in:graph,sourceSchema:schema).map(\.name),[" id, ","l_value","key","r_value"])
        graph.edges.append(.init(from:.init(node:"b",port:"table"),to:.init(node:"j",port:"left")))
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"j",in:graph,inputPort:"left",sourceSchema:schema).isEmpty)
        XCTAssertTrue(DataNodeSchemaProjection.upstreamSchema(for:"next",in:graph,sourceSchema:schema).isEmpty)
    }
    func testGridProjectionAndOuterNullabilityPreserveSourceTypes()throws {
        var next=config
        next["outputColumns"] = .array([
            JoinOutputColumn(side:"right",source:"value",name:"context",included:true).json,
            JoinOutputColumn(side:"left",source:" id, ",name:"id",included:true).json,
            JoinOutputColumn(side:"right",source:"key",name:"duplicate",included:false).json])
        next["newColumns"] = .string("drop")
        let result=try JoinTableConfiguration.outputSchema(next,left:left,right:right)
        XCTAssertEqual(result.map(\.name),["context","id"])
        XCTAssertEqual(result.map(\.dtype),["string","int64"])
        XCTAssertEqual(result.first?.nullable,true)
        let rows=try JoinOutputColumn.decode(next["outputColumns"])
        XCTAssertEqual(rows.map(\.source),["value"," id, ","key"])
        XCTAssertFalse(rows.last!.included)
        next["newColumns"] = .string("keep")
        XCTAssertEqual(try JoinTableConfiguration.outputSchema(next,left:left,right:right).map(\.name),["context","id","l_value"])
        next["newColumns"] = .string("error")
        XCTAssertThrowsError(try JoinTableConfiguration.validate(next,left:left,right:right))
    }
    func testGridRefusesDuplicateNamesMissingSourcesAndEmptyProjection()throws {
        var next=config;next["newColumns"] = .string("drop")
        for rows in [[],[JoinOutputColumn(side:"left",source:"absent",name:"x",included:true)],
            [.init(side:"left",source:"value",name:"same",included:true),.init(side:"right",source:"value",name:"same",included:true)]] as [[JoinOutputColumn]] {
            next["outputColumns"] = .array(rows.map(\.json))
            XCTAssertThrowsError(try JoinTableConfiguration.validate(next,left:left,right:right))
        }
        let row=JoinOutputColumn(side:"left",source:"value",name:"x",included:true)
        XCTAssertThrowsError(try JoinOutputColumn.decode(.array([row.json,row.json])))
    }
    func testImagePassThroughAndKeyRefusal()throws {
        let image=DataColumnSchema(name:"pixels",dtype:"image",nullable:false)
        XCTAssertEqual(try JoinTableConfiguration.outputSchema(config,left:left+[image],right:right).last(where:{$0.name == "pixels"})?.dtype,"image")
        var next=config;next["leftKeys"] = .array([.string("pixels")])
        XCTAssertThrowsError(try JoinTableConfiguration.validate(next,left:left+[image],right:right))
    }

}
