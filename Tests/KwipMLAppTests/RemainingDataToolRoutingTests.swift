import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RemainingDataToolRoutingTests:XCTestCase {
    func testReshapeAndSplitValidateThroughSharedAtomicRouter()throws {
        let schema:[DataColumnSchema]=[.init(name:"customer",dtype:"string",nullable:false),.init(name:"jan",dtype:"int64",nullable:false),.init(name:"feb",dtype:"int64",nullable:true)]
        var reshape=ReshapeTableConfiguration.defaults
        reshape["idColumns"] = .array([.string("customer")]);reshape["valueColumns"] = .array([.string("jan"),.string("feb")])
        XCTAssertEqual(try DataTableConfiguration.validate(reshape,operatorID:"kwip.data.reshapeTable",schema:schema),reshape)
        var split=SplitTextConfigurationDraft.defaults;split["source"] = .string("customer")
        XCTAssertEqual(try DataTableConfiguration.validate(split,operatorID:"kwip.data.splitText",schema:schema),split)
        reshape["fieldName"] = .string("customer")
        XCTAssertThrowsError(try DataTableConfiguration.validate(reshape,operatorID:"kwip.data.reshapeTable",schema:schema))
    }
    func testIdentityOperatorsProjectBothUniqueOutputsAndReshapeLong()throws {
        let temporal:[String:JSONValue]=["version":.integer(1),"unit":.string("day")]
        let schema:[DataColumnSchema]=[.init(name:"id",dtype:"string",nullable:false),.init(name:"jan",dtype:"date",nullable:false,temporal:temporal),.init(name:"feb",dtype:"date",nullable:true,temporal:temporal)]
        var reshape=ReshapeTableConfiguration.defaults;reshape["idColumns"] = .array([.string("id")]);reshape["valueColumns"] = .array([.string("jan"),.string("feb")])
        for port in ["unique","duplicates"] {
            let graph=DataFlowDocument(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV",config:[:]),.init(id:"sort",operatorID:"kwip.data.sort",config:[:]),.init(id:"unique",operatorID:"kwip.data.unique",config:[:]),.init(id:"reshape",operatorID:"kwip.data.reshapeTable",config:reshape),.init(id:"view",operatorID:"kwip.data.view",config:[:])],edges:[.init(from:.init(node:"source",port:"table"),to:.init(node:"sort",port:"table")),.init(from:.init(node:"sort",port:"table"),to:.init(node:"unique",port:"table")),.init(from:.init(node:"unique",port:port),to:.init(node:"reshape",port:"table")),.init(from:.init(node:"reshape",port:"table"),to:.init(node:"view",port:"table"))])
            let output=DataNodeSchemaProjection.upstreamSchema(for:"view",in:graph,sourceSchema:{_ in schema})
            XCTAssertEqual(output.map(\.name),["id","field","value"]);XCTAssertEqual(output.last?.temporal,temporal)
        }
    }
}
