import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PartitionConfigurationTests:XCTestCase {
    let schema:[DataColumnSchema]=[.init(name:" entity, ",dtype:"string",nullable:false),.init(name:"time",dtype:"string",nullable:false),.init(name:"millis",dtype:"int64",nullable:false),.init(name:"value",dtype:"float64",nullable:false)]
    var grouped:[String:JSONValue] {["groupBy":.array([.string(" entity, ")]),"nullKeys":.string("error"),"allocationUnit":.string("groups"),"seed":.integer(42),"validationPercent":.integer(20),"testPercent":.integer(20)]}
    var temporal:[String:JSONValue] {["timeColumn":.string("time"),"timeEncoding":.string("iso8601Offset"),"trainingEnd":.string("2026-01-01T00:00:00Z"),"validationEnd":.string("2026-02-01T00:00:00+00:00"),"embargoMilliseconds":.integer(1000),"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")]}
    func testGroupAllocationAndExactNames()throws {
        XCTAssertEqual(try PartitionTableConfiguration.validate(grouped,operatorID:"kwip.data.groupSplit",schema:schema),grouped)
        for (key,value) in [("groupBy",JSONValue.array([])),("groupBy",.array([.string(" ")])),("groupBy",.array([.string("missing")])),("allocationUnit",.string("rows")),("nullKeys",.string("group")),("seed",.integer(-1)),("validationPercent",.integer(80)),("testPercent",.number(20.5))] {
            var invalid=grouped;invalid[key]=value
            XCTAssertThrowsError(try PartitionTableConfiguration.validate(invalid,operatorID:"kwip.data.groupSplit",schema:schema),key)
        }
    }
    func testTemporalStrictTimesTypesAndOrdering()throws {
        XCTAssertEqual(try PartitionTableConfiguration.validate(temporal,operatorID:"kwip.data.temporalSplit",schema:schema),temporal)
        for bad in ["2026-01-01", "2026-01-01T00:00:00", "2026-02-30T00:00:00Z", "2026-01-01T00:00:00.1234567Z", "2026-01-01T00:00:60Z", "0000-01-01T00:00:00Z", "2026-01-01T00:00:00+24:00"] {
            var invalid=temporal;invalid["trainingEnd"] = .string(bad)
            XCTAssertThrowsError(try PartitionTableConfiguration.validate(invalid,operatorID:"kwip.data.temporalSplit",schema:schema),bad)
        }
        var invalid=temporal;invalid["validationEnd"] = .string("2025-12-31T19:00:00-05:00")
        XCTAssertThrowsError(try PartitionTableConfiguration.validate(invalid,operatorID:"kwip.data.temporalSplit",schema:schema),"Equal instants are not ordered cutoffs")
        invalid=temporal;invalid["timeColumn"] = .string("millis")
        XCTAssertThrowsError(try PartitionTableConfiguration.validate(invalid,operatorID:"kwip.data.temporalSplit",schema:schema))
    }
    func testExactDistantMicrosecondsAndUTCYearBounds()throws {
        XCTAssertEqual(try PartitionTableConfiguration.instant(.string("1970-01-01T00:00:00Z"),encoding:"iso8601Offset"),0)
        XCTAssertEqual(try PartitionTableConfiguration.instant(.string("0001-01-01T00:00:00Z"),encoding:"iso8601Offset"),-62135596800000000)
        let first=try PartitionTableConfiguration.instant(.string("9999-12-31T23:59:59.999998Z"),encoding:"iso8601Offset")
        let second=try PartitionTableConfiguration.instant(.string("9999-12-31T23:59:59.999999Z"),encoding:"iso8601Offset")
        XCTAssertEqual(second-first,1)
        XCTAssertThrowsError(try PartitionTableConfiguration.instant(.string("0001-01-01T00:00:00+01:00"),encoding:"iso8601Offset"))
        XCTAssertThrowsError(try PartitionTableConfiguration.instant(.string("9999-12-31T23:59:59-01:00"),encoding:"iso8601Offset"))
    }
    func testEpochBoundsEntitiesAndUnknownSchema()throws {
        var config=temporal;config["timeEncoding"] = .string("epochMilliseconds");config["timeColumn"] = .string("millis");config["trainingEnd"] = .integer(0);config["validationEnd"] = .integer(10000)
        config["entityColumns"] = .array([.string(" entity, ")]);config["entityPolicy"] = .string("requireDisjoint")
        XCTAssertEqual(try PartitionTableConfiguration.validate(config,operatorID:"kwip.data.temporalSplit",schema:schema),config)
        for (key,value) in [("trainingEnd",JSONValue.number(0.5)),("trainingEnd",.integer(-62135596800001)),("validationEnd",.integer(253402300800000)),("entityColumns",.array([])),("embargoMilliseconds",.integer(86400000001))] {
            var invalid=config;invalid[key]=value
            XCTAssertThrowsError(try PartitionTableConfiguration.validate(invalid,operatorID:"kwip.data.temporalSplit",schema:schema),key)
        }
        XCTAssertEqual(try PartitionTableConfiguration.validate(grouped,operatorID:"kwip.data.groupSplit",schema:[]),grouped)
    }
    func testAllStrategiesAndThreePortsPreserveSchema() {
        var entity=grouped;entity["strategy"] = .string("entity")
        entity.removeValue(forKey:"nullKeys");entity.removeValue(forKey:"allocationUnit")
        var chronological=temporal;chronological["strategy"] = .string("time")
        for config in [PartitionToolDraft.defaults,entity,chronological] {
            for port in ["training","validation","test"] {
                let graph=DataFlowDocument(nodes:[
                    .init(id:"source",operatorID:"kwip.data.sourceCSV",config:[:]),
                    .init(id:"partition",operatorID:"kwip.data.partition",config:config),
                    .init(id:"inspect",operatorID:"kwip.data.view",config:[:])],edges:[
                        .init(from:.init(node:"source",port:"table"),to:.init(node:"partition",port:"table")),
                        .init(from:.init(node:"partition",port:port),to:.init(node:"inspect",port:"table"))])
                XCTAssertEqual(DataNodeSchemaProjection.upstreamSchema(for:"inspect",in:graph,sourceSchema:{_ in self.schema}),schema)
            }
        }
    }
}
