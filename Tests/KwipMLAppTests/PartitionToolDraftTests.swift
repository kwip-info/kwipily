import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PartitionToolDraftTests:XCTestCase {
    let schema:[DataColumnSchema]=[.init(name:"entity",dtype:"string",nullable:false),.init(name:"time",dtype:"int64",nullable:false),.init(name:"class",dtype:"int64",nullable:false)]
    func testStrategySwitchRetainsIndependentDraftsAndSerializesOnlyActiveControls()throws {
        var draft=try PartitionToolDraft(configuration:PartitionToolDraft.defaults)
        draft.set("seed",.string("97"));draft.switchStrategy("entity")
        draft.set("groupBy",.array([.string("entity")]));draft.set("seed",.string("13"))
        let entity=try draft.configuration(schema:schema)
        XCTAssertEqual(entity["seed"],.integer(13));XCTAssertNil(entity["stratifyColumn"])
        draft.switchStrategy("random")
        XCTAssertEqual(try draft.configuration(schema:schema)["seed"],.integer(97))
        draft.switchStrategy("entity");XCTAssertEqual(try draft.configuration(schema:schema),entity)
    }
    func testFriendlyEmbargoUnitsRemainExactAndDoNotRoundFractionalMilliseconds()throws {
        var draft=try PartitionToolDraft(configuration:["strategy":.string("time"),"timeColumn":.string("time"),"timeEncoding":.string("epochMilliseconds"),"trainingEnd":0,"validationEnd":10000,"embargoMilliseconds":1000,"entityColumns":.array([]),"entityPolicy":.string("allowAcrossTime")])
        draft.setEmbargoUnit("hours");draft.embargoText="0.0005"
        XCTAssertEqual(try draft.configuration(schema:schema)["embargoMilliseconds"],.integer(1800))
        draft.setEmbargoUnit("seconds");XCTAssertEqual(draft.embargoText,"1.8")
        draft.embargoText="0.0001";XCTAssertThrowsError(try draft.configuration(schema:schema))
        draft.embargoText="NaN";XCTAssertThrowsError(try draft.configuration(schema:schema))
    }
    func testUnknownDuplicateImageKeysInvalidAllocationsAndTimeTypesRefuse()throws {
        var draft=try PartitionToolDraft(configuration:PartitionToolDraft.defaults)
        draft.set("validationPercent",.string("90"));XCTAssertThrowsError(try draft.configuration(schema:schema))
        draft.switchStrategy("entity");draft.set("groupBy",.array([.string("entity"),.string("entity")]));XCTAssertThrowsError(try draft.configuration(schema:schema))
        draft.set("groupBy",.array([.string("missing")]));XCTAssertThrowsError(try draft.configuration(schema:schema))
        draft.set("groupBy",.array([.string("picture")]));XCTAssertThrowsError(try draft.configuration(schema:[.init(name:"picture",dtype:"image",nullable:false)]))
    }
}
