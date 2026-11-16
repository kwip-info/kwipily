import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TrainingMappingDraftTests:XCTestCase {
    func testRowsKeepIdentityAndExactNamesThroughReorderAndRemoval()throws {
        var draft=TrainingMappingDraft(features:[" a, ","b","c"],target:" label ")
        let ids=draft.rows.map(\.id)
        draft.move(ids[2],by:-1)
        XCTAssertEqual(draft.features,[" a, ","c","b"])
        draft.setName("changed",for:ids[1])
        XCTAssertEqual(draft.features,[" a, ","c","changed"])
        draft.remove(ids[2]);draft.setName("stale",for:ids[2])
        XCTAssertEqual(draft.features,[" a, ","changed"])
        XCTAssertEqual(draft.target," label ")
        draft.move(ids[0],by:-1);draft.move(ids[1],by:1)
        XCTAssertEqual(draft.features,[" a, ","changed"])
    }
    func testDraftBoundsAndBlankRowsRemainExplicit()throws {
        var draft=TrainingMappingDraft(features:["a"],target:"label")
        draft.remove(draft.rows[0].id)
        XCTAssertEqual(draft.features,["a"])
        draft.add();XCTAssertEqual(draft.features,["a",""])
        XCTAssertThrowsError(try TrainingColumnMapping.validate(features:draft.features,target:draft.target))
        for _ in 0..<150 {draft.add()}
        XCTAssertEqual(draft.rows.count,TrainingColumnMapping.maximumColumns)
        XCTAssertEqual(Set(draft.rows.map(\.id)).count,draft.rows.count)
    }
}
