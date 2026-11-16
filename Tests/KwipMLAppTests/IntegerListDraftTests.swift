import XCTest
import KwipMLCore
@testable import KwipMLApp

final class IntegerListDraftTests:XCTestCase {
    let field=ComponentField(key:"shape",title:"Shape",kind:.integerList,defaultValue:.array([-1]),min:-1,max:100,minItems:1,maxItems:7)
    func testAtomicCommitAndRejectionPreserveDraft() {
        var draft=IntegerListDraft(value:.array([-1]))
        var applied:[JSONValue]=[]
        draft.text="2, 3"
        XCTAssertFalse(draft.commit(field:field,origin:"a",currentOrigin:"b") {applied.append($0);return true})
        XCTAssertTrue(applied.isEmpty)
        XCTAssertEqual(draft.text,"2, 3")
        XCTAssertFalse(draft.commit(field:field,origin:"a",currentOrigin:"a") {_ in false})
        XCTAssertTrue(draft.pending)
        XCTAssertTrue(draft.commit(field:field,origin:"a",currentOrigin:"a") {applied.append($0);return true})
        XCTAssertEqual(applied,[.array([2,3])])
        XCTAssertFalse(draft.pending)
        for text in ["", "2,", "2,,3", "2.5,3", "true", "101", "1,2,3,4,5,6,7,8", String(repeating:"1",count:4097)] {
            draft.text=text
            XCTAssertFalse(draft.commit(field:field,origin:"a",currentOrigin:"a") {_ in XCTFail("Invalid list applied");return true})
        }
    }
}
