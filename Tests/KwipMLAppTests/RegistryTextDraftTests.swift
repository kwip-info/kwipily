import XCTest
@testable import KwipMLApp
import KwipMLCore

final class RegistryTextDraftTests:XCTestCase {
    private var field:ComponentField {.init(key:"equation",title:"Custom notation",kind:.text,defaultValue:.string("naa->na"),min:nil,max:nil,minItems:1,maxItems:128)}
    func testTextDraftRetainsCommittedNotationAndAppliesCompleteValue() {
        var draft=IntegerListDraft(value:.string("naa->na"));XCTAssertEqual(draft.text,"naa->na");XCTAssertFalse(draft.pending)
        draft.text="nab->nba";var received:JSONValue?
        XCTAssertTrue(draft.commit(field:field,origin:"active",currentOrigin:"active",apply:{received=$0;return true}))
        XCTAssertEqual(received,.string("nab->nba"));XCTAssertFalse(draft.pending)
    }
    func testInvalidTextAndRetiredOriginNeverWriteTheComponent() {
        var draft=IntegerListDraft(value:.string("naa->na"));draft.text="";var calls=0
        XCTAssertFalse(draft.commit(field:field,origin:"active",currentOrigin:"active",apply:{_ in calls+=1;return true}));XCTAssertTrue(draft.pending)
        draft.text="nab->nba"
        XCTAssertFalse(draft.commit(field:field,origin:"active",currentOrigin:"retired",apply:{_ in calls+=1;return true}));XCTAssertEqual(calls,0)
        draft=IntegerListDraft(value:.string("naa->na"));XCTAssertEqual(draft.text,"naa->na");XCTAssertFalse(draft.pending)
    }
}
