import XCTest
import KwipMLCore
@testable import KwipMLApp

final class NumberListDraftTests:XCTestCase {
    let field=ComponentField(key:"weights",title:"Weights",kind:.numberList,defaultValue:.array([0.25,1]),min:-2,max:4,minItems:1,maxItems:4)
    func testDecimalFormattingAtomicApplyDiscardAndStaleRefusal() {
        var draft=RegistryNumberListDraft(value:.array([0.25,1]))
        XCTAssertEqual(draft.text,"0.25, 1.0")
        draft.text="0.5, 2e-3, -1"
        var writes:[JSONValue]=[]
        XCTAssertFalse(draft.commit(field:field,origin:"a",currentOrigin:"b") {writes.append($0);return true})
        XCTAssertEqual(draft.text,"0.5, 2e-3, -1");XCTAssertTrue(writes.isEmpty)
        XCTAssertFalse(draft.commit(field:field,origin:"a",currentOrigin:"a") {_ in false});XCTAssertTrue(draft.pending)
        XCTAssertTrue(draft.commit(field:field,origin:"a",currentOrigin:"a") {writes.append($0);return true})
        XCTAssertEqual(writes,[.array([.number(0.5),.number(0.002),.number(-1)])]);XCTAssertFalse(draft.pending)
        draft.text="invalid";draft.discard();XCTAssertEqual(draft.text,"0.5, 0.002, -1.0");XCTAssertFalse(draft.pending)
        for text in ["","1,","1,,2","true","nan","inf","1e999","[1,2]","1;2","5","1,1,1,1,1"] {
            draft.text=text
            XCTAssertFalse(draft.commit(field:field,origin:"a",currentOrigin:"a") {_ in XCTFail("Malformed list wrote");return true})
            XCTAssertEqual(draft.text,text);XCTAssertTrue(draft.pending)
        }
    }
    @MainActor func testEmptyCompatibilityAndExactRoundTrip() throws {
        var empty=field;empty.minItems=0;empty.defaultValue = .array([])
        XCTAssertEqual(RegistryNumberListParsing.parse(" \n",field:empty),.array([]))
        XCTAssertNil(RegistryNumberListParsing.parse("1,",field:empty))
        XCTAssertNil(RegistryFieldCompatibility.normalized(.array([true]),for:field))
        let value:JSONValue = .array([0.12345678901234567,0.000000000000001,3])
        XCTAssertEqual(RegistryNumberListParsing.parse(RegistryNumberListParsing.display(value),field:field),RegistryFieldCompatibility.normalized(value,for:field))
        XCTAssertEqual(RegistryFieldCompatibility.normalized(.array([1,0.25]),for:field),.array([.number(1),.number(0.25)]))
        XCTAssertEqual(TensorLearningCanvasAdapter.displayValue(value),RegistryNumberListParsing.display(value))
        XCTAssertNil(RegistryNumberListParsing.parse(String(repeating:"0",count:131073),field:field))
    }
}
