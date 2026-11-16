import XCTest
@testable import KwipMLApp

final class TensorPreviewFormattingTests:XCTestCase {
    func testOutputPortSelectionPrefersValuesAndResetsAbsentPresentationChoice() {
        XCTAssertEqual(TensorPreviewFormatting.selectedOutputPort(available:["indices","out"],preferred:nil),"out")
        XCTAssertEqual(TensorPreviewFormatting.selectedOutputPort(available:["indices","out"],preferred:"indices"),"indices")
        XCTAssertEqual(TensorPreviewFormatting.selectedOutputPort(available:["out"],preferred:"indices"),"out")
        XCTAssertNil(TensorPreviewFormatting.selectedOutputPort(available:[],preferred:"indices"))
    }

    func testIntegerPreviewPreservesFullSignedRangeFromWorkerJSON()throws {
        let bytes=Data(#"{"dtype":"int64","values":[9007199254740993,9223372036854775807,-9223372036854775808]}"#.utf8)
        let preview=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
        XCTAssertEqual(TensorPreviewFormatting.values(preview),"9007199254740993   9223372036854775807   -9223372036854775808")
    }
    func testFloatingPreviewAndMalformedTypedValuesAreDistinct() {
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"float32","values":[1.25,-0.125]]),"1.250   -0.125")
        for value:Any in [true,"2",1.5] {
            XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"int64","values":[value]]),"Invalid captured values")
        }
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"mystery","values":[1]]),"Unsupported captured type")
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"int64"]),"No captured values")
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"int64","values":Array(0..<12)]).split(separator:" ").count,8)
    }
    func testShortSummaryValidatesOnlyItsBoundedDisplayedPrefix() {
        let retained: [Any] = (0..<8).map { $0 as Any } + [Double.infinity as Any]
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype": "int64", "values": retained]), "0   1   2   3   4   5   6   7")
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype": "float64", "values": [[1, 2]]]), "Invalid captured values")
    }
    func testPolicyOmissionIsDistinctFromMissingOrInvalidValues() {
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"float32","values":[],"captureStatus":"omittedByPolicy"]),"Not captured by this run’s policy")
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"float32","values":[42],"captureStatus":"omittedByPolicy"]),"Invalid captured values")
        XCTAssertEqual(TensorPreviewFormatting.values(["dtype":"float32","values":[],"captureStatus":"unknown"]),"Invalid capture status")
    }

}

extension TensorPreviewFormattingTests {
    func testNamedPortSelectionKeepsExactIndicesAndInheritedPolicy() throws {
        let value:[String:Any] = ["outputs":["out":["shape":[1,2],"dtype":"float32","values":[1.25]],"indices":["shape":[1,2],"dtype":"int64","values":[Int64(9007199254740993)]]]]
        let ports=TensorPreviewFormatting.outputs(value)
        XCTAssertEqual(Set(ports.keys),["out","indices"])
        XCTAssertEqual(TensorPreviewFormatting.values(try XCTUnwrap(ports["indices"])),"9007199254740993")
        XCTAssertNil(TensorPreviewFormatting.output(value,port:"missing"))
        XCTAssertNil(TensorPreviewFormatting.output(["dtype":"float32","values":[1]],port:"indices"))
        var hidden=value;hidden["captureStatus"]="omittedByPolicy"
        XCTAssertEqual(TensorPreviewFormatting.values(try XCTUnwrap(TensorPreviewFormatting.output(hidden,port:"indices"))),"Invalid captured values")
        let omitted:[String:Any] = ["captureStatus":"omittedByPolicy","outputs":["indices":["dtype":"int64","values":[]]]]
        XCTAssertEqual(TensorPreviewFormatting.values(try XCTUnwrap(TensorPreviewFormatting.output(omitted,port:"indices"))),"Not captured by this run’s policy")
        XCTAssertTrue(TensorPreviewFormatting.outputs(["outputs":["indices":42]]).isEmpty)
    }
}
