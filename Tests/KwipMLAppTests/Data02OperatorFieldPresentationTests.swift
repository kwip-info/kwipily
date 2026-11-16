import XCTest
import KwipMLCore
@testable import KwipMLApp

final class Data02OperatorFieldPresentationTests: XCTestCase {
    func testExpressionColumnReferencePreservesExactNamesAndRefusesUnrepresentableNames() {
        XCTAssertEqual(DataOperatorFieldPresentation.expressionReference(" x, unit "),"[ x, unit ]")
        XCTAssertNil(DataOperatorFieldPresentation.expressionReference("bad]name"))
        XCTAssertNil(DataOperatorFieldPresentation.expressionReference("bad\nname"))
        XCTAssertNil(DataOperatorFieldPresentation.expressionReference(String(repeating:"a",count:201)))
        XCTAssertEqual(DataOperatorFieldPresentation.expressionReference("[allowed"),"[[allowed]")
    }
    func testSchemaFieldsAreOnlyDeclaredPropertiesInLearningOrder() {
        XCTAssertEqual(DataOperatorFieldPresentation.orderedKeys(["overwrite","filename","columns","separator"]),["filename","separator","columns","overwrite"])
        XCTAssertEqual(DataOperatorFieldPresentation.orderedKeys(["nullPolicy","expression","dtype","outputColumn","replaceExisting"]),["expression","outputColumn","dtype","nullPolicy","replaceExisting"])
    }
    func testPolicyLabelsExplainBehaviorRatherThanDisplayingInternalSpelling() {
        XCTAssertEqual(DataOperatorFieldPresentation.optionTitle("nonmatching",key:"nullPolicy"),"Send missing values to nonmatching output")
        XCTAssertEqual(DataOperatorFieldPresentation.optionTitle("full",key:"matchMode"),"Match the complete value")
        XCTAssertEqual(DataOperatorFieldPresentation.title(for:"replaceExisting"),"Allow replacing existing columns")
        XCTAssertEqual(DataOperatorFieldPresentation.optionTitle("\t",key:"separator"),"Tab")
    }
}
