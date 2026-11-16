import XCTest
@testable import KwipMLCore
final class MCPNamedBoundaryTests:XCTestCase {
    func testCurrentTextLessonCanUseDirectDefinitionBoundaryWithoutLosingNamedInputOwnership()throws {
        let original=try TextGenerationPracticeExample.document(device:"cpu")
        let result=try WorkflowModelBoundary.consolidate(original)
        XCTAssertEqual(result.namedModelInputs?["output"],original.namedModelInputs?["model"])
        XCTAssertNil(result.namedModelInputs?["model"])
        XCTAssertTrue(result.edges.contains{$0.from == .init(address:.init(domain:.model,localID:"output"),port:"definition") && $0.to.address==WorkflowExamples.trainAddress})
        XCTAssertNotNil(original.namedModelInputs?["model"],"The pure upgrade must preserve the original source and frozen History")
        XCTAssertNoThrow(try WorkflowCodec.validate(result))
    }
}
