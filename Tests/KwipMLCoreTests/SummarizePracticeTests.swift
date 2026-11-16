import XCTest
@testable import KwipMLCore
final class SummarizePracticeTests:XCTestCase {
    func testPracticeIsDataOnlyAndPreservesTextSeparator()throws {
        let document=try SummarizePracticeExample.document(projectID:"A5FE8DAB-C15F-4ACB-B3D6-2458C85C689A")
        XCTAssertEqual(document.nodes.count,3)
        XCTAssertEqual(document.edges.count,2)
        XCTAssertTrue(document.nodes.allSatisfy {$0.address.domain == .data})
        let example=PreparedWorkflowExample.component(SummarizePracticeExample.exampleID)
        XCTAssertEqual(example.sourceResources.first?.resource,SummarizePracticeExample.observationsResource)
        XCTAssertTrue(try String(contentsOf:SummarizePracticeExample.observationsURL(),encoding:.utf8).contains("A,Notebook,2,25.0"))
        let roundTrip=try WorkflowCodec.decode(WorkflowCodec.encode(document))
        XCTAssertEqual(roundTrip,document)
    }
}
