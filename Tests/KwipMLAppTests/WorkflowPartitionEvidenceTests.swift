import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPartitionEvidenceTests:XCTestCase {
    private func provenance(_ validation:[String] = ["c","d"], version:String=String(repeating:"b",count:64))->JSONValue {
        .object(["sourceHash":.string(String(repeating:"a",count:64)),"trainingSamples":2,"validationSamples":2,
            "trainingRowIDs":.array([.string("a"),.string("b")]),"validationRowIDs":.array(validation.map(JSONValue.string)),
            "preparation":.object(["trainingDataVersion":.string(String(repeating:"c",count:64)),"validationDataVersion":.string(version)])])
    }
    func testMembershipIgnoresOrderButValuesRemainDistinct()throws {
        let a=try XCTUnwrap(WorkflowPartitionEvidence.decode(provenance()))
        let reordered=try XCTUnwrap(WorkflowPartitionEvidence.decode(provenance(["d","c"])))
        XCTAssertEqual(a.validationMembershipSHA256,reordered.validationMembershipSHA256)
        XCTAssertEqual(WorkflowPartitionEvidence.comparison(a,reordered),.sameRowsAndValues)
        let transformed=try XCTUnwrap(WorkflowPartitionEvidence.decode(provenance(version:String(repeating:"d",count:64))))
        XCTAssertEqual(WorkflowPartitionEvidence.comparison(a,transformed),.sameRowsDifferentValues)
        let different=try XCTUnwrap(WorkflowPartitionEvidence.decode(provenance(["e","f"])))
        XCTAssertEqual(WorkflowPartitionEvidence.comparison(a,different),.differentRowsOrSource)
        XCTAssertEqual(WorkflowPartitionEvidence.comparison(a,nil),.unavailable)
    }
    func testMissingEvidenceIsUnknownAndMalformedEvidenceIsRejected()throws {
        XCTAssertNil(try WorkflowPartitionEvidence.decode(.object(["preparation":.object([:])])))
        XCTAssertThrowsError(try WorkflowPartitionEvidence.decode(provenance(["a","d"])))
        XCTAssertThrowsError(try WorkflowPartitionEvidence.decode(provenance(["c","c"])))
        XCTAssertThrowsError(try WorkflowPartitionEvidence.decode(provenance(["c"])))
        XCTAssertThrowsError(try WorkflowPartitionEvidence.decode(provenance(version:"invalid")))
        XCTAssertThrowsError(try WorkflowPartitionEvidence.decode(.object(["sourceHash":.string(String(repeating:"a",count:64))])))
    }
}
