import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LazyLinearStatePresentationTests: XCTestCase {
    private func receipt(mode: LearningMode) throws -> TensorLearningReceipt {
        let plan = try LazyLinearAssessment.plan(runtimeVersion: "test", seed: 42)
        let graph = GraphDocument(name: "My map", nodes: [.init(id:"a",type:"input",name:"Input",x:0,y:0), .init(id:"b",type:"lazy_linear",name:"Layer",x:10,y:0,config:["out_features":4,"bias":true]), .init(id:"c",type:"output",name:"Output",x:20,y:0)], edges:[.init(id:"ab",source:"a",target:"b"),.init(id:"bc",source:"b",target:"c")])
        let bytes = try LazyLinearLearningSubmission(graph:graph).encoded()
        let evidence = LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:SHA256.hash(data:bytes).map { String(format:"%02x",$0) }.joined(),fixtures:plan.fixtures.map { .init(fixtureID:$0.id,tensors:$0.expected) },elapsedMilliseconds:1,maximumTensorElements:100)
        return try TensorLearningReceipt(projectID:UUID().uuidString,operation:LazyLinearAssessment.operation,mode:mode,submissionData:bytes,plan:plan,evidence:evidence)
    }
    func testStateTableUsesReadableChecksWithoutPrivateInputs() throws {
        let receipt = try receipt(mode:.guided)
        let rows = TensorLearningStore.lazyStateRows(receipt:receipt,projectID:receipt.projectID,submission:receipt.submissionData,mode:.guided)
        XCTAssertEqual(rows?.count,7)
        XCTAssertTrue(rows?.contains { $0.check == "Output shape" && $0.measured == "[N, 2, 4]" } == true)
        XCTAssertTrue(rows?.contains { $0.check == "Parameter inventory matches selected settings" && $0.measured == "Yes" } == true)
        XCTAssertFalse(rows?.contains { $0.measured == "1" } == true)
    }
    func testAssessmentOriginAndModeAndStaleBindingsNeverExposeStateTable() throws {
        let guided=try receipt(mode:.guided), assessed=try receipt(mode:.assessment)
        XCTAssertNil(TensorLearningStore.lazyStateRows(receipt:guided,projectID:guided.projectID,submission:guided.submissionData,mode:.assessment))
        XCTAssertNil(TensorLearningStore.lazyStateRows(receipt:assessed,projectID:assessed.projectID,submission:assessed.submissionData,mode:.guided))
        XCTAssertNil(TensorLearningStore.lazyStateRows(receipt:guided,projectID:UUID().uuidString,submission:guided.submissionData,mode:.guided))
        XCTAssertNil(TensorLearningStore.lazyStateRows(receipt:guided,projectID:guided.projectID,submission:Data(),mode:.guided))
    }
}
