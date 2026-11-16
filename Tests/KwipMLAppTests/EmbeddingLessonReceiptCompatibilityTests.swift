import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingLessonReceiptCompatibilityTests: XCTestCase {
    func testAllHistoricalFailedReceiptsRetainExactFeedback() throws {
        let historical=["output":"For each index position, copy the selected row's three coordinates. Repetition repeats a row; it does not add rows together. Apply any forward norm limit first.","tableAfterForward":"Only selected rows may be renormalized. Padding suppression is a gradient policy, not an instruction to replace supplied padding values with zeros.","tableGradient":"Start with zeros shaped like W. Scatter-add each upstream row at its selected index. Suppress padding and divide by occurrence count only when requested. Do not differentiate integer indices or the no-grad renormalization."]
        for task in EmbeddingLearningTask.allCases {
            var plan=try EmbeddingAssessment.plan(task:task,runtimeVersion:"test",seed:13)
            for i in plan.criteria.indices {plan.criteria[i].nextStep=historical[plan.criteria[i].id]!}
            let graph=GraphDocument(name:"Historical lookup",nodes:[.init(id:"a",type:"input",name:"Indices",x:0,y:0),.init(id:"b",type:"embedding",name:"Lookup",x:100,y:0,config:try task.configuration.values),.init(id:"c",type:"output",name:"Vectors",x:200,y:0)],edges:[.init(id:"ab",source:"a",target:"b"),.init(id:"bc",source:"b",target:"c")])
            let bytes=try EmbeddingLearningSubmission(graph:graph).encoded()
            var measured=plan.fixtures.map {LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)}
            for i in measured.indices {for key in historical.keys {measured[i].tensors[key]!.values[0]+=1}}
            let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined(),fixtures:measured,elapsedMilliseconds:1,maximumTensorElements:4096)
            XCTAssertEqual(try LearningAssessor.evaluate(plan,evidence:evidence).status,.failed)
            let receipt=try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice,submissionData:bytes,plan:plan,evidence:evidence)
            XCTAssertNoThrow(try TensorLearningReceipt.decode(receipt.encoded()))
        }
    }
}
