import XCTest
import CryptoKit
@testable import KwipMLCore

final class MatrixLessonReceiptCompatibilityTests: XCTestCase {
    func testHistoricalFailedFeedbackReceiptStillReplaysAfterPresentationChanges() throws {
        for task in MatrixLearningTask.allCases {
            var oldPlan=try MatrixAssessment.plan(task:task,runtimeVersion:"test",seed:17)
            let original=["output":"Check the contraction width and the order of the two operands. Align broadcast sample axes after the observation axis.","leftGradient":"Each A[i,k] receives Σⱼ G[i,j]B[k,j]. Sum repeated uses across broadcast axes, never across observations.","rightGradient":"Each B[k,j] receives Σᵢ A[i,k]G[i,j]. Broadcast reuses values; reduce those uses back to B's original shape."]
            for index in oldPlan.criteria.indices {oldPlan.criteria[index].nextStep=original[oldPlan.criteria[index].id]!}
            let graph=GraphDocument(name:"Historical construction",nodes:[.init(id:"a",type:"input",name:"Left",x:0,y:0,config:["fixture":.string("left")]),.init(id:"b",type:"input",name:"Right",x:0,y:100,config:["fixture":.string("right")]),.init(id:"m",type:"matmul",name:"Multiply",x:100,y:0),.init(id:"o",type:"output",name:"Output",x:200,y:0)],edges:[.init(id:"am",source:"a",target:"m",targetPort:"a"),.init(id:"bm",source:"b",target:"m",targetPort:"b"),.init(id:"mo",source:"m",target:"o")])
            let bytes=try MatrixLearningSubmission(graph:graph).encoded()
            var measured=oldPlan.fixtures.map {LearningFixtureEvidence(fixtureID:$0.id,tensors:$0.expected)}
            for index in measured.indices {
                for key in ["output","leftGradient","rightGradient"] {measured[index].tensors[key]!.values[0]+=1}
            }
            let evidence=LearningEvidence(assessmentID:oldPlan.id,assessmentVersion:oldPlan.assessmentVersion,fixtureDigest:oldPlan.fixtureDigest,runtime:oldPlan.runtime,submissionDigest:SHA256.hash(data:bytes).map {String(format:"%02x",$0)}.joined(),fixtures:measured,elapsedMilliseconds:1,maximumTensorElements:4096)
            XCTAssertEqual(try LearningAssessor.evaluate(oldPlan,evidence:evidence).status,.failed)
            let receipt=try TensorLearningReceipt(projectID:UUID().uuidString,operation:task.rawValue,mode:.practice,submissionData:bytes,plan:oldPlan,evidence:evidence)
            XCTAssertNoThrow(try receipt.verifiedReplay(expectedProjectID:receipt.projectID))
            XCTAssertNoThrow(try TensorLearningReceipt.decode(receipt.encoded()))
        }
    }
}
