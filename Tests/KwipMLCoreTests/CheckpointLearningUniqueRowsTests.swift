import XCTest
@testable import KwipMLCore

final class CheckpointLearningUniqueRowsTests:XCTestCase {
    func testAllSuppliedRowsRemainDistinctAtProductionPrecisionAcrossLargeSeedSet()throws {
        for seed in 0..<512 {
            for task in CheckpointLearningTask.allCases {
                let plan=try CheckpointAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:UInt64(seed))
                for fixture in plan.fixtures {
                    let x=fixture.inputs["trainingInputs"]!.values+fixture.inputs["heldoutInputs"]!.values
                    let y=fixture.inputs["trainingTargets"]!.values+fixture.inputs["heldoutTargets"]!.values
                    let firstFeatures=(0..<12).map {Float(x[$0*2])}
                    XCTAssertEqual(Set(firstFeatures).count,12,"\(task.rawValue), seed \(seed), fixture \(fixture.id): rows must remain distinct after float32 conversion")
                    for r in 0..<12 {
                        XCTAssertEqual(y[r*2],0.5*x[r*2]-0.25*x[r*2+1]+0.125,accuracy:1e-15)
                        XCTAssertEqual(y[r*2+1],-0.25*x[r*2]+0.75*x[r*2+1]-0.125,accuracy:1e-15)
                    }
                }
            }
        }
    }
}
