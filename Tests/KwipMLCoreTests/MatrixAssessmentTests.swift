import XCTest
@testable import KwipMLCore

final class MatrixAssessmentTests:XCTestCase {
    func testPlansAreSeededBoundedAndGradeBothOperands()throws {
        for task in MatrixLearningTask.allCases {
            let plan=try MatrixAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:17)
            XCTAssertNoThrow(try plan.validate())
            XCTAssertEqual(plan,try MatrixAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:17))
            XCTAssertNotEqual(plan.fixtures,try MatrixAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:18).fixtures)
            XCTAssertEqual(Set(plan.criteria.map(\.id)),["output","leftGradient","rightGradient"])
            XCTAssertEqual(plan.fixtures.count,4)
            for fixture in plan.fixtures {
                XCTAssertEqual(Set(fixture.inputs.keys),["left","right","upstream"])
                XCTAssertEqual(Set(fixture.expected.keys),["output","leftGradient","rightGradient"])
            }
        }
    }
}
