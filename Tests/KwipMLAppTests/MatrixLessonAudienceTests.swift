import XCTest
import KwipMLCore
@testable import KwipMLApp

final class MatrixLessonAudienceTests: XCTestCase {
    func testRequiredTasksExplainVisualConstructionAndChangeSensitivityWithoutFormulaPrerequisites() {
        for task in MatrixLearningTask.allCases {
            XCTAssertTrue(task.task.contains("Matrix multiply"))
            XCTAssertTrue(task.task.contains("Check construction"))
            XCTAssertTrue(task.task.contains("small change"))
            XCTAssertFalse(task.task.contains("dL/d"))
            XCTAssertFalse(task.task.contains("⊙"))
            XCTAssertTrue(task.task.contains("observations"))
            let terms=Set(LessonAudienceGuide.make(operation:task.rawValue).terms.map { $0.name.lowercased() })
            XCTAssertTrue(terms.isSuperset(of:["matrix","array","directions","contraction","broadcasting","change sensitivity"]))
        }
        XCTAssertTrue(MatrixLearningTask.broadcast.task.contains("size 1"))
    }
    func testOptionalEquationsDefineTheirSymbolsAndDoNotReplacePublicRules() {
        let explanation=MatrixLessonReference.explanation
        for term in ["A is", "B is", "C is", "G is", "transpose", "sum"] { XCTAssertTrue(explanation.contains(term), term) }
        XCTAssertTrue(LessonAudienceGuide.requirements(operation:MatrixLearningTask.contraction.rawValue,fallback:"").contains("Do not transpose, scale or normalize"))
        XCTAssertTrue(LessonAudienceGuide.requirements(operation:MatrixLearningTask.broadcast.rawValue,fallback:"").contains("equal sizes or size 1"))
    }
    func testCopyRetainsVersionedNumericalPrivacyContract() throws {
        for task in MatrixLearningTask.allCases {
            let plan=try MatrixAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:31)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue)
            XCTAssertEqual(plan.assessmentVersion,ContractVersion())
            XCTAssertEqual(plan.runtime.capabilities,["matmul":ContractVersion()])
            XCTAssertEqual(Set(plan.criteria.map(\.id)),["output","leftGradient","rightGradient"])
            for criterion in plan.criteria {
                guard case .tensor(_,let absolute,let relative)=criterion.rule else {return XCTFail("Numerical criterion changed")}
                XCTAssertEqual(absolute,1e-6);XCTAssertEqual(relative,1e-6)
            }
            let fields=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(plan.publicTask(mode:.assessment))) as? [String:Any])
            XCTAssertNil(fields["fixtures"]);XCTAssertNil(fields["fixtureSeed"])
        }
    }
}
