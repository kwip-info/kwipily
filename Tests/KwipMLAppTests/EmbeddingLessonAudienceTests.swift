import XCTest
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingLessonAudienceTests: XCTestCase {
    func testRequiredTasksAndVocabularyExplainLookupWithoutFormulaPrerequisites() {
        for task in EmbeddingLearningTask.allCases {
            XCTAssertTrue(task.task.contains("Embedding"))
            XCTAssertTrue(task.task.contains("Check construction"))
            XCTAssertFalse(task.task.contains("dL/d")); XCTAssertFalse(task.task.contains("⊙"))
            let terms=Set(LessonAudienceGuide.make(operation:task.rawValue).terms.map { $0.name.lowercased() })
            XCTAssertTrue(terms.isSuperset(of:["lookup row","vector","category index","repeat accumulation","frequency","padding","freeze","norm","gradient"]))
        }
        XCTAssertTrue(EmbeddingLearningTask.renormalize.task.contains("before lookup"))
        XCTAssertTrue(EmbeddingLearningTask.paddingFrequency.task.contains("supplied values"))
    }
    func testPublicGuidanceDisclosesDiscreteAndFrozenForwardBehaviorWithoutPromisingFrozenLessonAdmission() {
        let requirements=LessonAudienceGuide.requirements(operation:EmbeddingLearningTask.renormalize.rawValue,fallback:"")
        XCTAssertTrue(requirements.contains("Freeze"))
        XCTAssertTrue(requirements.contains("before lookup"))
        XCTAssertTrue(requirements.contains("whole-number"))
        XCTAssertTrue(requirements.contains("no input gradient"))
        XCTAssertTrue(requirements.contains("disabled"))
    }
    func testOptionalSymbolsAndImmutableAssessmentContracts() throws {
        for symbol in ["W is","V is","D is","N is","index","gradient"] { XCTAssertTrue(EmbeddingLessonReference.explanation.contains(symbol),symbol) }
        for task in EmbeddingLearningTask.allCases {
            let plan=try EmbeddingAssessment.plan(task:task,runtimeVersion:"test",seed:12)
            XCTAssertEqual(plan.id,"assessment."+task.rawValue); XCTAssertEqual(plan.assessmentVersion,ContractVersion())
            XCTAssertEqual(Set(plan.criteria.map(\.id)),["output","tableAfterForward","tableGradient"])
            let fields=try XCTUnwrap(JSONSerialization.jsonObject(with:JSONEncoder().encode(plan.publicTask(mode:.assessment))) as? [String:Any])
            XCTAssertNil(fields["fixtures"]);XCTAssertNil(fields["fixtureSeed"])
        }
    }
}
