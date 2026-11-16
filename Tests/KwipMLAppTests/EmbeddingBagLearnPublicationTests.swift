import XCTest
import KwipMLCore
@testable import KwipMLApp

final class EmbeddingBagLearnPublicationTests:XCTestCase {
    func testSixTasksAreFindableWithoutReplacingOrdinaryEmbeddingLab()throws {
        let lab=try XCTUnwrap(LearnCatalogContent.items.first {$0.id == "embedding-bag"})
        XCTAssertEqual(lab.kind,"Construction");XCTAssertEqual(lab.category,"Layers & activations")
        for task in EmbeddingBagLearningTask.allCases {
            for query in [task.rawValue,task.title] {XCTAssertTrue(lab.matches(section:"Library",search:query),query)}
        }
        for query in ["empty groups","item weights","offsets","padding","mean","sum","max"] {
            XCTAssertTrue(lab.matches(section:"Library",search:query),query)
        }
        XCTAssertFalse(lab.matches(section:"Examples",search:"bag"))
        XCTAssertFalse(lab.searchText.localizedCaseInsensitiveContains("mastery"))
        XCTAssertNotNil(LearnCatalogContent.items.first {$0.id == "embedding"})
        XCTAssertEqual(LearnCatalogContent.items.count,Set(LearnCatalogContent.items.map(\.id)).count)
    }
    func testSixGuidesKeepPublicControlsDefineTermsAndDoNotAssignPrivateAnswers()throws {
        for task in EmbeddingBagLearningTask.allCases {
            let text=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"Unreviewed fallback")
            XCTAssertFalse(text.contains("Unreviewed fallback"),task.rawValue)
            for phrase in ["**Build**","**Configure**","**Check**","Bag Input","EmbeddingBag","Check construction","no code","whole number","small change","offset"] {
                XCTAssertTrue(text.localizedCaseInsensitiveContains(phrase),task.rawValue+": "+phrase)
            }
            let guide=LessonAudienceGuide.make(operation:task.rawValue)
            XCTAssertEqual(guide.goal,task.title)
            for term in ["Bag","Category ID","Vector","Offset","Gradient"] {XCTAssertTrue(guide.terms.contains {$0.name == term},task.rawValue+": "+term)}
            let plan=try EmbeddingBagAssessment.plan(task:task,runtimeVersion:"presentation-test",seed:42)
            for criterion in plan.criteria {
                let feedback=try XCTUnwrap(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:criterion.id))
                XCTAssertNotEqual(feedback.title,criterion.id)
                XCTAssertTrue(feedback.nextStep.contains("Check construction"))
                XCTAssertFalse(feedback.nextStep.contains("["),"No private fixture arrays in repair text")
                XCTAssertFalse(feedback.nextStep.contains("write code"))
            }
            XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"indicesGradient"))
            XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"unknown"))
            if !(try task.configuration.usesPerSampleWeights) {
                XCTAssertNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:"sourceWeightsGradient"))
            }
        }
        XCTAssertTrue(LessonAudienceGuide.requirements(operation:EmbeddingBagLearningTask.paddingNorm.rawValue,fallback:"").contains("Maximum norm to 1"))
    }
    @MainActor func testMissingDiscoveredPairBlocksAllSixLessonsAndOffersRefresh()throws {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        for task in EmbeddingBagLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue))
            XCTAssertNotNil(lesson.runtimeReadinessError,task.rawValue)
            XCTAssertTrue(lesson.canRefreshLessonRuntime,task.rawValue)
            XCTAssertFalse(lesson.canSubmit)
            XCTAssertTrue(lesson.statusMessage.contains("group"),lesson.statusMessage)
            XCTAssertFalse(lesson.statusMessage.contains("input gradients"))
        }
    }
}
