import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class CurrentPracticeLessonContentTests:XCTestCase {
    func testRetiredConstructionProgressIDsRedirectWithoutHidingCurrentMathLessons() {
        for operation in OptimizerLearningTask.allCases.map(\.rawValue)+SchedulerLearningTask.allCases.map(\.rawValue)+LossLearningTask.allCases.map(\.rawValue)+GradientLearningTask.allCases.map(\.rawValue)+InitializationLearningTask.allCases.map(\.rawValue)+DataLoadingLearningTask.allCases.map(\.rawValue) {
            XCTAssertTrue(RetiredTrainingLessonDestination.contains(operation),operation)
        }
        XCTAssertFalse(RetiredTrainingLessonDestination.contains(nil))
        XCTAssertFalse(RetiredTrainingLessonDestination.contains("composition-build-reuse"))
        XCTAssertFalse(RetiredTrainingLessonDestination.contains("reshape"))
    }
    func testRetiredTensorSubmissionAndRuntimeRefreshRefuseWithoutChangingSavedDraft()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let workspace=WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        for operation in RetiredTrainingLessonDestination.operations.sorted() {
            XCTAssertTrue(lesson.start(operation:operation),operation)
            let draft=try XCTUnwrap(lesson.draft)
            XCTAssertFalse(lesson.canSubmit,operation)
            XCTAssertFalse(lesson.canRefreshLessonRuntime,operation)
            XCTAssertTrue(lesson.runtimeReadinessError?.contains("read-only") == true,operation)
            lesson.submit()
            XCTAssertTrue(lesson.statusMessage.contains("read-only"),operation)
            XCTAssertFalse(workspace.isAssessingLearning,operation)
            XCTAssertEqual(lesson.draft,draft,operation)
        }
        XCTAssertTrue(lesson.start(operation:"softmax-last-axis"))
        XCTAssertFalse(lesson.runtimeReadinessError?.contains("read-only") == true)
    }
    func testCurrentBundleTrainingExamplesDescribeOwnedControlsAndUnscoredPractice() {
        let ids=OptimizerConfiguration.componentIDs.union(SchedulerConfiguration.componentIDs).union(LossConfiguration.componentIDs)
        let entries=LearnCatalogContent.componentExampleEntries.filter {ids.contains($0.componentID)}
        XCTAssertEqual(Set(entries.map(\.componentID)),ids)
        for entry in entries {
            let text=entry.learning.paragraphs.joined(separator:" ")
            XCTAssertFalse(text.contains("Connect this recipe"),entry.componentID)
            XCTAssertFalse(text.contains("Train’s objective input"),entry.componentID)
            XCTAssertFalse(text.contains("build the scoring connections"),entry.componentID)
            XCTAssertFalse(text.contains("Private lessons measure"),entry.componentID)
            XCTAssertTrue(text.contains("Train"),entry.componentID)
        }
    }
    func testCurrentCourseCoverageDoesNotTeachRemovedToolIdentities() {
        let removed:Set<String>=["kwip.data.cast","kwip.data.selectRename","kwip.data.filter","kwip.data.missing","kwip.data.split","kwip.data.groupSplit","kwip.data.temporalSplit","kwip.data.standardScalerFit","kwip.data.imputerFit","kwip.data.oneHotEncoderFit","kwip.data.transform"]
        XCTAssertTrue(DataPreparationLessonContent.operatorCoverage.isDisjoint(with:removed))
        XCTAssertTrue(DataPreparationLessonContent.operatorCoverage.isSuperset(of:["kwip.data.columns","kwip.data.partition","kwip.data.preprocess"]))
    }
    func testDataNormalizationIsHistoricalAndJourneyUsesCurrentUnscoredPreprocess()throws {
        XCTAssertFalse(LearnCatalogContent.items.contains {$0.id=="normalization"})
        XCTAssertTrue(LearnCatalogContent.historicalEntryIDs.contains("normalization"))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"normalization"))
        let first=try XCTUnwrap(LearningJourney.all.first {$0.id=="first-model"})
        let step=try XCTUnwrap(first.steps.first {$0.entryID=="preprocess"})
        XCTAssertEqual(step.evidence,.unscoredPractice)
        XCTAssertFalse(LearningJourney.all.flatMap(\.steps).contains {$0.entryID=="normalization"})
    }

}
