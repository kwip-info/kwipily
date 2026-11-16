import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DataProgramLearnIntegrationTests:XCTestCase {
    @MainActor private func context() -> (WorkspaceStore,TensorLearningStore,URL) {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        return (workspace,TensorLearningStore(workspace:workspace),storage)
    }
    @MainActor func testEightHistoricalRecipeCanvasesRemainReadableButCannotRefreshOrSubmit()throws {
        let (workspace,active,storage)=context();defer {try? FileManager.default.removeItem(at:storage)}
        for task in DataLoadingLearningTask.allCases {
            XCTAssertTrue(active.start(operation:task.rawValue),task.rawValue)
            guard let canvas=active.canvas else {XCTFail("Recipe canvas missing");continue}
            XCTAssertEqual(Set(canvas.registry.components.map(\.id)),["dataset_source","data_loader","train"])
            XCTAssertNil(canvas.registry.component(type:"input"))
            guard let train=canvas.graph.nodes.first(where:{$0.type=="train"}) else {XCTFail("Supplied Train missing");continue}
            XCTAssertTrue(canvas.isFixedTrainingContext(train))
            XCTAssertFalse(canvas.configure(nodeID:train.id,values:["epochs":.integer(10)]))
            for type in ["dataset_source","data_loader"] {canvas.addNode(type:type,x:30,y:30)}
            let source=try XCTUnwrap(canvas.graph.nodes.first {$0.type=="dataset_source"})
            let groups=try XCTUnwrap(canvas.graph.nodes.first {$0.type=="data_loader"})
            XCTAssertTrue(canvas.configure(nodeID:source.id,values:task.datasetValues))
            XCTAssertTrue(canvas.configure(nodeID:groups.id,values:task.loaderValues))
            canvas.connect(source:source.id,sourcePort:"out",target:train.id,targetPort:"dataset")
            canvas.connect(source:groups.id,sourcePort:"out",target:train.id,targetPort:"loader")
            XCTAssertTrue(canvas.validation.isValid,canvas.validation.issues.map(\.message).joined(separator:"; "))
            XCTAssertNoThrow(try active.draft?.submissionData())
            XCTAssertTrue(canvas.axisExplanation.contains("recipe"))
            XCTAssertNotNil(active.runtimeReadinessError)
            XCTAssertFalse(active.canRefreshLessonRuntime)
            XCTAssertTrue(active.runtimeReadinessError?.contains("read-only") == true)
            XCTAssertFalse(active.canSubmit)
        }
        XCTAssertNotNil(workspace.activeProjectID)
    }
    @MainActor func testProgramLessonsAreActionControlsAndRequireIndependentRuntime()throws {
        let (workspace,lesson,storage)=context();defer {try? FileManager.default.removeItem(at:storage)}
        for task in ProgramExportLearningTask.allCases {
            XCTAssertTrue(lesson.start(operation:task.rawValue))
            XCTAssertNil(lesson.canvas,"Artifact action has no fake forward graph")
            XCTAssertEqual(lesson.draft?.programExportRecipe,try ProgramExportConfiguration(values:task.values).recipeValue)
            XCTAssertNotNil(lesson.runtimeReadinessError)
            XCTAssertTrue(lesson.canRefreshLessonRuntime)
            XCTAssertFalse(lesson.canSubmit)
            XCTAssertNoThrow(try lesson.draft?.submissionData())
        }
        XCTAssertNotNil(workspace.activeProjectID)
    }
    @MainActor func testCurrentTrainPracticeIsFindableAndHistoricalTaskGuidanceRemainsReadable()throws {
        let practice = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "train-data-access" })
        XCTAssertTrue(practice.matches(section: "Practice", search: "data access"))
        XCTAssertTrue(practice.matches(section: "Practice", search: "batches"))
        XCTAssertFalse(LearnCatalogContent.items.contains { $0.id == "data-loading" }, "Connected recipes are historical after Train consolidation")
        XCTAssertTrue(LearnCatalogContent.historicalEntryIDs.contains("data-loading"))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID: "data-loading"))
        for component in ["dataset_source", "data_loader"] {
            XCTAssertEqual(ComponentLessonDestination.entryID(type: component), "train-data-access", "Saved recipe identities redirect to the current Train practice while historical progress stays readable")
        }
        for task in DataLoadingLearningTask.allCases {
            XCTAssertFalse(LearnCatalogContent.items.contains { $0.matches(section: "Practice", search: task.rawValue) }, "Historical recipe tasks must not advertise current connected authoring")
        }
        for task in ProgramExportLearningTask.allCases {
            XCTAssertTrue(LearnCatalogContent.items.contains { $0.kind == "Construction" && $0.matches(section: "Library", search: task.rawValue) }, task.rawValue)
        }
        let operations=DataLoadingLearningTask.allCases.map(\.rawValue)+ProgramExportLearningTask.allCases.map(\.rawValue)
        for operation in operations {
            let text=LessonAudienceGuide.requirements(operation:operation,fallback:"UNREVIEWED")
            XCTAssertFalse(text.contains("UNREVIEWED"),operation)
            XCTAssertTrue(text.contains("**Configure**"),operation)
            XCTAssertTrue(text.contains("**Check**"),operation)
            XCTAssertFalse(text.contains("write code"),operation)
            let guide=LessonAudienceGuide.make(operation:operation)
            XCTAssertEqual(guide.goal,TensorLearningTask.title(for:operation))
            XCTAssertLessThan(text.count,1800,"Keep required guidance focused on this task")
        }
    }
}
