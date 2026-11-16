import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TrainDataAccessPracticeTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func testOrdinaryPracticeCompilesInlineControlsWithoutRecipeOrNewSourceNodes() throws {
        let example = PreparedWorkflowExample.component(TrainDataAccessPracticeExample.exampleID)
        let document = try example.document(device: "cpu")
        let baseline = try WorkflowExamples.preparedParallelClassifier(projectID: document.projectID, device: "cpu")
        let registry = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        let plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
            modelRegistry: .builtIn(), dataRegistry: registry, contractVersion: 12)
        XCTAssertEqual(document.schemaVersion, 13)
        XCTAssertEqual(document.nodes.count, baseline.nodes.count)
        XCTAssertEqual(document.edges, baseline.edges)
        XCTAssertEqual(plan.sourceAddresses, [WorkflowExamples.sourceAddress])
        XCTAssertEqual(plan.featureColumns, ["signal", "context"])
        XCTAssertEqual(plan.targetColumn, "label")
        XCTAssertEqual(plan.training, .init(node: "fit", port: "training"))
        XCTAssertEqual(plan.validation, .init(node: "fit", port: "validation"))
        XCTAssertEqual(plan.datasetRecipe?.configuration.values, DatasetConfiguration.defaults)
        XCTAssertEqual(plan.loaderRecipe?.configuration.values, try DataLoaderConfiguration(values: TrainDataAccessPracticeExample.batching).values)
        XCTAssertEqual(plan.loaderRecipe?.configuration.batchSize, 7)
        XCTAssertEqual(plan.settings.batchSize, 7)
        XCTAssertFalse(document.nodes.contains { node in
            switch node.payload { case .datasetRecipe, .batchRecipe: return true; default: return false }
        })
        let train = try XCTUnwrap(document.nodes.first { $0.address == WorkflowExamples.trainAddress })
        guard case .train(let config) = train.payload else { return XCTFail("Missing Train") }
        XCTAssertEqual(config["datasetAccess"], .object(DatasetConfiguration.defaults))
        XCTAssertEqual(config["batching"], .object(TrainDataAccessPracticeExample.batching))
        XCTAssertEqual(example.observationsResource, "workflow-observations")
        XCTAssertEqual(example.sourceResources.map(\.resource), ["workflow-observations"])
        let encoded = try WorkflowCodec.encode(document)
        XCTAssertEqual(try WorkflowCodec.decode(encoded), document)
        XCTAssertFalse(String(decoding: encoded, as: UTF8.self).contains("/Users/"))
    }

    @MainActor func testPracticeLaunchOwnsExistingPublicSourceAndKeepsInlineControlsDurable() throws {
        let storage = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: storage) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: storage, detectCapabilities: false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(TrainDataAccessPracticeExample.exampleID, device: "cpu"), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let path = try XCTUnwrap(host.bindings.path(for: WorkflowExamples.sourceAddress))
        XCTAssertTrue(path.hasPrefix(storage.appendingPathComponent("example-sources").path + "/"))
        XCTAssertEqual(try String(contentsOfFile: path, encoding: .utf8).split(whereSeparator: \.isNewline).count, 65)
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: host.owner.projectID), .workflow(host.owner.document))
        let train = try XCTUnwrap(host.owner.document.nodes.first { $0.address == WorkflowExamples.trainAddress })
        guard case .train(let config) = train.payload else { return XCTFail("Missing Train") }
        XCTAssertEqual(config["batching"], .object(TrainDataAccessPracticeExample.batching))
        XCTAssertEqual(workspace.activeSection, "Build")
    }

    @MainActor func testCurrentDiscoveryExplainsMaterializedBoundaryAndRetainsAllLegacyAssessedTasks() throws {
        XCTAssertTrue(LearnCatalogContent.items.contains { $0.id == TrainDataAccessPracticeExample.lessonID })
        XCTAssertFalse(LearnCatalogContent.items.contains { $0.id == "data-loading" })
        XCTAssertTrue(LearnCatalogContent.historicalEntryIDs.contains("data-loading"))
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID: "data-loading"))
        XCTAssertFalse(LearningLibraryPresentation.items.contains { $0.id == "data-loading" })
        XCTAssertFalse(LearningJourney.all.flatMap { $0.steps + $0.optionalSteps }.contains { $0.entryID == "data-loading" })
        XCTAssertFalse(TrainDataAccessLessonContent.isAssessed)
        XCTAssertTrue(TrainDataAccessLessonContent.boundary.contains("no scored pass"))
        XCTAssertTrue(TrainDataAccessLessonContent.boundary.contains("100,000 prepared rows"))
        XCTAssertTrue(TrainDataAccessLessonContent.boundary.contains("dense prepared tensors"))
        XCTAssertTrue(TrainDataAccessLessonContent.boundary.contains("does not make model training out of core"))
        XCTAssertTrue(TrainDataAccessLessonContent.legacyExplanation.contains("Train → Data access and Batches"))
        XCTAssertTrue(TrainDataAccessLessonContent.legacyExplanation.contains("does not offer new connected"))
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "dataset_source"), "train-data-access")
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "data_loader"), "train-data-access")
        XCTAssertEqual(DataLoadingLearningTask.allCases.map(\.rawValue), [
            "dataset-indexed-observations", "dataset-aligned-tensors", "dataset-selected-view", "dataset-finite-stream",
            "loader-complete-and-short-groups", "loader-repeatable-mixing", "loader-aligned-sequences", "loader-continue-grouping"
        ])
        for task in DataLoadingLearningTask.allCases {
            let graph = try DataLoadingExamples.lessonGraph(task: task)
            XCTAssertEqual(Set(graph.nodes.map(\.type)), ["dataset_source", "data_loader", "train"])
            XCTAssertEqual(task.componentIDs, ["dataset_source", "data_loader"])
            let assessment = try DataLoadingAssessment.plan(task: task, runtimeVersion: "test", seed: 7)
            XCTAssertEqual(assessment.id, "assessment." + task.rawValue)
            XCTAssertEqual(assessment.objectiveID, "data-loading." + task.rawValue)
            try DataLoadingAssessment.validate(assessment)
        }
    }
}
