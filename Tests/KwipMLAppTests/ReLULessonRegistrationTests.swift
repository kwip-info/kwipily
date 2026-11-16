import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ReLULessonRegistrationTests: XCTestCase {
    func testCatalogMakesMeasuredLessonFindableWithoutTrainingOrMasteryClaims() throws {
        let item = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "relu-lesson" })
        XCTAssertEqual(item.category, "Layers & activations")
        XCTAssertEqual(item.kind, "Construction")
        XCTAssertTrue(item.matches(section: "Library", search: "relu"))
        XCTAssertFalse(item.searchText.localizedCaseInsensitiveContains("calibrated probabilities"))
        XCTAssertFalse(item.searchText.localizedCaseInsensitiveContains("mastery"))
    }
    func testSharedBundledExamplesIncludePublishedPoolingAndActivationsAndSeparatePractice() throws {
        let entries = LearnCatalogContent.componentExampleEntries
        let pooling = Set(BundledPoolingComponents.entries.compactMap { $0.value.published && $0.value.learning != nil ? $0.key : nil })
        let activations = Set(BundledActivationComponents.entries.compactMap { $0.value.published && $0.value.learning != nil ? $0.key : nil })
        let normalization = Set(BundledNormalizationComponents.entries.compactMap { $0.value.published && $0.value.learning != nil ? $0.key : nil })
        let layers = Set(BundledLayerComponents.entries.compactMap { $0.value.published && $0.value.learning != nil ? $0.key : nil })
        let tensorLayout = Set(BundledTensorLayoutComponents.entries.compactMap { $0.value.published && $0.value.learning != nil ? $0.key : nil })
        XCTAssertEqual(Set(entries.map(\.componentID)), pooling.union(activations).union(normalization).union(layers).union(tensorLayout).union(["embedding_bag", "bag_input", "matmul", "embedding", "index_input", "conv_transpose1d","conv_transpose2d","conv_transpose3d","conv1d", "conv2d", "conv3d", "concat", "add", "stack", "permute", "split", "chunk", "gather", "einsum", "unfold", "fold", "optimizer_sgd", "optimizer_adam", "optimizer_adamw", "optimizer_rmsprop", "optimizer_adagrad", "scheduler_step_lr","scheduler_cosine_annealing_lr","scheduler_reduce_on_plateau","scheduler_one_cycle_lr","scheduler_linear_lr","rnn","lstm","gru","rnn_cell","lstm_cell","gru_cell", "multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention","rotary_position","loss_cross_entropy","loss_bce_with_logits","loss_mse","loss_l1","loss_nll","loss_kl_div","loss_ctc","loss_triplet_margin"]))
        XCTAssertEqual(Set(entries.map { $0.item.id }).count, entries.count)
        let relu = try XCTUnwrap(entries.first { $0.componentID == "relu" })
        XCTAssertEqual(relu.item.kind, "Example")
        XCTAssertTrue(relu.item.matches(section: "Examples", search: "relu"))
        XCTAssertTrue(relu.item.matches(section: "PyTorch", search: "relu"))
        XCTAssertFalse(relu.item.matches(section: "Practice", search: "relu"))
        XCTAssertFalse(relu.item.matches(section: "TensorFlow", search: "relu"))
        XCTAssertEqual(relu.learning.exampleID, try XCTUnwrap(BundledActivationComponents.entries["relu"]?.learning?.exampleID))
        XCTAssertEqual(relu.item.id, relu.componentID, "The generic example renderer must launch the authored component ID.")
        let practice = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "relu-lesson" })
        XCTAssertTrue(practice.matches(section: "Library", search: "relu"))
        XCTAssertFalse(practice.matches(section: "Examples", search: "relu"))
        let document = try PreparedWorkflowExample.component(relu.componentID).document(device: "cpu")
        XCTAssertTrue(document.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "relu" }; return false })
    }
    @MainActor func testSpecificRuntimeGateBlocksSubmissionAndDoesNotGateHistoricalAffineTask() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start(identity: .relu, mode: .practice)
        XCTAssertTrue(lesson.isActive, lesson.statusMessage)
        XCTAssertTrue(try XCTUnwrap(lesson.lessonRuntimeIssue).contains("ReLU"))
        XCTAssertFalse(lesson.canSubmit)
        let count = lesson.attempts.count
        lesson.submit()
        XCTAssertEqual(lesson.attempts.count, count)
        XCTAssertFalse(workspace.isAssessingLearning)
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        // Historic compatible workers have no optional ReLU descriptor.
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"], descriptor: JSONEncoder().encode(contract))
        XCTAssertNil(lesson.lessonRuntimeIssue)
        XCTAssertNil(workspace.capabilityBoundaryError(for: ExampleFlows.reluClassifier()))
        lesson.leave(returnToPreviousProject: true)
        lesson.start(identity: .composition, mode: .guided)
        XCTAssertNil(lesson.lessonRuntimeIssue)
    }
    @MainActor func testNewTaskIdentityModeAndHintsSurviveManagedReopenWithoutCompletion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start(identity: .relu, mode: .practice)
        XCTAssertTrue(lesson.isActive, lesson.statusMessage)
        lesson.revealHint("relu-independent")
        lesson.save()
        let project = try XCTUnwrap(workspace.activeProjectID)
        let reopened = AffineLearningStore(workspace: workspace)
        reopened.reopenCurrentProject()
        XCTAssertEqual(workspace.activeProjectID, project)
        XCTAssertEqual(try reopened.draft?.resolvedLessonIdentity(), .relu)
        XCTAssertEqual(reopened.mode, .practice)
        XCTAssertTrue(reopened.draft?.hintIDs.contains("relu-independent") == true)
        XCTAssertFalse(reopened.hasCurrentCompletion)
    }
}
