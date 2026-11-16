import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SigmoidLessonRegistrationTests: XCTestCase {
    func testCatalogMakesMeasuredLessonFindableWithoutTrainingOrMasteryClaims() throws {
        let item = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "sigmoid-lesson" })
        XCTAssertEqual(item.category, "Layers & activations")
        XCTAssertEqual(item.kind, "Construction")
        XCTAssertTrue(item.matches(section: "Library", search: "sigmoid"))
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
        let sigmoid = try XCTUnwrap(entries.first { $0.componentID == "sigmoid" })
        XCTAssertEqual(sigmoid.item.kind, "Example")
        XCTAssertTrue(sigmoid.item.matches(section: "Examples", search: "sigmoid"))
        XCTAssertTrue(sigmoid.item.matches(section: "PyTorch", search: "sigmoid"))
        XCTAssertFalse(sigmoid.item.matches(section: "Practice", search: "sigmoid"))
        XCTAssertFalse(sigmoid.item.matches(section: "TensorFlow", search: "sigmoid"))
        XCTAssertEqual(sigmoid.learning.exampleID, try XCTUnwrap(BundledActivationComponents.entries["sigmoid"]?.learning?.exampleID))
        XCTAssertEqual(sigmoid.item.id, sigmoid.componentID, "The generic example renderer must launch the authored component ID.")
        let practice = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "sigmoid-lesson" })
        XCTAssertTrue(practice.matches(section: "Library", search: "sigmoid"))
        XCTAssertFalse(practice.matches(section: "Examples", search: "sigmoid"))
        let document = try PreparedWorkflowExample.component(sigmoid.componentID).document(device: "cpu")
        XCTAssertTrue(document.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "sigmoid" }; return false })
    }
    @MainActor func testSpecificRuntimeGateBlocksSubmissionAndDoesNotGateHistoricalAffineTask() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start(identity: .sigmoid, mode: .practice)
        XCTAssertTrue(lesson.isActive, lesson.statusMessage)
        XCTAssertTrue(try XCTUnwrap(lesson.lessonRuntimeIssue).contains("Sigmoid"))
        XCTAssertFalse(lesson.canSubmit)
        let count = lesson.attempts.count
        lesson.submit()
        XCTAssertEqual(lesson.attempts.count, count)
        XCTAssertFalse(workspace.isAssessingLearning)
        lesson.leave(returnToPreviousProject: true)
        lesson.start(identity: .composition, mode: .guided)
        XCTAssertNil(lesson.lessonRuntimeIssue)
    }
    @MainActor func testNewTaskIdentityModeAndHintsSurviveManagedReopenWithoutCompletion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start(identity: .sigmoid, mode: .practice)
        XCTAssertTrue(lesson.isActive, lesson.statusMessage)
        lesson.revealHint("sigmoid-independent")
        lesson.save()
        let project = try XCTUnwrap(workspace.activeProjectID)
        let reopened = AffineLearningStore(workspace: workspace)
        reopened.reopenCurrentProject()
        XCTAssertEqual(workspace.activeProjectID, project)
        XCTAssertEqual(try reopened.draft?.resolvedLessonIdentity(), .sigmoid)
        XCTAssertEqual(reopened.mode, .practice)
        XCTAssertTrue(reopened.draft?.hintIDs.contains("sigmoid-independent") == true)
        XCTAssertFalse(reopened.hasCurrentCompletion)
    }
}
