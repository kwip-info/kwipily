import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TanhLessonRegistrationTests: XCTestCase {
    func testCatalogMakesMeasuredLessonFindableWithoutTrainingOrMasteryClaims() throws {
        let item = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "tanh-lesson" })
        XCTAssertEqual(item.category, "Layers & activations")
        XCTAssertEqual(item.kind, "Construction")
        XCTAssertTrue(item.matches(section: "Library", search: "tanh"))
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
        let tanh = try XCTUnwrap(entries.first { $0.componentID == "tanh" })
        XCTAssertEqual(tanh.item.kind, "Example")
        XCTAssertTrue(tanh.item.matches(section: "Examples", search: "tanh"))
        XCTAssertTrue(tanh.item.matches(section: "PyTorch", search: "tanh"))
        XCTAssertFalse(tanh.item.matches(section: "Practice", search: "tanh"))
        XCTAssertFalse(tanh.item.matches(section: "TensorFlow", search: "tanh"))
        XCTAssertEqual(tanh.learning.exampleID, try XCTUnwrap(BundledActivationComponents.entries["tanh"]?.learning?.exampleID))
        XCTAssertEqual(tanh.item.id, tanh.componentID, "The generic example renderer must launch the authored component ID.")
        let practice = try XCTUnwrap(LearnCatalogContent.items.first { $0.id == "tanh-lesson" })
        XCTAssertTrue(practice.matches(section: "Library", search: "tanh"))
        XCTAssertFalse(practice.matches(section: "Examples", search: "tanh"))
        let document = try PreparedWorkflowExample.component(tanh.componentID).document(device: "cpu")
        XCTAssertTrue(document.nodes.contains { if case .modelPrimitive(let type, _, _) = $0.payload { return type == "tanh" }; return false })
    }
    @MainActor func testSpecificRuntimeGateBlocksSubmissionAndDoesNotGateHistoricalAffineTask() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start(identity: .tanh, mode: .practice)
        XCTAssertTrue(lesson.isActive, lesson.statusMessage)
        XCTAssertTrue(try XCTUnwrap(lesson.lessonRuntimeIssue).contains("Tanh"))
        XCTAssertFalse(lesson.canSubmit)
        let count = lesson.attempts.count
        lesson.submit()
        XCTAssertEqual(lesson.attempts.count, count)
        XCTAssertFalse(workspace.isAssessingLearning)
        let contract = try BuiltInCapabilityBridge.executionContract(registry: workspace.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        // Historic compatible workers have no optional Tanh descriptor.
        try workspace.installDiscoveredRuntimeContract(version: "2.14.1", expectedVersion: nil, devices: ["cpu"], descriptor: JSONEncoder().encode(contract))
        XCTAssertNil(lesson.lessonRuntimeIssue)
        XCTAssertNil(workspace.capabilityBoundaryError(for: ExampleFlows.tanhClassifier()))
        lesson.leave(returnToPreviousProject: true)
        lesson.start(identity: .composition, mode: .guided)
        XCTAssertNil(lesson.lessonRuntimeIssue)
    }
    @MainActor func testNewTaskIdentityModeAndHintsSurviveManagedReopenWithoutCompletion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(storageOverride: root, detectCapabilities: false)
        let lesson = AffineLearningStore(workspace: workspace)
        lesson.start(identity: .tanh, mode: .practice)
        XCTAssertTrue(lesson.isActive, lesson.statusMessage)
        lesson.revealHint("tanh-independent")
        lesson.save()
        let project = try XCTUnwrap(workspace.activeProjectID)
        let reopened = AffineLearningStore(workspace: workspace)
        reopened.reopenCurrentProject()
        XCTAssertEqual(workspace.activeProjectID, project)
        XCTAssertEqual(try reopened.draft?.resolvedLessonIdentity(), .tanh)
        XCTAssertEqual(reopened.mode, .practice)
        XCTAssertTrue(reopened.draft?.hintIDs.contains("tanh-independent") == true)
        XCTAssertFalse(reopened.hasCurrentCompletion)
    }
}
