import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModuleCompositionPortDescriptorTests: XCTestCase {
    private var repository: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private func assertConnectedShape(_ descriptor: CanvasPortDescriptor, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(descriptor.kind, .tensor, file: file, line: line)
        XCTAssertEqual(descriptor.direction, .input, file: file, line: line)
        XCTAssertTrue(descriptor.declared.contains("float32"), file: file, line: line)
        XCTAssertTrue(descriptor.declared.contains("Shape follows connected input"), file: file, line: line)
        XCTAssertFalse(descriptor.declared.contains("[batch, 2]"), "Legacy classification width must not describe a general model boundary", file: file, line: line)
        XCTAssertTrue(descriptor.measured.isEmpty, "An authored declaration is not a measured result", file: file, line: line)
    }
    func testPreparedLessonOutputsWithOneAndThreeColumnsDoNotClaimBinaryClassificationWidth() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        for (task, width) in [(ModuleCompositionLearningTask.orderName, 3), (.buildReuse, 1)] {
            let project = UUID().uuidString
            let registry = try ComponentRegistry.builtIn()
            let drafts = TensorLearningDraftSession(repository: .init(directory: directory))
            XCTAssertTrue(drafts.create(projectID: project, operation: task.rawValue, mode: .guided))
            let model = try ModuleCompositionLearningDraft.model(projectID: project, task: task, prepared: true)
            XCTAssertTrue(drafts.update { $0.compositionModel = model })
            let owner = try ModuleCompositionLearningCanvasAuthority(session: drafts, registry: registry, isCurrent: { true }, isBusy: { false })
            let adapter = try ModelCanvasAdapter(owner: owner)
            let expanded = try ModelExecutionSnapshotBuilder.build(.blockModel(model), registry: registry, context: .explicitObjective)
            let validation = GraphValidator.validate(expanded.graph, registry: registry, context: .explicitObjective)
            XCTAssertTrue(validation.isValid, "Prepared construction must remain valid at its real output width")
            let output = try XCTUnwrap(expanded.graph.nodes.first { $0.type == "output" })
            let inferred = validation.tensorPortDescriptor(nodeID: output.id, port: "in", direction: .input, graph: expanded.graph, registry: registry)
            let incoming = try XCTUnwrap(expanded.graph.edges.first { $0.target == output.id && $0.targetPort == "in" })
            let tensor = try XCTUnwrap(validation.outputTensorContracts[.init(incoming.source, incoming.sourcePort)])
            XCTAssertEqual(tensor.layout.sample.dimensions, [width], "Independent numeric inference establishes the fixture's actual width")
            XCTAssertTrue(inferred.declared.contains("[N, \(width)]"), "Inferred layouts use N for the observation count")
            assertConnectedShape(adapter.canvasPortDescriptor(nodeID: "output", port: "in", direction: .input))
        }
    }
    func testUnfinishedOutputRemainsTypedWithoutInventingConnectedOrMeasuredDimensions() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let drafts = TensorLearningDraftSession(repository: .init(directory: directory))
        XCTAssertTrue(drafts.create(projectID: UUID().uuidString, operation: ModuleCompositionLearningTask.orderName.rawValue, mode: .guided))
        let owner = try ModuleCompositionLearningCanvasAuthority(session: drafts, registry: .builtIn(), isCurrent: { true }, isBusy: { false })
        let adapter = try ModelCanvasAdapter(owner: owner)
        XCTAssertTrue(adapter.graph.edges.isEmpty)
        assertConnectedShape(adapter.canvasPortDescriptor(nodeID: "output", port: "in", direction: .input))
        XCTAssertEqual(adapter.canvasPortDescriptor(nodeID: "output", port: "missing", direction: .input).kind, .unknown)
    }
    func testOrdinaryClassificationWorkspaceKeepsItsTwoColumnContract() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: directory, detectCapabilities: false)
        workspace.loadExample()
        let output = try XCTUnwrap(workspace.graph.nodes.first { $0.type == "output" })
        let descriptor = workspace.canvasPortDescriptor(nodeID: output.id, port: "in", direction: .input)
        XCTAssertEqual(descriptor.kind, .tensor)
        XCTAssertTrue(descriptor.declared.contains("[N, 2]"))
        XCTAssertTrue(descriptor.declared.contains("float32"))
        XCTAssertTrue(descriptor.measured.isEmpty)
    }
}
