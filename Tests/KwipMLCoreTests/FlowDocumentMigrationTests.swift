import XCTest
@testable import KwipMLCore

final class FlowDocumentMigrationTests: XCTestCase {
    func testExistingReusableWorkflowRoundTripsWithoutChangingExecution() throws {
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier(device: "cpu")
        let before = source
        let plan = try FlowDocumentMigration.plan(source)
        XCTAssertEqual(source, before)
        XCTAssertEqual(plan.subflows.count, source.definitions.count)
        let restored = try plan.snapshot().workflow()
        XCTAssertEqual(restored, source)
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry = try ComponentRegistry.builtIn()
        let data = try DataFlowOperatorRegistry.load(from: root.appendingPathComponent("runner/dataflow-operators.json"))
        XCTAssertEqual(try WorkflowCompiler.compile(restored, train: WorkflowExamples.trainAddress, modelRegistry: registry, dataRegistry: data),
                       try WorkflowCompiler.compile(source, train: WorkflowExamples.trainAddress, modelRegistry: registry, dataRegistry: data))
        for dependency in plan.root.dependencies {
            XCTAssertNotNil(plan.subflows[dependency.relativePath])
        }
    }
    func testFlatWorkflowNeedsNoSubflowFiles() throws {
        let source = try WorkflowExamples.preparedMaxPool3dClassifier()
        let plan = try FlowDocumentMigration.plan(source)
        XCTAssertTrue(plan.subflows.isEmpty)
        XCTAssertEqual(try plan.snapshot().workflow(), source)
    }
    func testIdentityCollisionAndUnresolvedReferenceCannotProducePartialPlan() throws {
        var source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        XCTAssertThrowsError(try FlowDocumentMigration.plan(source, newID: { UUID(uuidString: source.projectID)! }))
        source.definitions.removeAll()
        XCTAssertThrowsError(try FlowDocumentMigration.plan(source))
    }
    func testPendingDraftMigratesToIndependentFileWithoutApplyingIt() throws {
        var source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        let definition = try XCTUnwrap(source.definitions.first)
        let instance = try XCTUnwrap(source.nodes.first { if case .modelBlock = $0.payload { return true }; return false })
        var proposed = definition
        proposed.title = "Unfinished edits"
        proposed.body.edges.removeAll() // An intentionally incomplete construction remains an editable draft.
        source.drafts = [.init(projectID: source.projectID, instancePath: [instance.address.localID], baseReference: definition.reference, baseRevision: source.revision, proposed: proposed)]
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let original = try WorkflowCodec.encode(source)
        let originalURL = directory.appendingPathComponent("Original.kwipml")
        try original.write(to: originalURL)
        let plan = try FlowDocumentMigration.plan(source)
        XCTAssertEqual(plan.subflows.count, source.definitions.count + 1)
        XCTAssertEqual(try plan.snapshot().workflow(), source)
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Converted"))
        let restored = try FlowDocumentStore.snapshot(at: root).workflow()
        XCTAssertEqual(restored, source)
        XCTAssertEqual(restored.definitions, [definition], "Proposed draft must not replace the published definition")
        XCTAssertEqual(try Data(contentsOf: originalURL), original)
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertThrowsError(try WorkflowCompiler.compile(restored, train: WorkflowExamples.trainAddress,
            modelRegistry: .builtIn(), dataRegistry: .load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))))
    }
    func testUnresolvedProposedReferencesStayDraftsAndMissingDraftRefusesRestore() throws {
        var source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        var proposed = try XCTUnwrap(source.definitions.first)
        let instance = try XCTUnwrap(source.nodes.first { if case .modelBlock = $0.payload { return true }; return false })
        proposed.body.instances.append(.init(id:"unfinished",definition:.init("not-yet-defined")))
        source.drafts = [.init(projectID:source.projectID,instancePath:[instance.address.localID],
            baseReference:proposed.reference,baseRevision:source.revision,proposed:proposed)]
        let plan = try FlowDocumentMigration.plan(source)
        XCTAssertEqual(plan.root.schemaVersion,2)
        XCTAssertEqual(plan.root.draftDependencies.count,1)
        let snapshot = try plan.snapshot()
        XCTAssertEqual(try snapshot.workflow(),source)
        var missing = snapshot
        missing.documents.removeAll { if case .modelBlockDraft = $0.payload { return true }; return false }
        XCTAssertThrowsError(try missing.workflow())
        var wrongParent = snapshot
        let index = try XCTUnwrap(wrongParent.documents.firstIndex { if case .modelBlockDraft = $0.payload { return true }; return false })
        if case .modelBlockDraft(var draft) = wrongParent.documents[index].payload {
            draft.baseRevision += 1
            wrongParent.documents[index].payload = .modelBlockDraft(draft)
        }
        let rootIndex = try XCTUnwrap(wrongParent.documents.firstIndex { if case .workflow = $0.payload { return true }; return false })
        wrongParent.documents[rootIndex].draftDependencies[0].revision = try FlowDocumentCodec.reference(wrongParent.documents[index])
        wrongParent.root = try FlowDocumentCodec.reference(wrongParent.documents[rootIndex])
        XCTAssertThrowsError(try wrongParent.workflow(), "Correct hashes cannot bypass parent/base-revision validation")
    }

}
