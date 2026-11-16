import XCTest
@testable import KwipMLCore

final class FlowDocumentRevisionTests: XCTestCase {
    func testSavingChildKeepsParentsPinnedUntilExplicitAdoption() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        let plan = try FlowDocumentMigration.plan(source)
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let dependency = try XCTUnwrap(plan.root.dependencies.first)
        let childURL = root.deletingLastPathComponent().appendingPathComponent(dependency.relativePath)
        var child = try FlowDocumentCodec.decode(Data(contentsOf: childURL))
        let old = try FlowDocumentCodec.reference(child)
        child.revision += 1
        if case .modelBlock(var block) = child.payload { block.title = "Edited child"; child.payload = .modelBlock(block) }
        try FlowDocumentStore.save(child, at: childURL, expected: old)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow(), source)
        var parent = plan.root
        parent.revision += 1
        if case .workflow(var workflow) = parent.payload { workflow.revision = parent.revision; parent.payload = .workflow(workflow) }
        parent.dependencies[0].revision = try FlowDocumentCodec.reference(child)
        try FlowDocumentStore.save(parent, at: root, expected: FlowDocumentCodec.reference(plan.root))
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow().definitions.first?.title, "Edited child")
        XCTAssertThrowsError(try FlowDocumentStore.save(child, at: childURL, expected: old))
        XCTAssertEqual(try FlowDocumentCodec.decode(Data(contentsOf: childURL)), child)
    }
    func testRefusesWrongIdentityAndRevisionWithoutChangingFile() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedMaxPool3dClassifier())
        let root = try FlowDocumentStore.commitMigration(plan, to: directory.appendingPathComponent("Project"))
        let bytes = try Data(contentsOf: root)
        let expected = try FlowDocumentCodec.reference(plan.root)
        XCTAssertThrowsError(try FlowDocumentStore.save(plan.root, at: root, expected: expected))
        var wrong = plan.root; wrong.id = UUID(); wrong.revision += 1
        XCTAssertThrowsError(try FlowDocumentStore.save(wrong, at: root, expected: expected))
        XCTAssertEqual(try Data(contentsOf: root), bytes)
    }
    func testDuplicateDependencyPathUsesItsOwnPinnedArchive() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let plan = try FlowDocumentMigration.plan(WorkflowExamples.preparedSeparableConv2dClassifier())
        let shared = try XCTUnwrap(plan.subflows.values.first)
        guard case .modelBlock(var wrapperDefinition) = shared.payload else { return XCTFail("Expected block") }
        let sharedDefinition = wrapperDefinition.reference
        wrapperDefinition.reference.id = "wrapper"
        let pin = try FlowDocumentCodec.reference(shared)
        let wrapper = FlowDocument(payload: .modelBlock(wrapperDefinition), dependencies: [
            .init(definition: sharedDefinition, revision: pin, relativePath: "SharedCopy.kwipflow")])
        var root = plan.root
        root.dependencies = [
            .init(definition: sharedDefinition, revision: pin, relativePath: "Shared.kwipflow"),
            .init(definition: wrapperDefinition.reference, revision: try FlowDocumentCodec.reference(wrapper), relativePath: "Wrapper.kwipflow")]
        let rootURL = directory.appendingPathComponent("Main.kwipflow")
        let copyURL = directory.appendingPathComponent("SharedCopy.kwipflow")
        for (name, document) in [("Main.kwipflow", root), ("Shared.kwipflow", shared),
                                 ("SharedCopy.kwipflow", shared), ("Wrapper.kwipflow", wrapper)] {
            try FlowDocumentCodec.encode(document).write(to: directory.appendingPathComponent(name))
        }
        let original = try FlowDocumentStore.snapshot(at: rootURL)
        var advanced = shared
        advanced.revision += 1
        try FlowDocumentStore.save(advanced, at: copyURL, expected: pin)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: rootURL), original)
        // A duplicate location must still prove its pin; the first valid copy cannot hide corruption.
        let archive = directory.appendingPathComponent(".revisions")
            .appendingPathComponent(pin.documentID.uuidString).appendingPathComponent(pin.sha256 + ".kwipflow")
        try Data("corrupt".utf8).write(to: archive)
        XCTAssertThrowsError(try FlowDocumentStore.snapshot(at: rootURL))
    }

}
