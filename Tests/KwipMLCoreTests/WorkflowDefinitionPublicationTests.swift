import XCTest
@testable import KwipMLCore

final class WorkflowDefinitionPublicationTests: XCTestCase {
    private func fixture() throws -> (ProjectLibrary, WorkflowSourcePersistence) {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source))
        _ = try library.activateFileBackedWorkflow(id: source.projectID)
        return (library, try WorkflowSourcePersistence(library: library, projectID: source.projectID))
    }
    private func addingDefinition(_ source: WorkflowDocument) -> WorkflowDocument {
        var next = source, definition = source.definitions[0]
        definition.reference = .init("private-block-" + UUID().uuidString)
        definition.title = "Private reusable block"
        next.definitions.append(definition); next.revision += 1
        return next
    }
    func testAdditiveDefinitionPublicationRoundTripsAndUndoRedoRetainChildren() throws {
        let (library, storage) = try fixture(), before = storage.state
        let root = try XCTUnwrap(storage.fileURL), originalPin = try XCTUnwrap(before.flowDocument?.dependencies.first)
        let originalChild = root.deletingLastPathComponent().appendingPathComponent(originalPin.relativePath)
        let originalBytes = try Data(contentsOf: originalChild)
        let next = addingDefinition(before.document)
        _ = try storage.save(next)
        let published = storage.state
        XCTAssertEqual(try library.loadSource(id: next.projectID), .workflow(next))
        XCTAssertEqual(storage.state.flowDocument?.dependencies.first, originalPin)
        let added = try XCTUnwrap(storage.state.flowDocument?.dependencies.last)
        let addedURL = root.deletingLastPathComponent().appendingPathComponent(added.relativePath)
        XCTAssertEqual(addedURL.deletingLastPathComponent().path, root.deletingLastPathComponent().path)
        XCTAssertEqual(try FlowDocumentCodec.reference(FlowDocumentStore.readHead(at: addedURL)), added.revision)
        var undone = before.document; undone.revision = next.revision + 1
        _ = try storage.save(undone, restoring: before.flowDocument)
        XCTAssertTrue(FileManager.default.fileExists(atPath: addedURL.path))
        var redone = next; redone.revision = undone.revision + 1
        _ = try storage.save(redone, restoring: published.flowDocument)
        XCTAssertEqual(try library.loadSource(id: next.projectID), .workflow(redone))
        XCTAssertEqual(try Data(contentsOf: originalChild), originalBytes)
    }
    func testStaleRootAndExistingDefinitionMutationPublishNothing() throws {
        let (library, storage) = try fixture()
        let stale = try WorkflowSourcePersistence(library: library, projectID: storage.state.document.projectID)
        let root = try XCTUnwrap(storage.fileURL), folder = root.deletingLastPathComponent()
        var ordinary = storage.state.document; ordinary.name = "Saved first"; ordinary.revision += 1
        _ = try storage.save(ordinary)
        let files = try FileManager.default.contentsOfDirectory(atPath: folder.path)
        XCTAssertThrowsError(try stale.save(addingDefinition(stale.state.document)))
        var invalid = addingDefinition(storage.state.document); invalid.definitions[0].title = "Silent overwrite"
        XCTAssertThrowsError(try storage.save(invalid))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted(), files.sorted())
        XCTAssertEqual(try library.loadSource(id: ordinary.projectID), .workflow(ordinary))
    }
    func testNestedNewDefinitionRetainsArchivedExistingPinAndUnusedDefinitionCanBePruned() throws {
        let (_, storage) = try fixture(), before = storage.state
        let root = try XCTUnwrap(storage.fileURL), original = before.document.definitions[0]
        let dependency = try XCTUnwrap(before.flowDocument?.dependencies.first)
        let childURL = root.deletingLastPathComponent().appendingPathComponent(dependency.relativePath)
        var child = try FlowDocumentStore.readHead(at: childURL)
        child.revision += 1
        if case .modelBlock(var block) = child.payload { block.title = "New unadopted head"; child.payload = .modelBlock(block) }
        try FlowDocumentStore.save(child, at: childURL, expected: dependency.revision)
        var next = before.document
        let wrapper = CompositeDefinition(reference: .init("nested-new"), title: "Nested", body: .init(instances: [.init(id: "existing", definition: original.reference)]),
            inputs: original.inputs.map { .init($0.id, targets: [.init("existing", $0.id)]) },
            outputs: original.outputs.map { .init($0.id, source: .init("existing", $0.id)) })
        next.definitions.append(wrapper); next.revision += 1
        _ = try storage.save(next)
        let added = try XCTUnwrap(storage.state.flowDocument?.dependencies.last)
        let saved = try FlowDocumentStore.readHead(at: root.deletingLastPathComponent().appendingPathComponent(added.relativePath))
        XCTAssertEqual(saved.dependencies, [dependency])
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow(), next)
        XCTAssertEqual(try FlowDocumentStore.readHead(at: childURL), child)
        // Remove only the unused wrapper reference. Its immutable child remains for Undo.
        var pruned = next; pruned.definitions.removeLast(); pruned.revision += 1
        _ = try storage.save(pruned)
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.deletingLastPathComponent().appendingPathComponent(added.relativePath).path))
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow(), pruned)
    }
    func testInterruptedPublicationRetainsRecoverablePlanAndLeavesRootUnchanged() throws {
        let (_, storage) = try fixture(), before = storage.state
        let root = try XCTUnwrap(storage.fileURL), bytes = try Data(contentsOf: root)
        enum Interrupted: Error { case stop }
        storage.publicationDidWrite = { url in if url.pathExtension == "kwipflow" { throw Interrupted.stop } }
        let next = addingDefinition(before.document)
        XCTAssertThrowsError(try storage.save(next))
        XCTAssertEqual(storage.state, before); XCTAssertEqual(try Data(contentsOf: root), bytes)
        let files = try FileManager.default.contentsOfDirectory(at: root.deletingLastPathComponent(), includingPropertiesForKeys: nil)
        let receiptURL = try XCTUnwrap(files.first { $0.lastPathComponent.hasPrefix(".kwipflow-publication-") })
        let receipt = try JSONDecoder().decode(FlowDefinitionPublication.Receipt.self, from: Data(contentsOf: receiptURL))
        XCTAssertEqual(receipt.expected, try FlowDocumentCodec.reference(try XCTUnwrap(before.flowDocument)))
        XCTAssertEqual(receipt.children.count, 1)
        let path = try XCTUnwrap(receipt.children.keys.first)
        XCTAssertEqual(try FlowDocumentStore.readHead(at: root.deletingLastPathComponent().appendingPathComponent(path)), receipt.children[path])
        storage.publicationDidWrite = nil
        _ = try storage.save(next)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow(), next)
        XCTAssertTrue(FileManager.default.fileExists(atPath: receiptURL.path))
    }

    func testCompletedCanonicalPrivateForkKeepsOriginalFileAndIndependentReferences() throws {
        let (_, storage) = try fixture(), before = storage.state.document
        let node = try XCTUnwrap(before.nodes.first { if case .modelBlock = $0.payload { return true }; return false })
        let pin = try XCTUnwrap(storage.state.flowDocument?.dependencies.first), root = try XCTUnwrap(storage.fileURL)
        let originalURL = root.deletingLastPathComponent().appendingPathComponent(pin.relativePath)
        let original = try Data(contentsOf: originalURL)
        var draft = try WorkflowBlockEditing.beginDraft(before, root: node.address)
        var proposed = try XCTUnwrap(draft.drafts.first?.proposed); proposed.title = "Private fork"
        draft = try WorkflowBlockEditing.updateDraft(draft, proposed: proposed, expectedBaseRevision: before.revision, root: node.address)
        let completed = try WorkflowBlockEditing.applyDraft(draft, root: node.address, expectedBaseRevision: before.revision, registry: .builtIn())
        XCTAssertTrue(completed.drafts.isEmpty)
        _ = try storage.save(completed)
        XCTAssertEqual(try FlowDocumentStore.snapshot(at: root).workflow(), completed)
        XCTAssertNotEqual(completed.nodes.first { $0.address == node.address }?.payload, node.payload)
        XCTAssertEqual(completed.nodes.filter { $0.address != node.address }, before.nodes.filter { $0.address != node.address })
        XCTAssertEqual(try Data(contentsOf: originalURL), original)
    }
    func testExclusivePublicationRefusesExistingFileOrSymlinkWithoutChangingIt() throws {
        for symlink in [false, true] {
            let (_, storage) = try fixture(), current = try XCTUnwrap(storage.state.flowDocument)
            let root = try XCTUnwrap(storage.fileURL), id = UUID(), next = addingDefinition(storage.state.document)
            let plan = try FlowDefinitionPublication.plan(current: current, snapshot: FlowDocumentStore.snapshot(at: root), next: next, newID: { id })
            let child = root.deletingLastPathComponent().appendingPathComponent(id.uuidString + ".kwipflow")
            let sentinel = Data("Existing file must survive".utf8)
            let destination = root.deletingLastPathComponent().appendingPathComponent("unrelated.txt")
            if symlink {
                try sentinel.write(to: destination)
                try FileManager.default.createSymbolicLink(at: child, withDestinationURL: destination)
            } else { try sentinel.write(to: child) }
            let before = try Data(contentsOf: root)
            XCTAssertThrowsError(try FlowDefinitionPublication.publish(plan, beside: root, expected: FlowDocumentCodec.reference(current)))
            XCTAssertEqual(try Data(contentsOf: child), sentinel)
            XCTAssertEqual(try Data(contentsOf: root), before)
        }
    }

    func testParentReplacementWithIdenticalRootAndClosureCannotReceiveCommit() throws {
        let (_, storage) = try fixture(), before = storage.state
        let root = try XCTUnwrap(storage.fileURL), folder = root.deletingLastPathComponent()
        let moved = folder.deletingLastPathComponent().appendingPathComponent("moved-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: moved) }
        func tree(_ directory: URL) throws -> [String: Data] {
            let urls = try XCTUnwrap(FileManager.default.enumerator(at: directory, includingPropertiesForKeys: [.isRegularFileKey]))
            var result: [String: Data] = [:]
            for case let url as URL in urls where try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true {
                result[String(url.path.dropFirst(directory.path.count))] = try Data(contentsOf: url)
            }
            return result
        }
        var replacement: [String: Data]?
        storage.publicationDidWrite = { url in
            guard url.pathExtension == "kwipflow" else { return }
            try FileManager.default.moveItem(at: folder, to: moved)
            try FileManager.default.copyItem(at: moved, to: folder)
            replacement = try tree(folder)
        }
        XCTAssertThrowsError(try storage.save(addingDefinition(before.document)))
        XCTAssertEqual(storage.state, before)
        XCTAssertEqual(try tree(folder), try XCTUnwrap(replacement), "The replacement directory must receive no archive or head writes")
        XCTAssertEqual(try FlowDocumentStore.readHead(at: moved.appendingPathComponent(root.lastPathComponent)), before.flowDocument)
    }

}
