import XCTest
import Darwin
@testable import KwipMLCore

final class FlowPublicationRetentionTests: XCTestCase {
    func testExistingEvidenceConsumesBudgetAndSymlinksAreRefused() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let file = directory.appendingPathComponent(".kwipflow-draft-retained.kwipflow")
        try Data(repeating: 7, count: 12).write(to: file)
        let fd = open(directory.path, O_RDONLY | O_DIRECTORY)
        XCTAssertGreaterThanOrEqual(fd, 0); defer { close(fd) }
        let policy = FlowPublicationRetention(maximumFiles: 2, maximumBytes: 16)
        XCTAssertNoThrow(try policy.check(directory: fd, incoming: [Data(repeating: 1, count: 4)]))
        XCTAssertThrowsError(try policy.check(directory: fd, incoming: [Data(repeating: 1, count: 5)]))
        XCTAssertThrowsError(try policy.check(directory: fd, incoming: [Data(), Data()]))
        try FileManager.default.removeItem(at: file)
        try FileManager.default.createSymbolicLink(at: file, withDestinationURL: directory.appendingPathComponent("outside"))
        XCTAssertThrowsError(try policy.check(directory: fd, incoming: []))
    }
    func testTwoRootsCannotInterleaveFolderRetentionAdmission() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source)); _ = try library.activateFileBackedWorkflow(id: source.projectID)
        let first = try WorkflowSourcePersistence(library: library, projectID: source.projectID)
        let firstURL = try XCTUnwrap(first.fileURL), folder = firstURL.deletingLastPathComponent()
        var secondSource = source; secondSource.projectID = UUID().uuidString
        var portable = secondSource; portable.definitions = []
        let secondRoot = FlowDocument(id: UUID(uuidString: secondSource.projectID)!, payload: .workflow(portable),
                                     dependencies: try XCTUnwrap(first.state.flowDocument).dependencies)
        let secondURL = folder.appendingPathComponent("Second.kwipflow")
        try FlowDocumentCodec.encode(secondRoot).write(to: secondURL)
        let secondLibrary = ProjectLibrary(directory: directory.appendingPathComponent("second-private"), externalFlowRoot: secondURL, projectID: secondSource.projectID)
        let second = try WorkflowSourcePersistence(library: secondLibrary, projectID: secondSource.projectID)
        first.publicationRetention = .init(maximumFiles: 3, maximumBytes: 1_000_000)
        second.publicationRetention = first.publicationRetention
        func begin(_ value: WorkflowDocument) throws -> WorkflowDocument {
            var result = try WorkflowBlockEditing.beginDraft(value, root: .init(domain: .model, localID: "convolution"))
            result.revision += 1; result.drafts[0].baseRevision = result.revision
            return result
        }
        let secondBefore = second.state, secondBytes = try Data(contentsOf: secondURL)
        var interleaved = false
        first.publicationDidWrite = { url in
            guard url.pathExtension == "json" else { return }
            interleaved = true
            let names = try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted()
            XCTAssertThrowsError(try second.save(begin(secondSource)))
            XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).sorted(), names)
            XCTAssertEqual(try Data(contentsOf: secondURL), secondBytes)
            XCTAssertEqual(second.state, secondBefore)
        }
        // Establish the second root's ordinary CAS lock before the observed interleaving.
        try Data().write(to: folder.appendingPathComponent(".Second.kwipflow.lock"))
        _ = try first.save(begin(source))
        XCTAssertTrue(interleaved)
    }
    func testRetentionRefusalPreservesRootAndEvidenceAndAllowsDiscardWithoutPublication() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let library = ProjectLibrary(directory: directory)
        let source = try WorkflowExamples.preparedSeparableConv2dClassifier()
        _ = try library.saveSource(.workflow(source)); _ = try library.activateFileBackedWorkflow(id: source.projectID)
        let storage = try WorkflowSourcePersistence(library: library, projectID: source.projectID)
        let root = WorkflowNodeAddress(domain: .model, localID: "convolution")
        var begun = try WorkflowBlockEditing.beginDraft(source, root: root)
        begun.revision += 1; begun.drafts[0].baseRevision = begun.revision
        _ = try storage.save(begun)
        let before = storage.state
        let url = try XCTUnwrap(storage.fileURL), bytes = try Data(contentsOf: url)
        storage.publicationRetention = .init(maximumFiles: 0, maximumBytes: 0)
        var changed = begun
        changed.revision += 1; changed.drafts[0].baseRevision = changed.revision
        changed.drafts[0].proposed.title = "Another edit"
        XCTAssertThrowsError(try storage.save(changed)) { error in
            XCTAssertTrue(error.localizedDescription.contains("Inspect"))
        }
        XCTAssertEqual(storage.state, before)
        XCTAssertEqual(try Data(contentsOf: url), bytes)
        var discarded = begun; discarded.drafts = []; discarded.revision += 1
        _ = try storage.save(discarded)
        XCTAssertEqual(try library.loadSource(id: source.projectID), .workflow(discarded))
    }
}
