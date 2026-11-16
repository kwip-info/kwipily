import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FileInputViewPracticeTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func testPracticeUsesSourceFilesToViewAndNoCSVOrPrivateBindingInGraph() throws {
        let example = PreparedWorkflowExample.component(FileInputViewPracticeExample.exampleID)
        let document = try example.document(device: "cpu")
        let registry = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        let plan = try WorkflowDataPreparationPlan.compile(document, registry: registry)
        XCTAssertEqual(Set(plan.nodes.map(\.operatorID)), ["kwip.data.sourceFiles", "kwip.data.view"])
        XCTAssertEqual(document.nodes.count, 2)
        XCTAssertEqual(document.edges.count, 1)
        XCTAssertTrue(example.sourceResources.isEmpty)
        XCTAssertEqual(FileInputViewPracticeExample.resourceNames.count, 12)
        let encoded = String(decoding: try WorkflowCodec.encode(document), as: UTF8.self)
        XCTAssertFalse(encoded.contains("/Users/"))
        XCTAssertFalse(encoded.contains("sourceBindings"))
    }
    @MainActor func testPracticeLaunchCopiesOwnedCollectionAndPersistsDirectoryBinding() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(FileInputViewPracticeExample.exampleID, device: "cpu"), workspace.statusMessage)
        let session = try XCTUnwrap(workspace.workflowSession)
        let directory = try XCTUnwrap(session.bindings.path(for: FileInputViewPracticeExample.sourceAddress))
        XCTAssertTrue(directory.hasPrefix(root.appendingPathComponent("example-sources").path + "/"))
        let names = try FileManager.default.contentsOfDirectory(atPath: directory)
        XCTAssertEqual(Set(names), Set(FileInputViewPracticeExample.resourceNames))
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: session.owner.projectID), .workflow(session.owner.document))
        XCTAssertEqual(workspace.activeSection, "Build")
    }
    @MainActor func testMissingPracticeMemberDoesNotPersistPartiallyBoundProject() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertFalse(workspace.loadPreparedWorkflowExample(device: "cpu", kind: .component(FileInputViewPracticeExample.exampleID), resourceURL: { name in
            if name.hasSuffix(".docx") { return nil }
            return self.repository.appendingPathComponent("Sources/KwipMLApp/Resources/FileInputPractice").appendingPathComponent(name)
        }))
        XCTAssertNil(workspace.workflowSession)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("example-sources").path))
    }
    @MainActor func testLessonIsExactlyRoutedAndDoesNotAwardAssessmentCompletion() {
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "kwip.data.sourceFiles"), FileInputViewPracticeExample.lessonID)
        XCTAssertTrue(LearnCatalogContent.items.contains { $0.id == FileInputViewPracticeExample.lessonID })
        XCTAssertFalse(FileInputLessonContent.isAssessed)
        XCTAssertTrue(FileInputLessonContent.boundary.contains("no scored pass"))
        XCTAssertTrue(FileInputLessonContent.boundary.contains("not an ML tensor"))
    }
}
