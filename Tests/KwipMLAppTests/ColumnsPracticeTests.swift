import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ColumnsPracticeTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func testPracticeCompilesOneGridAndRetainsAllFourAuthoredChoices() throws {
        let example = PreparedWorkflowExample.component(ColumnsPracticeExample.exampleID)
        let document = try example.document(device: "cpu")
        let registry = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        let plan = try WorkflowDataPreparationPlan.compile(document, registry: registry)
        XCTAssertEqual(Set(plan.nodes.map(\.operatorID)), ["kwip.data.sourceCSV", "kwip.data.columns", "kwip.data.view"])
        XCTAssertEqual(document.nodes.count, 3)
        XCTAssertEqual(document.edges.count, 2)
        XCTAssertEqual(example.sourceResources.map(\.resource), [ColumnsPracticeExample.observationsResource])
        let node = try XCTUnwrap(document.nodes.first { $0.address == ColumnsPracticeExample.columnsAddress })
        guard case .dataOperator(_, let config) = node.payload,
              case .array(let rows) = config["rows"] else { return XCTFail("Practice must use the actual grid configuration") }
        let values = try rows.map { row -> [String: JSONValue] in
            guard case .object(let object) = row else { throw DataFlowContractError("Expected grid row") }
            return object
        }
        XCTAssertEqual(values.map { $0["source"] }, [.string("score_text"), .string("person_id"), .string("active_text"), .string("notes")])
        XCTAssertEqual(values.map { $0["included"] }, [.bool(true), .bool(true), .bool(true), .bool(false)])
        XCTAssertEqual(values.map { $0["outputName"] }, [.string("score"), .string("person_id"), .string("active"), .string("notes")])
        XCTAssertEqual(values.map { $0["targetType"] }, [.string("float64"), .string("keep"), .string("bool"), .string("keep")])
        let text = String(decoding: try WorkflowCodec.encode(document), as: UTF8.self)
        XCTAssertFalse(text.contains("/Users/"))
        XCTAssertFalse(text.contains("sourceBindings"))
    }

    @MainActor func testPracticeLaunchOwnsSyntheticSourceAndPersistsEditableGraph() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(ColumnsPracticeExample.exampleID, device: "cpu"), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let source = try XCTUnwrap(host.bindings.path(for: WorkflowExamples.sourceAddress))
        XCTAssertTrue(source.hasPrefix(root.appendingPathComponent("example-sources").path + "/"))
        let csv = try String(contentsOfFile: source, encoding: .utf8)
        XCTAssertEqual(csv.split(whereSeparator: \.isNewline).count, 7)
        XCTAssertTrue(csv.contains("001,12.5,true"))
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: host.owner.projectID), .workflow(host.owner.document))
        XCTAssertEqual(workspace.activeSection, "Build")
    }

    @MainActor func testLessonIsDiscoverableWithoutAwardingLegacyOrCurrentMastery() {
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "kwip.data.columns"), ColumnsPracticeExample.lessonID)
        XCTAssertTrue(LearnCatalogContent.items.contains { $0.id == ColumnsPracticeExample.lessonID })
        XCTAssertFalse(ColumnsLessonContent.isAssessed)
        XCTAssertTrue(ColumnsLessonContent.boundary.contains("no scored pass"))
        XCTAssertTrue(ColumnsLessonContent.boundary.contains("does not filter or reorder rows"))
        XCTAssertTrue(ColumnsLessonContent.boundary.contains("Scalar values cannot be cast to Image"))
        XCTAssertTrue(ColumnsLessonContent.boundary.contains("Keep type also retains an incoming image column"))
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "kwip.data.selectRename"), "data-preparation-course")
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "kwip.data.cast"), "data-preparation-course")
    }
}
