import XCTest
import KwipMLCore
@testable import KwipMLApp

final class FilterPracticeTests: XCTestCase {
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func testPracticeCompilesTypedSourceAndBothStableOutputBranches() throws {
        let example = PreparedWorkflowExample.component(FilterPracticeExample.exampleID)
        let document = try example.document(device: "cpu")
        let registry = try DataFlowOperatorRegistry.load(from: repository.appendingPathComponent("runner/dataflow-operators.json"))
        let plan = try WorkflowDataPreparationPlan.compile(document, registry: registry)
        XCTAssertEqual(Set(plan.nodes.map(\.operatorID)), ["kwip.data.sourceCSV", "kwip.data.filterRows", "kwip.data.view"])
        XCTAssertEqual(document.nodes.count, 4)
        XCTAssertEqual(document.edges.count, 3)
        XCTAssertEqual(example.sourceResources.map(\.resource), [FilterPracticeExample.observationsResource])
        XCTAssertEqual(example.observationsResource, FilterPracticeExample.observationsResource)
        let outputs = document.edges.filter { $0.from.address == FilterPracticeExample.filterAddress }
        XCTAssertEqual(Set(outputs.map { $0.from.port }), ["matching", "nonmatching"])
        XCTAssertEqual(Set(outputs.map { $0.to.address }), [FilterPracticeExample.matchingAddress, FilterPracticeExample.nonmatchingAddress])
        let source = try XCTUnwrap(document.nodes.first { $0.address == WorkflowExamples.sourceAddress })
        guard case .dataOperator(_, let sourceConfig) = source.payload else { return XCTFail("Missing Table Input") }
        XCTAssertEqual(sourceConfig["schemaOverrides"], .object(["amount": .string("Float64")]))
        XCTAssertEqual(sourceConfig["emptyField"], .string("null"))
        let tool = try XCTUnwrap(document.nodes.first { $0.address == FilterPracticeExample.filterAddress })
        guard case .dataOperator(_, let config) = tool.payload else { return XCTFail("Missing Filter") }
        XCTAssertEqual(config, FilterPracticeExample.configuration)
        XCTAssertEqual(config["mode"], .string("simple"))
        XCTAssertEqual(config["nullPolicy"], .string("nonmatching"))
        let encoded = String(decoding: try WorkflowCodec.encode(document), as: UTF8.self)
        XCTAssertFalse(encoded.contains("/Users/"))
        XCTAssertFalse(encoded.contains("sourceBindings"))
    }

    @MainActor func testPracticeLaunchCopiesPublicRowsIncludingMissingObservationAndPersistsGraph() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(FilterPracticeExample.exampleID, device: "cpu"), workspace.statusMessage)
        let host = try XCTUnwrap(workspace.workflowSession)
        let source = try XCTUnwrap(host.bindings.path(for: WorkflowExamples.sourceAddress))
        XCTAssertTrue(source.hasPrefix(root.appendingPathComponent("example-sources").path + "/"))
        let csv = try String(contentsOfFile: source, encoding: .utf8)
        XCTAssertEqual(csv.split(whereSeparator: \.isNewline).count, 9)
        XCTAssertTrue(csv.contains("A04,,ready,unknown amount"))
        XCTAssertTrue(csv.contains("A03,18,READY"))
        XCTAssertEqual(try workspace.projectLibrary.loadSource(id: host.owner.projectID), .workflow(host.owner.document))
        XCTAssertEqual(workspace.activeSection, "Build")
    }

    @MainActor func testCurrentLessonRoutesDoNotOfferRetiredToolDestinations() {
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "kwip.data.filterRows"), FilterPracticeExample.lessonID)
        XCTAssertTrue(LearnCatalogContent.items.contains { $0.id == FilterPracticeExample.lessonID })
        XCTAssertEqual(ComponentLessonDestination.entryID(type: "kwip.data.filter"), "data-preparation-course")
        XCTAssertNil(ComponentLessonDestination.entryID(type: "kwip.data.filterExpression"))
        XCTAssertNil(ComponentLessonDestination.entryID(type: "kwip.data.regexMatch"))
        XCTAssertFalse(FilterLessonContent.isAssessed)
        XCTAssertTrue(FilterLessonContent.boundary.contains("no scored pass"))
        XCTAssertTrue(FilterLessonContent.boundary.contains("does not create training"))
        XCTAssertTrue(FilterLessonContent.steps.joined().contains("Null is never automatically treated as zero"))
        let topic = DataPreparationLessonContent.topics.first { $0.id == "clean-and-check" }
        XCTAssertTrue(topic?.operators.contains("kwip.data.filterRows") == true)
        XCTAssertTrue(topic?.exampleIDs.contains(FilterPracticeExample.exampleID) == true)
    }
}
