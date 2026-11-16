import XCTest
import Foundation
import KwipMLCore
@testable import KwipMLApp

final class WorkspaceTests: XCTestCase {
    func testRenameDoesNotStaleComputationButShapeEditDoes() async throws {
        try await MainActor.run {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
            store.loadExample(); store.resultSignature = store.semanticSignature(store.graph)
            store.selectNode("branch-a"); store.renameNode("User label")
            XCTAssertFalse(store.resultsStale)
            store.setConfig("out_features", value: 10)
            XCTAssertTrue(store.resultsStale)
            XCTAssertFalse(store.validation.isValid)
            store.undo(); XCTAssertFalse(store.resultsStale)
            XCTAssertEqual(store.graph.nodes.first(where: {$0.id == "branch-a"})?.name, "User label")
        }
    }
    func testInvalidNumericEditsAndBlockLimitDoNotDiscardSavedProject() async throws {
        try await MainActor.run {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
            store.loadExample(); store.selectNode("branch-a")
            store.setConfig("out_features", value: .infinity)
            store.setConfig("out_features", value: 1e100)
            XCTAssertEqual(store.selectedNode?.config["out_features"], 8)
            XCTAssertNil(store.saveError)
            store.graph.nodes = (0..<128).map { GraphNode(id: "n\($0)", type: "relu", name: "Node \($0)") }
            store.addNode(type: "relu", x: 10, y: 10)
            XCTAssertEqual(store.graph.nodes.count, 128)
        }
    }
    func testFailedSaveBlocksProjectSwitch() async throws {
        try await MainActor.run {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            // A regular file at .local makes the library unavailable without permission changes.
            try Data("fixture".utf8).write(to: root.appendingPathComponent(".local"))
            let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
            store.selectNode("branch-a"); store.renameNode("Unsaved work")
            XCTAssertTrue(store.hasUnsavedChanges); XCTAssertNotNil(store.saveError)
            store.newProject()
            XCTAssertEqual(store.graph.nodes.first(where: {$0.id == "branch-a"})?.name, "Unsaved work")
            XCTAssertTrue(store.hasUnsavedChanges)
        }
    }
    func testCourseBuildsExactlyTheReferenceFlow() async throws {
        try await MainActor.run {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let store = WorkspaceStore(rootOverride: repo,storageOverride:root, detectCapabilities: false)
            store.startCourse()
            let host=try XCTUnwrap(store.workflowSession)
            XCTAssertEqual(host.owner.document.nodes.count,1)
            for count in 2...(try WorkflowGuidedBuildPlan.parallelClassifier().steps.count) {store.advanceCourse();XCTAssertEqual(host.owner.document.nodes.count,count)}
            let plan=try WorkflowCompiler.compile(host.owner.document,train:WorkflowExamples.trainAddress,
                modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry,contractVersion:15)
            XCTAssertEqual(try GraphSemantics.canonicalData(plan.model.graph,registry:store.registry),
                try GraphSemantics.canonicalData(ExampleFlows.parallelClassifier,registry:store.registry))
            XCTAssertEqual(plan.model.graph.edges.count,8)
            let completed=host.owner.document
            store.advanceCourse();XCTAssertEqual(host.owner.document,completed)

        }
    }
    func testCSVContentChangeMarksPreviousResultStale() async throws {
        try await MainActor.run {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let csv = root.appendingPathComponent("data.csv")
            try Data("x,y,label\n1,2,0\n".utf8).write(to: csv)
            let store = WorkspaceStore(rootOverride: root, detectCapabilities: false)
            store.setExecution { $0.datasetCSV = csv.path; $0.featureColumns = ["x","y"]; $0.targetColumn = "label" }
            store.resultSignature = store.semanticSignature(store.graph); store.lastRunSourceHash = store.sourceDigest(csv.path)
            XCTAssertFalse(store.resultsStale)
            try Data("x,y,label\n3,4,1\n".utf8).write(to: csv)
            XCTAssertTrue(store.resultsStale)
        }
    }
    func testCanvasFormatsUntrustedNumericValuesWithoutTrapping() {
        XCTAssertEqual(CanvasDisplayValue.dimension(8), "8")
        XCTAssertFalse(CanvasDisplayValue.dimension(1e100).isEmpty)
        XCTAssertEqual(CanvasDisplayValue.dimension(.infinity), "Invalid")
        XCTAssertEqual(CanvasDisplayValue.dimension(.nan), "Invalid")
    }
}
