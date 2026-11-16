import XCTest
import Foundation
import KwipMLCore
@testable import KwipMLApp

final class ProjectRestoreTests: XCTestCase {
    @MainActor private func withFixture(_ action: (WorkspaceStore, URL) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
        let id=UUID().uuidString;store.activeProjectID=id
        store.rememberLastRun(root.path)
        defer {
            try? FileManager.default.removeItem(at:root)
            var paths=UserDefaults.standard.dictionary(forKey:"lastRunByProject") as? [String:String] ?? [:]
            paths.removeValue(forKey:id);UserDefaults.standard.set(paths,forKey:"lastRunByProject")
        }
        let graph=try JSONSerialization.jsonObject(with:JSONEncoder().encode(store.graph))
        let manifest:[String:Any]=["status":"completed","graph":graph,"runtime":["device":"cpu"]]
        try JSONSerialization.data(withJSONObject:manifest).write(to:root.appendingPathComponent("manifest.json"))
        try Data("checkpoint-fixture".utf8).write(to:root.appendingPathComponent("checkpoint.pt"))
        try Data("class Model: pass\n".utf8).write(to:root.appendingPathComponent("model.py"))
        try JSONSerialization.data(withJSONObject:[["epoch":1,"trainLoss":0.6,"validationLoss":0.7,"accuracy":0.8]]).write(to:root.appendingPathComponent("metrics.json"))
        try action(store,root)
    }
    func testCompleteRunRestoresWithSignatureAndMetrics() async throws {
        try await MainActor.run { try withFixture { store,_ in
            store.restoreLastRun()
            XCTAssertTrue(store.lastCompleted);XCTAssertNotNil(store.outputDirectory)
            XCTAssertEqual(store.resultSignature,store.semanticSignature(store.graph))
            XCTAssertEqual(store.metrics.count,1);XCTAssertFalse(store.generatedCode.isEmpty)
        } }
    }
    func testMissingArtifactsNeverPublishRestoredCompletion() async throws {
        try await MainActor.run {
            for file in ["checkpoint.pt","model.py","metrics.json"] {
                try withFixture { store,root in
                    try FileManager.default.removeItem(at:root.appendingPathComponent(file))
                    store.restoreLastRun()
                    XCTAssertFalse(store.lastCompleted,file);XCTAssertNil(store.outputDirectory,file);XCTAssertNil(store.resultSignature,file)
                }
            }
        }
    }
    func testCorruptGraphAndMetricsNeverPublishCompletion() async throws {
        try await MainActor.run {
            try withFixture { store,root in
                try JSONSerialization.data(withJSONObject:["status":"completed","graph":["wrong":true]]).write(to:root.appendingPathComponent("manifest.json"))
                store.restoreLastRun();XCTAssertFalse(store.lastCompleted);XCTAssertNil(store.outputDirectory)
            }
            try withFixture { store,root in
                try Data("[]".utf8).write(to:root.appendingPathComponent("metrics.json"))
                store.restoreLastRun();XCTAssertFalse(store.lastCompleted)
            }
        }
    }
    func testCourseCannotChangeStateDuringPreview() async throws {
        try await MainActor.run { try withFixture { store,_ in
            let before=store.graph;store.isPreviewing=true;store.showPreview=true
            store.startCourse()
            XCTAssertEqual(store.graph,before);XCTAssertNil(store.tutorialStep);XCTAssertTrue(store.showPreview)
        } }
    }
}
