import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class SourceInspectionHostPolicyTests: XCTestCase {
    private enum Host: CaseIterable { case legacy, preparation, canvas }
    private struct Fixture {
        let workspace: WorkspaceStore
        let root: URL
        let start: () -> Void
        let busy: () -> Bool
        let message: () -> String
        let columns: () -> [String]
        let inject: (@escaping SourceInspectionExecutor) -> Void
    }
    private func fixture(_ host: Host, project: Bool = true) throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let workspace = WorkspaceStore(rootOverride: repo, storageOverride: root, detectCapabilities: false)
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        workspace.activeProjectID = project ? UUID().uuidString : nil
        let csv = root.appendingPathComponent("source.csv")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("a,b\n1,2\n".utf8).write(to: csv)
        switch host {
        case .legacy:
            workspace.graph.execution = ExecutionSettings(datasetCSV: csv.path)
            return Fixture(workspace: workspace, root: root, start: { workspace.inspectData() },
                busy: { workspace.isInspectingData }, message: { workspace.dataError ?? "" },
                columns: { workspace.dataColumns }, inject: { workspace.dataInspectionExecute = $0 })
        case .preparation:
            let editor = DataFlowWorkspaceStore(workspace: workspace)
            editor.project.sourceBindings["source"] = csv.path
            return Fixture(workspace: workspace, root: root, start: { editor.inspectSource(id: "source") },
                busy: { editor.isInspectingSource }, message: { editor.statusMessage },
                columns: { editor.sourceHeaders["source"]?.columns ?? [] }, inject: { editor.sourceInspectionExecute = $0 })
        case .canvas:
            let address = WorkflowNodeAddress(domain: .data, localID: "csv")
            let library = ProjectLibrary(directory: root.appendingPathComponent("projects"))
            let document = WorkflowDocument(projectID: workspace.activeProjectID ?? UUID().uuidString,
                nodes: [.init(address: address, payload: .dataOperator(operatorID: "kwip.data.sourceCSV", config: ["separator": .string(",")]))])
            _ = try library.saveSource(.workflow(document))
            let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID,
                modelRegistry: .builtIn(), dataRegistry: .load(from: repo.appendingPathComponent("runner/dataflow-operators.json")))
            workspace.activeProjectID = owner.projectID
            let bindings = try WorkflowPrivateSourceBindings(owner: owner, directory: root)
            XCTAssertTrue(bindings.setPath(csv.path, for: address, lease: owner.lease(), expectedRevision: 0))
            let provider = try WorkflowSourceProvider(workspace: workspace, owner: owner, bindings: bindings)
            return Fixture(workspace: workspace, root: root, start: { XCTAssertTrue(provider.inspectSource(for: address, lease: owner.lease())) },
                busy: { provider.isInspecting }, message: { provider.statusMessage },
                columns: { provider.recordedSchema(for: address, in: owner.document).map(\.name) }, inject: { provider.inspectionExecute = $0 })
        }
    }
    private func waitUntil(_ predicate: () -> Bool) async throws {
        for _ in 0..<1000 {
            if predicate() { return }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("Source inspection did not reach the expected state")
    }
    func testEveryHostCapturesPersistedGlobalAndProjectPolicyBeforeAsyncWork() async throws {
        for host in Host.allCases {
            for overridden in [false, true] {
                let f = try fixture(host, project: overridden)
                XCTAssertTrue(f.workspace.setExecutionPreferences(try .init(maximumRunSeconds: 73, cpuThreads: 2, maximumResidentBytes: 268_435_456)))
                if overridden {
                    XCTAssertTrue(f.workspace.setProjectExecutionOverrides(try .init(maximumRunSeconds: 41, cpuThreads: 3, memory: .limit(536_870_912)), projectID: try XCTUnwrap(f.workspace.activeProjectID)))
                }
                var captured: ResolvedExecutionBudget?
                f.inject { _, _, _, window, _, _, _, exit in
                    captured = window.policy; exit(1, "fixture finished", nil); return .init(cancel: {})
                }
                f.start()
                XCTAssertTrue(f.workspace.setExecutionPreferences(try .init(maximumRunSeconds: 99, cpuThreads: 7)))
                try await waitUntil { !f.busy() }
                XCTAssertEqual(captured?.maximumRunSeconds, overridden ? 41 : 73, "\(host)")
                XCTAssertEqual(captured?.resources?.cpuThreads, overridden ? 3 : 2, "\(host)")
                XCTAssertEqual(captured?.resources?.maximumResidentBytes, overridden ? 536_870_912 : 268_435_456, "\(host)")
            }
        }
    }
    func testEveryHostRefusesQuotaBeforeSourceCopyOrInjectedWorker() async throws {
        for host in Host.allCases {
            let f = try fixture(host)
            XCTAssertTrue(f.workspace.setExecutionPreferences(try .init(managedStorageLimitBytes: 1)))
            var launched = false
            f.inject { _, _, _, _, _, _, _, exit in launched = true; exit(1, "unexpected launch", nil); return .init(cancel: {}) }
            f.start(); try await waitUntil { !f.busy() }
            XCTAssertFalse(launched, "\(host)")
            XCTAssertTrue(f.message().lowercased().contains("storage"), "\(host): \(f.message())")
            for namespace in ["source-inspection-staging", "workflow-operations"] {
                let children = (try? FileManager.default.contentsOfDirectory(atPath: f.root.appendingPathComponent(namespace).path)) ?? []
                XCTAssertTrue(children.isEmpty, "\(host) wrote \(namespace) before admission")
            }
            XCTAssertEqual(try WorkflowStorageAdmission(workspace: f.root).snapshot().reservedGrowthBytes, 0)
        }
    }
    func testEveryDefaultHostCompletesAdmittedInspectionWithPinnedPolicy() async throws {
        for host in Host.allCases {
            let f = try fixture(host)
            XCTAssertTrue(f.workspace.setExecutionPreferences(try .init(maximumRunSeconds: 60, cpuThreads: 2,
                maximumResidentBytes: 2_147_483_648, managedStorageLimitBytes: 1_100_000)))
            f.start(); try await waitUntil { !f.busy() }
            XCTAssertEqual(f.columns(), ["a", "b"], "\(host): \(f.message())")
            let enumerator = try XCTUnwrap(FileManager.default.enumerator(at: f.root.appendingPathComponent("workflow-operations"), includingPropertiesForKeys: nil))
            let requests = enumerator.compactMap { $0 as? URL }.filter { $0.lastPathComponent == "request.json" }
            XCTAssertEqual(requests.count, 1, "\(host)")
            let request = try WorkflowToolRequest.decode(Data(contentsOf: XCTUnwrap(requests.first)))
            XCTAssertEqual(request.toolMode, .sourceInspection)
            XCTAssertEqual(request.executionBudget.maximumRunSeconds, 60)
            XCTAssertEqual(request.executionBudget.resources?.cpuThreads, 2)
            XCTAssertEqual(request.executionBudget.resources?.maximumResidentBytes, 2_147_483_648)
            XCTAssertEqual(try WorkflowStorageAdmission(workspace: f.root).snapshot().reservedGrowthBytes, 0)
        }
    }
}
