import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparationLaunchPlanTests: XCTestCase {
    private func root() throws -> URL {
        let value = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: value, withIntermediateDirectories: true)
        let canonical = try XCTUnwrap(realpath(value.path, nil))
        defer { free(canonical) }
        return URL(fileURLWithPath: String(cString: canonical))
    }
    private func graph() throws -> DataFlowDocument {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try DataFlowCodec.decodeGraph(Data(contentsOf: repository.appendingPathComponent("examples/dataflow/scaler.json")))
    }
    private func source(_ root: URL) throws -> URL {
        let url = root.appendingPathComponent("Original observations.csv")
        try Data("feature,label\n1,0\n2,1\n".utf8).write(to: url)
        return url
    }
    func testLargeCacheAdmissionRequiresCapturedWorkloadBeforeStaging() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root), policy = try ExecutionWorkloadPolicy(preset: .balanced)
        let plan = try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source": source.path], labels: [:],
            cachePolicy: .materialize, cacheReference: nil, workspace: root, sourcePermissions: root.appendingPathComponent("grants"), preparationCacheLimitBytes: 300_000_000, workload: policy)
        defer { plan.removeSourceStage() }
        let args = plan.arguments(in: root)
        let index = try XCTUnwrap(args.firstIndex(of: "--dataflow-cache-limit-bytes"))
        XCTAssertEqual(args[index + 1], "300000000")
        XCTAssertThrowsError(try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source": source.path], labels: [:],
            cachePolicy: .materialize, cacheReference: nil, workspace: root, sourcePermissions: root.appendingPathComponent("grants"), preparationCacheLimitBytes: 300_000_000))
    }
    func testCapturedCacheAllowanceIsOnlyPassedForActiveCaching() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root)
        for policy in [WorkflowPreparationCachePolicy.off, .materialize] {
            let plan = try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source": source.path], labels: [:],
                cachePolicy: policy, cacheReference: nil, workspace: root, sourcePermissions: root.appendingPathComponent("grants"), preparationCacheLimitBytes: 12345)
            defer { plan.removeSourceStage() }
            let args = plan.arguments(in: root)
            if policy == .off { XCTAssertFalse(args.contains("--dataflow-cache-limit-bytes")) }
            else {
                let index = try XCTUnwrap(args.firstIndex(of: "--dataflow-cache-limit-bytes"))
                XCTAssertEqual(args[index + 1], "12345")
            }
        }
    }
    func testSourcesArePrivateDescriptorsAndBindingsUseEventualOperationDirectory() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root), raw = try Data(contentsOf: source)
        let plan = try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source": source.path], labels: ["source": "My observations"], cachePolicy: .off, cacheReference: nil, workspace: root, sourcePermissions: root.appendingPathComponent("grants"))
        defer { plan.removeSourceStage() }
        try Data("changed".utf8).write(to: source)
        let operation = root.appendingPathComponent("workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
        let inputs = try plan.inputs(in: operation)
        let files = inputs.compactMap { input -> (String, URL, String?)? in
            if case .file(let path, let source, let pin) = input { return (path, source, pin) }; return nil
        }
        XCTAssertEqual(files.count, 1)
        let file = try XCTUnwrap(files.first)
        XCTAssertEqual(file.0, "input/sources/\(WorkflowArchiveCodec.hash(raw))/Original observations.csv")
        XCTAssertTrue(file.1.path.hasPrefix(root.appendingPathComponent("preparation-source-staging").path + "/"), "Captured path: \(file.1.path); root: \(root.path)")
        XCTAssertEqual(try Data(contentsOf: file.1), raw)
        XCTAssertEqual(file.2, WorkflowArchiveCodec.hash(raw))
        XCTAssertEqual(plan.sourceDigests[source.path], WorkflowArchiveCodec.hash(raw))
        let bytes = try XCTUnwrap(inputs.compactMap { input -> Data? in
            if case .bytes("input/bindings.json", let data) = input { return data }; return nil
        }.first)
        let bindings = try JSONDecoder().decode([String: String].self, from: bytes)
        XCTAssertEqual(bindings["source"], operation.appendingPathComponent(file.0).path)
        XCTAssertFalse(plan.arguments(in: operation).contains(source.path))
        plan.removeSourceStage()
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.1.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
    }
    func testSharedSourceDeduplicatesWithoutRetainingPayloads() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root)
        var graph = try graph(); var second = graph.nodes[0]; second.id = "other-source"; graph.nodes.append(second)
        let plan = try PreparationLaunchPlan.capture(graph: graph, bindings: ["source": source.path, "other-source": source.path], labels: [:], cachePolicy: .materialize, cacheReference: nil, workspace: root, sourcePermissions: root.appendingPathComponent("grants"))
        defer { plan.removeSourceStage() }
        let inputs = try plan.inputs(in: root)
        XCTAssertEqual(inputs.filter { if case .file = $0 { return true }; return false }.count, 1)
        XCTAssertEqual(plan.sourceDigests.count, 1)
        XCTAssertTrue(plan.arguments(in: root).contains("materialize"))
    }
    func testCancellationAndRevokedSourceLeaveNoPrivateStage() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root), graph = try graph(), grants = root.appendingPathComponent("grants")
        var checks = 0
        XCTAssertThrowsError(try PreparationLaunchPlan.capture(graph: graph, bindings: ["source": source.path], labels: [:], cachePolicy: .off, cacheReference: nil, workspace: root, sourcePermissions: grants, checkCancellation: {
            checks += 1; if checks == 6 { throw CancellationError() }
        }))
        try SourceAccess(directory: grants).revoke(path: source.path)
        XCTAssertThrowsError(try PreparationLaunchPlan.capture(graph: graph, bindings: ["source": source.path], labels: [:], cachePolicy: .off, cacheReference: nil, workspace: root, sourcePermissions: grants))
        let stages = root.appendingPathComponent("preparation-source-staging")
        if FileManager.default.fileExists(atPath: stages.path) {
            XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: stages.path).isEmpty)
        }
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains {
            $0.hasPrefix("preparation-source-") && $0 != "preparation-source-staging"
        })
    }
    func testInvalidLabelsBindingsAndMissingReuseReject() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root), graph = try graph(), grants = root.appendingPathComponent("grants")
        for (bindings, labels, policy) in [([String:String](), [:], WorkflowPreparationCachePolicy.off), (["source":source.path], ["missing":"Label"], .off), (["source":source.path], ["source":" "], .off), (["source":source.path], [:], .reuse)] {
            XCTAssertThrowsError(try PreparationLaunchPlan.capture(graph: graph, bindings: bindings, labels: labels, cachePolicy: policy, cacheReference: nil, workspace: root, sourcePermissions: grants))
        }
    }
    func testReuseKeepsPinnedDescriptorsAndDoesNotRequireCurrentGraphDigest() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root), prior = root.appendingPathComponent("prior")
        try FileManager.default.createDirectory(at: prior, withIntermediateDirectories: true)
        let bytes = Data("prior graph".utf8); try bytes.write(to: prior.appendingPathComponent("dataflow.json"))
        let manifest: [String: Any] = ["schemaVersion":1,"kind":"dataflowExecution","status":"completed","graphDigest":String(repeating:"a",count:64),"artifacts":[["path":"dataflow.json","bytes":bytes.count,"sha256":WorkflowArchiveCodec.hash(bytes)]]]
        let raw = try JSONSerialization.data(withJSONObject: manifest, options: [.sortedKeys]); try raw.write(to: prior.appendingPathComponent("manifest.json"))
        let reference = DataFlowCacheReference(manifestPath: prior.appendingPathComponent("manifest.json").path, manifestSHA256: WorkflowArchiveCodec.hash(raw))
        let plan = try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source":source.path], labels: [:], cachePolicy: .reuse, cacheReference: reference, workspace: root, sourcePermissions: root.appendingPathComponent("grants"))
        defer { plan.removeSourceStage() }
        let inputs = try plan.inputs(in: root)
        XCTAssertTrue(inputs.contains { $0.path == "input/reuse/dataflow.json" })
        XCTAssertTrue(plan.arguments(in: root).contains(reference.manifestSHA256))
        try Data("tampered".utf8).write(to: prior.appendingPathComponent("dataflow.json"))
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs, purpose: .preparation))
    }
    func testInsufficientStorageAdmissionDoesNotCreateSourceStage() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root)
        XCTAssertThrowsError(try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source": source.path],
            labels: [:], cachePolicy: .off, cacheReference: nil, workspace: root,
            sourcePermissions: root.appendingPathComponent("grants"), projectID: UUID().uuidString, storageLimitBytes: 1))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("preparation-source-staging").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.path))
    }

    func testSmallSourceStorageAdmissionUsesActualBoundsInsteadOfMaximumPreparationAllowance() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root)
        // 302 MB worker artifacts/scratch plus bounded actual capture fits here;
        // the 1.5 GB maximum preparation capture allowance does not.
        var plan: PreparationLaunchPlan? = try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source": source.path],
            labels: [:], cachePolicy: .off, cacheReference: nil, workspace: root,
            sourcePermissions: root.appendingPathComponent("grants"), projectID: UUID().uuidString, storageLimitBytes: 305_000_000)
        XCTAssertNotNil(plan?.storageReservation)
        let inputs = try XCTUnwrap(plan).inputs(in: root)
        let copiedSource = try XCTUnwrap(inputs.compactMap { input -> URL? in
            if case .file(_, let url, _) = input { return url }; return nil
        }.first)
        XCTAssertEqual(try Data(contentsOf: copiedSource), try Data(contentsOf: source))
        plan = nil
        XCTAssertFalse(FileManager.default.fileExists(atPath: copiedSource.path))
        let retry = try PreparationLaunchPlan.capture(graph: graph(), bindings: ["source": source.path],
            labels: [:], cachePolicy: .off, cacheReference: nil, workspace: root,
            sourcePermissions: root.appendingPathComponent("grants"), projectID: UUID().uuidString, storageLimitBytes: 305_000_000)
        retry.removeSourceStage()
    }

    func testSourceGrowthAfterAdmissionRejectsBeforeSourceCopyAndReleasesReservation() throws {
        let root = try root(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try source(root), graph = try graph(), projectID = UUID().uuidString
        let stages = root.appendingPathComponent("preparation-source-staging")
        var grew = false
        XCTAssertThrowsError(try PreparationLaunchPlan.capture(graph: graph, bindings: ["source": source.path],
            labels: [:], cachePolicy: .off, cacheReference: nil, workspace: root,
            sourcePermissions: root.appendingPathComponent("grants"), projectID: projectID, storageLimitBytes: 305_000_000,
            checkCancellation: {
                if !grew, FileManager.default.fileExists(atPath: stages.path) {
                    try Data(repeating: 32, count: 1_000_000).write(to: source)
                    grew = true
                }
            }))
        XCTAssertTrue(grew)
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: stages.path).isEmpty)
        try Data("feature,label\n1,0\n".utf8).write(to: source)
        let retry = try PreparationLaunchPlan.capture(graph: graph, bindings: ["source": source.path],
            labels: [:], cachePolicy: .off, cacheReference: nil, workspace: root,
            sourcePermissions: root.appendingPathComponent("grants"), projectID: projectID, storageLimitBytes: 305_000_000)
        retry.removeSourceStage()
    }

}
