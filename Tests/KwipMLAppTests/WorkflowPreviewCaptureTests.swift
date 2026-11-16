import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPreviewCaptureTests: XCTestCase {
    func testCaptureRegistersCleanupLeaseAndRetainsItUntilConsumption() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let capture = try WorkflowPreviewCapture.stage(workspace: root, inputs: [.bytes(path: "input/graph.json", data: Data("{}".utf8))])
        let canonical = capture.directory.deletingLastPathComponent().deletingLastPathComponent()
        XCTAssertTrue(FileManager.default.fileExists(atPath: capture.directory.appendingPathComponent(".kwip-scratch.json").path))
        XCTAssertTrue(try WorkflowScratchCleanup.scan(workspace: canonical).candidates.isEmpty)
        XCTAssertEqual(capture.files.map(\.path), ["input/graph.json"], "Cleanup controls must never become worker input")
        capture.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: capture.directory.path))
    }

    func testReservedStageIdentityIsUsedAndCannotReplaceExistingCapture() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let stageID = UUID()
        let inputs: [WorkflowPreviewCapture.Input] = [.bytes(path: "input/graph.json", data: Data("{}".utf8))]
        let capture = try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs, stageID: stageID)
        XCTAssertEqual(capture.directory.lastPathComponent, stageID.uuidString)
        XCTAssertEqual(capture.directory.deletingLastPathComponent().lastPathComponent, "preview-staging")
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs, stageID: stageID))
        XCTAssertEqual(try Data(contentsOf: capture.directory.appendingPathComponent("input/graph.json")), Data("{}".utf8))
        capture.remove()
    }

    func testPreparationCaptureSupportsDigestPathsAndUsesSeparateOwnedStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let bytes = Data("a,b\n1,2\n".utf8)
        let hash = WorkflowArchiveCodec.hash(bytes)
        let path = "input/sources/\(hash)/observations.csv"
        let capture = try WorkflowPreviewCapture.stage(workspace: root, inputs: [
            .bytes(path: path, data: bytes),
            .bytes(path: "input/bindings.json", data: Data("{}".utf8))
        ], purpose: .preparation)
        XCTAssertEqual(capture.directory.deletingLastPathComponent().lastPathComponent, "preparation-staging")
        XCTAssertEqual(try Data(contentsOf: capture.directory.appendingPathComponent(path)), bytes)
        XCTAssertEqual(capture.files.first(where: { $0.path == path })?.sha256, hash)
        capture.remove()
        XCTAssertFalse(FileManager.default.fileExists(atPath: capture.directory.path))
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: [
            .bytes(path: "input/cache/activation-cache.pt", data: bytes)
        ], purpose: .preparation))
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: [
            .bytes(path: "input/node-labels.json", data: Data(repeating: 0, count: 65_537))
        ], purpose: .preparation))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("preparation-staging").path).isEmpty)
    }
    func testCaptureCopiesPinnedInputsAndPreservesCSVBasename() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("observations.csv")
        let csv = Data("a,b,label\n1,2,0\n".utf8); try csv.write(to: source)
        let snapshot = try WorkflowPreviewCapture.stage(workspace: root, inputs: [
            .bytes(path: "input/graph.json", data: Data("{}".utf8)),
            .file(path: "input/data/observations.csv", source: source, expectedSHA256: WorkflowArchiveCodec.hash(csv))
        ])
        defer { snapshot.remove() }
        XCTAssertEqual(snapshot.files.map(\.path), ["input/data/observations.csv", "input/graph.json"])
        XCTAssertEqual(try Data(contentsOf: snapshot.directory.appendingPathComponent("input/data/observations.csv")), csv)
        try Data("changed".utf8).write(to: source)
        XCTAssertEqual(try Data(contentsOf: snapshot.directory.appendingPathComponent("input/data/observations.csv")), csv)
    }
    func testCaptureRejectsChangedPinsAndSymlinksAndRemovesOwnPartialStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("cache.pt"); try Data([1,2,3]).write(to: source)
        let link = root.appendingPathComponent("linked.pt")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        for url in [source, link] {
            XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: [
                .bytes(path: "input/graph.json", data: Data("{}".utf8)),
                .file(path: "input/cache/activation-cache.pt", source: url, expectedSHA256: String(repeating: "0", count: 64))
            ]))
        }
        let stages = root.appendingPathComponent("preview-staging")
        XCTAssertTrue((try? FileManager.default.contentsOfDirectory(atPath: stages.path).isEmpty) ?? true)
        XCTAssertEqual(try Data(contentsOf: source), Data([1,2,3]))
    }
    func testCaptureRefusesDuplicateUnsafeAndOversizedInputs() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let graph = WorkflowPreviewCapture.Input.bytes(path: "input/graph.json", data: Data("{}".utf8))
        for inputs in [[graph, graph], [.bytes(path: "input/../escape", data: Data())],
                       [.bytes(path: "input/cache/activation-cache.json", data: Data(repeating: 0, count: 100_001))]] {
            XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs))
        }
    }
    func testCancellationDuringCaptureRemovesPartialStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        var checks = 0
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: [
            .bytes(path: "input/graph.json", data: Data(repeating: 32, count: 200_000))
        ], checkCancellation: { checks += 1; if checks >= 3 { throw CancellationError() } }))
        XCTAssertGreaterThanOrEqual(checks, 3)
        XCTAssertTrue((try? FileManager.default.contentsOfDirectory(atPath: root.appendingPathComponent("preview-staging").path).isEmpty) ?? true)
    }
    func testByteCountPreflightReadsMetadataWithoutCreatingCaptureAndExactBoundSucceeds() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.csv")
        try Data("x,y\n1,2\n".utf8).write(to: source)
        let inputs: [WorkflowPreviewCapture.Input] = [
            .bytes(path: "input/graph.json", data: Data("{}".utf8)),
            .file(path: "input/data/source.csv", source: source, expectedSHA256: nil)
        ]
        let bound = try WorkflowPreviewCapture.byteCountBound(inputs: inputs)
        XCTAssertEqual(bound, 10)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path), ["source.csv"])
        let capture = try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs, maximumBytes: bound)
        defer { capture.remove() }
        XCTAssertEqual(capture.files.reduce(0) { $0 + $1.bytes }, bound)
    }

    func testFileGrowthAfterPreflightCannotExceedReservationAndRemovesStage() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.csv")
        try Data([1, 2]).write(to: source)
        let inputs: [WorkflowPreviewCapture.Input] = [.file(path: "input/data/source.csv", source: source, expectedSHA256: nil)]
        let bound = try WorkflowPreviewCapture.byteCountBound(inputs: inputs)
        try Data(repeating: 32, count: 200_000).write(to: source)
        let stageID = UUID()
        let stage = root.appendingPathComponent("preview-staging/" + stageID.uuidString)
        var maximumObserved = 0
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs, stageID: stageID,
            maximumBytes: bound, checkCancellation: {
                let file = stage.appendingPathComponent("input/data/source.csv")
                let attributes = try? FileManager.default.attributesOfItem(atPath: file.path)
                maximumObserved = max(maximumObserved, (attributes?[.size] as? NSNumber)?.intValue ?? 0)
            }))
        XCTAssertLessThanOrEqual(maximumObserved, bound)
        XCTAssertFalse(FileManager.default.fileExists(atPath: stage.path))
        XCTAssertEqual(try Data(contentsOf: source).count, 200_000)
    }

    func testPreflightRejectsUnsafePathsSymlinksAndOversizedFileMetadata() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.json")
        try Data().write(to: source)
        let handle = try FileHandle(forWritingTo: source)
        try handle.truncate(atOffset: 2_000_001)
        try handle.close()
        let link = root.appendingPathComponent("link.json")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        let graph = WorkflowPreviewCapture.Input.bytes(path: "input/graph.json", data: Data())
        let invalid: [[WorkflowPreviewCapture.Input]] = [
            [], [graph, graph], [.bytes(path: "input/../graph.json", data: Data())],
            [.file(path: "input/graph.json", source: source, expectedSHA256: nil)],
            [.file(path: "input/graph.json", source: link, expectedSHA256: nil)],
            [.file(path: "input/graph.json", source: root, expectedSHA256: nil)],
            [.file(path: "input/graph.json", source: source, expectedSHA256: "invalid")],
            [.bytes(path: "input/graph.json", data: Data(repeating: 0, count: 2_000_001))]
        ]
        for inputs in invalid { XCTAssertThrowsError(try WorkflowPreviewCapture.byteCountBound(inputs: inputs)) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("preview-staging").path))
    }

    func testCaptureRejectsNegativeReservationBeforeCreatingWorkspaceAndAllowsEmptyExactBound() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let inputs: [WorkflowPreviewCapture.Input] = [.bytes(path: "input/graph.json", data: Data())]
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs, maximumBytes: -1))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.path))
        let capture = try WorkflowPreviewCapture.stage(workspace: root, inputs: inputs, maximumBytes: 0)
        defer { capture.remove() }
        XCTAssertEqual(capture.files.first?.bytes, 0)
        XCTAssertEqual(WorkflowPreviewCapture.Purpose.modelTraining.totalBytes, 330_000_000)
    }

}

extension WorkflowPreviewCaptureTests {
    func testAnchoredCaptureRejectsAncestorLinksAndChangedPins() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let archive = root.appendingPathComponent("archive")
        try FileManager.default.createDirectory(at: archive, withIntermediateDirectories: true)
        let bytes = Data(repeating: 65, count: 200_000)
        try bytes.write(to: archive.appendingPathComponent("table.json"))
        let input = WorkflowPreviewCapture.Input.anchoredFile(path: "input/preparation/table.json", root: root,
            parts: ["archive", "table.json"], expectedSHA256: WorkflowArchiveCodec.hash(bytes))
        XCTAssertEqual(try WorkflowPreviewCapture.byteCountBound(inputs: [input], purpose: .preparationInspection), bytes.count)
        let snapshot = try WorkflowPreviewCapture.stage(workspace: root, inputs: [input], purpose: .preparationInspection)
        defer { snapshot.remove() }
        XCTAssertEqual(try Data(contentsOf: snapshot.directory.appendingPathComponent(input.path)), bytes)
        try Data("changed".utf8).write(to: archive.appendingPathComponent("table.json"))
        XCTAssertThrowsError(try WorkflowPreviewCapture.stage(workspace: root, inputs: [input], purpose: .preparationInspection))
        try FileManager.default.moveItem(at: archive, to: root.appendingPathComponent("moved"))
        try FileManager.default.createSymbolicLink(at: archive, withDestinationURL: root.appendingPathComponent("moved"))
        XCTAssertThrowsError(try WorkflowPreviewCapture.byteCountBound(inputs: [input], purpose: .preparationInspection))
    }
}

extension WorkflowPreviewCaptureTests {
    func testPreparedConsumersBindSegmentAllowanceToLaunchPolicy() throws {
        let workload = try ExecutionWorkloadPolicy(preset: .balanced)
        let path = "input/preparation/table-pack-" + String(repeating: "a", count: 32) + ".kwipdata"
        let limits: [(String, ExecutionWorkloadPolicy?) throws -> Int] = [
            { try BackendFitOperationToolContract.fileLimit(path: $0, workload: $1) },
            { try WorkflowPreviewToolContract.fileLimit(path: $0, workload: $1) },
            { try WorkflowModelTrainingToolContract.fileLimit(path: $0, workload: $1) }
        ]
        for limit in limits {
            XCTAssertEqual(try limit(path, workload), Int(workload.limit(.retainedBytes)))
            XCTAssertEqual(try limit(path, nil), 50_000_000)
            XCTAssertEqual(try limit("input/preparation/not-a-segment.bin", workload), 50_000_000)
        }
        XCTAssertEqual(try WorkflowPreparationToolContract.fileLimit(path: path.replacingOccurrences(of: "/preparation/", with: "/reuse/"), workload: workload), Int(workload.limit(.retainedBytes)))
    }
}

extension WorkflowPreviewCaptureTests {
    func testTableQueryCapacityIsCapturedSeparatelyFromPageFraming() throws {
        let policy = try ExecutionWorkloadPolicy(preset: .balanced)
        let query = TableViewQuery(offset: 6_000_000, limit: 100, search: "", sourceSHA256: String(repeating: "a", count: 64), indexSHA256: String(repeating: "b", count: 64), sortColumn: 512)
        XCTAssertThrowsError(try query.validate())
        XCTAssertNoThrow(try query.validate(workload: policy))
        XCTAssertEqual(try TableViewQuery.decode(WorkflowArchiveCodec.encode(query), workload: policy), query)
        var oversizedPage = query; oversizedPage.limit = 101
        XCTAssertThrowsError(try oversizedPage.validate(workload: policy))
        XCTAssertEqual(try WorkflowPreviewCapture.Purpose.tableViewQuery.fileLimit("input/view.sqlite3", workload: policy), Int(policy.limit(.retainedBytes)))
        XCTAssertEqual(try WorkflowPreviewCapture.Purpose.tableViewQuery.fileLimit("input/view.table.json", workload: policy), 50_000_000)
    }
}
