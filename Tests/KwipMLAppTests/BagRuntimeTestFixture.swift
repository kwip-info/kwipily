import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

/// Only immutable runtime inputs are shared. Every test still constructs fresh
/// stores, sources, owners, reservations and real worker sessions.
@MainActor enum BagRuntimeTestFixture {
    private struct Cached {
        let repository: URL
        let registry: Data
        let sourcePins: [String:String]
        let root: URL
    }
    private static var cached: Cached?
    static func runtime(repository: URL, registry: ComponentRegistry) throws -> URL {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let encoded = try encoder.encode(registry)
        let pins = try sourcePins(repository)
        if let cached {
            guard cached.repository == repository, cached.registry == encoded, cached.sourcePins == pins else {
                throw WorkflowPhaseError("Bag fixture runtime inputs changed during the test process; no cached capability/archive may be reused.")
            }
            for (path, pin) in pins {
                guard WorkflowArchiveCodec.hash(try Data(contentsOf: cached.root.appendingPathComponent(path))) == pin else {
                    throw WorkflowPhaseError("Immutable copied Bag runtime was modified.")
                }
            }
            guard try Data(contentsOf: cached.root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json")) == encoded else {
                throw WorkflowPhaseError("Immutable Bag registry was modified.")
            }
            return cached.root
        }
        let runtimeRoot = try BagOwnedFixtureDirectories.shared.make()
        let manager = FileManager.default
        for part in ["runner", "scripts", "Sources/KwipMLCore/Resources"] {
            try manager.createDirectory(at: runtimeRoot.appendingPathComponent(part), withIntermediateDirectories: true)
        }
        // Copy source files, not caches/tests/symlinks; the resulting full worker inventory is captured.
        for part in ["runner", "scripts"] {
            for url in try manager.contentsOfDirectory(at: repository.appendingPathComponent(part), includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey]) {
                let info = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                if info.isRegularFile == true && info.isSymbolicLink != true && !url.lastPathComponent.hasSuffix(".pyc") {
                    try manager.copyItem(at: url, to: runtimeRoot.appendingPathComponent(part).appendingPathComponent(url.lastPathComponent))
                }
            }
        }
        // Guarded discovery/training attest these existing development assets before import.
        // Stage all three before capturing any runtime evidence; never rebuild or bypass the guard.
        try manager.createDirectory(at: runtimeRoot.appendingPathComponent("runner/native"), withIntermediateDirectories: true)
        try manager.copyItem(at: repository.appendingPathComponent("runner/native/lifetime_watchdog.c"),
                             to: runtimeRoot.appendingPathComponent("runner/native/lifetime_watchdog.c"))
        try manager.createDirectory(at: runtimeRoot.appendingPathComponent(".local/native-watchdog"), withIntermediateDirectories: true)
        for name in ["lifetime_watchdog.dylib", "lifetime_watchdog.json"] {
            try manager.copyItem(at: repository.appendingPathComponent(".local/native-watchdog").appendingPathComponent(name),
                                 to: runtimeRoot.appendingPathComponent(".local/native-watchdog").appendingPathComponent(name))
        }
        // Development RuntimeLocation explicitly permits the host interpreter; worker code remains copied/pinned.
        try manager.createSymbolicLink(at: runtimeRoot.appendingPathComponent(".venv"), withDestinationURL: repository.appendingPathComponent(".venv"))
        try encoded.write(to: runtimeRoot.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"))

        cached = .init(repository: repository, registry: encoded, sourcePins: pins, root: runtimeRoot)
        return runtimeRoot
    }
    private static func sourcePins(_ repository: URL) throws -> [String:String] {
        var paths: [String] = []
        for part in ["runner", "scripts"] {
            for url in try FileManager.default.contentsOfDirectory(at: repository.appendingPathComponent(part), includingPropertiesForKeys: [.isRegularFileKey,.isSymbolicLinkKey]) {
                let info = try url.resourceValues(forKeys: [.isRegularFileKey,.isSymbolicLinkKey])
                if info.isRegularFile == true && info.isSymbolicLink != true && !url.lastPathComponent.hasSuffix(".pyc") { paths.append(part + "/" + url.lastPathComponent) }
            }
        }
        paths += ["runner/native/lifetime_watchdog.c", ".local/native-watchdog/lifetime_watchdog.dylib", ".local/native-watchdog/lifetime_watchdog.json"]
        return try Dictionary(uniqueKeysWithValues: paths.map { ($0, WorkflowArchiveCodec.hash(try Data(contentsOf: repository.appendingPathComponent($0)))) })
    }
}

/// XCTest owns only these unique fixture paths; cleanup never scans user data.
private final class BagOwnedFixtureDirectories: NSObject, XCTestObservation {
    static let shared = BagOwnedFixtureDirectories()
    private var roots: [URL] = []
    override private init() { super.init(); XCTestObservationCenter.shared.addTestObserver(self) }
    func make() throws -> URL {
        guard let physical = realpath(FileManager.default.temporaryDirectory.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        defer { free(physical) }
        let root = URL(fileURLWithPath: String(cString: physical)).appendingPathComponent("KwipML-BagTest-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        roots.append(root); return root
    }
    func testBundleDidFinish(_ testBundle: Bundle) {
        for root in roots { try? FileManager.default.removeItem(at: root) }
        roots.removeAll()
    }
}

@MainActor enum BagHistoryTestFixture {
    private static var captures: [Bool: BagWorkflowSessionTests.Captured] = [:]
    static func clone(objective: Bool, into root: URL) async throws -> BagWorkflowSessionTests.Captured {
        let captured: BagWorkflowSessionTests.Captured
        if let cached = captures[objective] { captured = cached }
        else {
            let base = try BagOwnedFixtureDirectories.shared.make()
            let fresh = try await BagWorkflowSessionTests().capture(objective: objective, root: base)
            _ = try XCTUnwrap(WorkflowHistoryReader.loadModelContent(fresh.reference, root: base))
            captures[objective] = fresh; captured = fresh
        }
        // Reverify the frozen archive before each physical clone; never share a
        // mutable History tree with a tamper test or reuse a failed capture.
        let base = captured.directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        _ = try XCTUnwrap(WorkflowHistoryReader.loadModelContent(captured.reference, root: base))
        let target = root.appendingPathComponent("workflow-runs/\(captured.reference.projectID)/\(captured.reference.directoryID.uuidString)")
        try copyTree(captured.directory, to: target)
        _ = try XCTUnwrap(WorkflowHistoryReader.loadModelContent(captured.reference, root: root))
        return .init(reference: captured.reference, directory: target, plan: captured.plan, workerRunID: captured.workerRunID)
    }
    private static func copyTree(_ source: URL, to target: URL) throws {
        let info = try source.resourceValues(forKeys: [.isRegularFileKey,.isDirectoryKey,.isSymbolicLinkKey])
        guard info.isSymbolicLink != true else { throw WorkflowFileLeaseError.unsafePath }
        if info.isDirectory == true {
            try FileManager.default.createDirectory(at: target, withIntermediateDirectories: true)
            for child in try FileManager.default.contentsOfDirectory(at: source, includingPropertiesForKeys: nil) {
                try copyTree(child, to: target.appendingPathComponent(child.lastPathComponent))
            }
        } else if info.isRegularFile == true {
            try FileManager.default.copyItem(at: source, to: target)
            let a = try FileManager.default.attributesOfItem(atPath: source.path), b = try FileManager.default.attributesOfItem(atPath: target.path)
            guard (a[.systemFileNumber] as? NSNumber) != (b[.systemFileNumber] as? NSNumber) else { throw WorkflowFileLeaseError.unsafePath }
        } else { throw WorkflowFileLeaseError.unsafePath }
    }
}
