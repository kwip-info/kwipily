import Foundation
import KwipMLCore

/// Source-only exports carry the checked worker sources, never the Python executable,
/// private inputs, credentials, caches or a user's captured data.
enum WorkflowCodePackage {
    static func contents(files: [String: String], runtime: RuntimeLocation, expectedEvidence: WorkflowRuntimeEvidence) throws -> [String: Data] {
        guard !files.isEmpty, files.count <= 128 else { throw WorkflowArchiveError("No bounded Code package is available.") }
        var result: [String: Data] = [:], total = 0
        for (name, text) in files {
            guard safePath(name), !name.hasPrefix("runner/"), !name.hasPrefix("Sources/") else {
                throw WorkflowArchiveError("The generated package contains an unsafe filename.")
            }
            let bytes = Data(text.utf8)
            guard bytes.count <= 2_000_000 else { throw WorkflowArchiveError("A generated source file exceeds its size limit.") }
            result[name] = bytes; total += bytes.count
        }
        let evidence = try WorkflowRuntimeEvidence.capture(runtime)
        guard evidence == expectedEvidence else { throw WorkflowArchiveError("The runtime changed since source generation. Generate the source again before exporting.") }
        for (name, hash) in evidence.workerFiles.sorted(by: { $0.key < $1.key }) {
            guard name.hasPrefix("runner/"), name.hasSuffix(".py") || name.hasSuffix(".json") || name == "runner/laya-LICENSE.txt" else { continue }
            let bytes = try DiagnosticFileIO.read(root: runtime.workingDirectoryURL, parts: name.split(separator: "/").map(String.init), limit: 10_000_000)
            guard WorkflowArchiveCodec.hash(bytes) == hash else { throw WorkflowArchiveError("The runtime changed while exporting the package.") }
            guard result[name] == nil else { throw WorkflowArchiveError("A generated file conflicts with the runtime.") }
            result[name] = bytes; total += bytes.count
        }
        let registry = try DiagnosticFileIO.read(root: runtime.registryURL.deletingLastPathComponent(), parts: [runtime.registryURL.lastPathComponent], limit: 10_000_000)
        guard WorkflowArchiveCodec.hash(registry) == evidence.workerFiles["components.json"] else {
            throw WorkflowArchiveError("The model registry changed while exporting the package.")
        }
        result["Sources/KwipMLCore/Resources/components.json"] = registry
        result["components.json"] = registry
        total += registry.count * 2
        guard total <= 40_000_000 else { throw WorkflowArchiveError("Code package exceeds its source size limit.") }
        return result
    }
    static func safePath(_ name: String) -> Bool {
        let parts = name.split(separator: "/", omittingEmptySubsequences: false)
        return !name.isEmpty && name.utf8.count <= 512 && !name.contains("\\") && !name.contains("\0")
            && parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
    static func write(_ files: [String: Data], to destination: URL) throws {
        let manager = FileManager.default
        guard !manager.fileExists(atPath: destination.path), files.keys.allSatisfy(safePath) else {
            throw WorkflowArchiveError("Choose a new package folder; existing files will not be replaced.")
        }
        let staging = destination.deletingLastPathComponent().appendingPathComponent(".kwip-code-" + UUID().uuidString)
        try manager.createDirectory(at: staging, withIntermediateDirectories: false)
        defer { try? manager.removeItem(at: staging) }
        for (name, bytes) in files.sorted(by: { $0.key < $1.key }) {
            try Task.checkCancellation()
            let file = staging.appendingPathComponent(name)
            try manager.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try bytes.write(to: file, options: .withoutOverwriting)
        }
        try manager.moveItem(at: staging, to: destination)
    }
}
