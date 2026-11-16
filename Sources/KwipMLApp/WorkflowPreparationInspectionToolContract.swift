import Foundation
import KwipMLCore

/// Reopen a complete historical preparation without granting access to its original sources.
enum WorkflowPreparationInspectionToolContract {
    static let prefix = "input/preparation/"
    static func fileLimit(path: String, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 3, parts[0] == "input", parts[1] == "preparation",
              !parts[2].isEmpty, parts[2] != ".", parts[2] != "..",
              !parts[2].contains("\0"), parts[2].utf8.count <= 255 else { throw WorkflowFileLeaseError.unsafePath }
        if let workload, PreparationArtifactPolicy.isCapacityFileName(String(parts[2])) {return Int(workload.limit(.retainedBytes))}
        return parts[2] == "manifest.json" ? 2_000_000 : 50_000_000
    }
    static func validate(arguments: [String], files: [WorkflowArchiveFile], directory: URL?, workload: ExecutionWorkloadPolicy? = nil) throws {
        guard arguments.count == 4, arguments[0] == "--inspect-dataflow", arguments[2] == "--dataflow-manifest-sha256",
              arguments.allSatisfy({ !$0.isEmpty && !$0.contains("\0") && $0.utf8.count <= 4096 }),
              (2...WorkflowPreviewCapture.Purpose.preparationInspection.captureCount(workload)).contains(files.count), Set(files.map(\.path)).count == files.count,
              let manifest = files.first(where: { $0.path == prefix + "manifest.json" }), arguments[3] == manifest.sha256,
              arguments[1].hasPrefix("/"), !arguments[1].contains("//"),
              !arguments[1].split(separator: "/").contains("."), !arguments[1].split(separator: "/").contains("..") else {
            throw WorkflowFileLeaseError.unsafePath
        }
        if let directory {
            guard directory.isFileURL, arguments[1] == directory.appendingPathComponent(manifest.path).path else { throw WorkflowFileLeaseError.unsafePath }
        } else {
            guard arguments[1].hasSuffix("/" + manifest.path) else { throw WorkflowFileLeaseError.unsafePath }
        }
        var artifactBytes = 0
        for file in files {
            let limit = try fileLimit(path: file.path, workload: workload)
            guard (0...limit).contains(file.bytes), WorkflowArchiveCodec.isHash(file.sha256) else { throw WorkflowFileLeaseError.unsafePath }
            if file.path != manifest.path {
                guard file.bytes <= Int(workload?.limit(.retainedBytes) ?? 200_000_000) - artifactBytes else { throw WorkflowFileLeaseError.unsafePath }
                artifactBytes += file.bytes
            }
        }
    }
    static func verifyInputs(arguments: [String], files: [WorkflowArchiveFile], directory: URL, workload: ExecutionWorkloadPolicy? = nil) throws {
        try validate(arguments: arguments, files: files, directory: directory, workload: workload)
        for file in files {
            try WorkflowPreviewToolContract.verifyFile(file: file, directory: directory, limit: fileLimit(path: file.path, workload: workload))
        }
        let data = try WorkflowEvaluationOwnership.readProtocol(directory.appendingPathComponent("input/preparation"),
            name: "manifest.json", limit: 2_000_000)
        try WorkflowCompletedPreparationInventory.verify(data: data, files: files, prefix: prefix, workload: workload)
    }
}
