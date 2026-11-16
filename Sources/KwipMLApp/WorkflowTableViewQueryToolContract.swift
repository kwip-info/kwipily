import Foundation
import KwipMLCore

enum WorkflowTableViewQueryToolContract {
    static func fileLimit(path: String, workload: ExecutionWorkloadPolicy? = nil) throws -> Int {
        if let workload,path.hasPrefix("input/"),PreparationArtifactPolicy.isSegmentName(String(path.dropFirst(6))) {return Int(workload.limit(.retainedBytes))}
        switch path {
        case "input/view-query.json": return 256_000
        case "input/view.sqlite3": return TableTransferCapacity(workload: workload).artifactBytes
        case "input/view.table.json": return TableTransferLimits.artifactBytes
        default: throw WorkflowFileLeaseError.unsafePath
        }
    }
    static func validate(arguments: [String], files: [WorkflowArchiveFile], directory: URL?, workload: ExecutionWorkloadPolicy? = nil) throws {
        guard arguments.count == 2, arguments[0] == "--query-table-view", arguments[1].hasPrefix("/"),
              !arguments[1].contains("\0"), !arguments[1].contains("//"), arguments[1].utf8.count <= 4096,
              !arguments[1].split(separator: "/").contains(".."), !arguments[1].split(separator: "/").contains("."),
              files.count >= 2, files.count <= (workload == nil ? 2 : 62_500), Set(files.map(\.path)).count == files.count else { throw WorkflowFileLeaseError.unsafePath }
        let ordinary = Set(files.map(\.path).filter { !$0.hasPrefix("input/table-pack-") })
        guard [Set(["input/view-query.json", "input/view.sqlite3"]), Set(["input/view-query.json", "input/view.table.json"])].contains(ordinary),
              files.count == 2 || ordinary.contains("input/view.table.json") && workload != nil else {throw WorkflowFileLeaseError.unsafePath}
        var total = 0
        for file in files {
            guard (1...(try fileLimit(path: file.path, workload: workload))).contains(file.bytes), WorkflowArchiveCodec.isHash(file.sha256) else { throw WorkflowFileLeaseError.unsafePath }
        }
        for file in files {
            let maximum = Int(workload?.limit(.retainedBytes) ?? 50_000_000) + 256_000
            guard file.bytes <= maximum - total else {throw WorkflowFileLeaseError.unsafePath}
            total += file.bytes
        }
        if let directory {
            guard arguments[1] == directory.appendingPathComponent("input/view-query.json").path else { throw WorkflowFileLeaseError.unsafePath }
        } else {
            guard arguments[1].hasSuffix("/input/view-query.json") else { throw WorkflowFileLeaseError.unsafePath }
        }
    }
    static func verifyInputs(arguments: [String], files: [WorkflowArchiveFile], directory: URL, workload: ExecutionWorkloadPolicy? = nil) throws {
        try validate(arguments: arguments, files: files, directory: directory, workload: workload)
        for file in files { try WorkflowPreviewToolContract.verifyFile(file: file, directory: directory, limit: fileLimit(path: file.path, workload: workload)) }
        let bytes = try DiagnosticFileIO.read(root: directory, parts: ["input", "view-query.json"], limit: 256_000)
        let query = try TableViewQuery.decode(bytes, workload: workload)
        let input = query.inputKind == "table" ? "input/view.table.json" : "input/view.sqlite3"
        guard files.first(where: { $0.path == input })?.sha256 == query.indexSHA256 else { throw WorkflowFileLeaseError.unsafePath }
    }
}
