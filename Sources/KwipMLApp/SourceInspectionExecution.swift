import Foundation
import KwipMLCore

typealias SourceInspectionExecutor = @MainActor (URL, String, RuntimeLocation, ExecutionBudgetWindow, URL, String, String, @escaping (Int32, String, SourceInspectionResult?) -> Void) async throws -> WorkflowWorkerHandle

/// Inspection returns a typed table, not an artifact directory. Shared ownership survives verification.
@MainActor enum SourceInspectionExecution {
    static func launch(workspace: URL, projectID: String, runtime: RuntimeLocation,
                       window: ExecutionBudgetWindow, sourceURL: URL, sourceSHA256: String, separator: String,
                       configuration: [String: JSONValue]? = nil,collectionMembers:[FileInputCapturedMember]=[],isFileInput:Bool=false,
                       storageAdmission: WorkflowCaptureStorageReservation? = nil,
                       launcher: @escaping WorkflowWorkerLauncher = WorkflowExecutionSession.launchWorker,
                       onExit: @escaping (Int32, String, SourceInspectionResult?) -> Void) async throws -> WorkflowWorkerHandle {
        let path = isFileInput ? "input/file-collections/"+sourceSHA256+"/manifest.json":"input/data/" + sourceURL.lastPathComponent
        _ = try WorkflowSourceInspectionToolContract.fileLimit(path: path)
        guard sourceURL.isFileURL, WorkflowArchiveCodec.isHash(sourceSHA256) else { throw WorkflowFileLeaseError.unsafePath }
        let parsingArguments = try configuration.map {
            if isFileInput {let validated=try FileInputConfiguration.validate($0);let data=try WorkflowArchiveCodec.encode(validated);return ["--file-options",String(decoding:data,as:UTF8.self)]}
            return try TableInputConfiguration.format(of: $0) == .csv && !CSVParsingConfiguration.isPolars($0)
                ? ["--csv-options", try CSVParsingConfiguration.transportJSON(TableInputConfiguration.validate($0))]
                : ["--table-options", try WorkflowSourceInspectionToolContract.configurationJSON($0)]
        }
            ?? ["--csv-separator", separator]
        var provisional: SourceInspectionResult?
        var diagnostic: CSVSourceDiagnostic?
        return try await WorkflowCapturedToolExecution.launch(mode: .sourceInspection, workspace: workspace,
            projectID: projectID, runtime: runtime, window: window,
            operationID: storageAdmission?.operationID ?? UUID().uuidString, storageAdmission: storageAdmission,
            inputs: { _ in [.file(path: path, source: sourceURL, expectedSHA256: sourceSHA256)]+collectionMembers.map {.file(path:"input/file-collections/"+sourceSHA256+"/"+$0.capturePath,source:$0.url,expectedSHA256:$0.sha256)} },
            arguments: { ["--inspect-data", $0.appendingPathComponent(path).path] + parsingArguments },
            launcher: launcher, onLine: { line in
                guard let data = line.data(using: .utf8) else { return }
                if let parsed = CSVSourceDiagnostic.decodeEvent(data) { diagnostic = parsed }
                if let result = try? SourceInspectionResult.decode(data, expectedSourceSHA256: sourceSHA256) { provisional = result }
            }, onExit: { code, message, verifiedToken in
                onExit(code, code == 1 ? (diagnostic?.transport ?? message) : message, code == 0 && verifiedToken != nil ? provisional : nil)
            })
    }
    nonisolated static func verifyResult(directory: URL, request: WorkflowToolRequest, completion: Data,
                                         window: ExecutionBudgetWindow) throws -> URL {
        try Task.checkCancellation()
        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        guard request.toolMode == .sourceInspection else { throw WorkflowFileLeaseError.unsafePath }
        try request.validate(directory: directory, verifyInputs: true)
        let result = try SourceInspectionResult.decode(completion, expectedSourceSHA256: request.inputFiles.first(where: {request.arguments[1]==directory.appendingPathComponent($0.path).path})?.sha256 ?? "")
        if request.arguments[2]=="--file-options" {
            let requested=try JSONDecoder().decode([String:JSONValue].self,from:Data(request.arguments[3].utf8))
            guard let parsing=result.parsing,result.collection != nil,try FileInputConfiguration.validate(parsing.options)==FileInputConfiguration.validate(requested),result.columnSchema != nil else {throw WorkflowArchiveError("File Input inspection returned a different collection or configuration.")}
        }
        if ["--csv-options", "--table-options"].contains(request.arguments[2]) {
            let requested = try JSONDecoder().decode([String: JSONValue].self, from: Data(request.arguments[3].utf8))
            guard let parsing = result.parsing,
                  try TableInputConfiguration.validate(parsing.options) == TableInputConfiguration.validate(requested) else {
                throw WorkflowArchiveError("Source inspection returned reader settings that do not match the requested configuration. Refresh the preview with the current settings.")
            }
        }
        if request.arguments[2] == "--table-options", result.columnSchema == nil {
            throw WorkflowArchiveError("Source inspection did not return verified column types. Refresh the preview or check that the development reader and app are up to date.")
        }
        try Task.checkCancellation()
        _ = try window.remainingSeconds(atUptime: ProcessInfo.processInfo.systemUptime)
        return directory
    }
}
