import Foundation
import Darwin
import KwipMLCore

/// Typed app-owned tools share operation ownership without impersonating a model phase.
struct WorkflowToolRequest: Codable, Equatable {
    enum Mode: String, Codable { case mixtureValidation, selfPlay, gameSession, modelAsset, textGeneration, programExport, backendFit, learningRuntime, learningAssessment, codeExport, modelPreview, modelTraining, dataPreparation, sourceInspection, preparationInspection, tableViewQuery, runtimeCapabilities }
    var schemaVersion: Int = 2
    var kind: String = "workflowOperation"
    let operationID: String
    let projectID: String
    let requestIdentity: String
    let toolMode: Mode
    let arguments: [String]
    let inputFiles: [WorkflowArchiveFile]
    let executionBudget: ResolvedExecutionBudget
    let remainingSeconds: Double

    func validate() throws {
        guard schemaVersion == 2, kind == "workflowOperation",
              UUID(uuidString: operationID)?.uuidString == operationID,
              UUID(uuidString: projectID)?.uuidString == projectID,
              WorkflowArchiveCodec.isHash(requestIdentity), remainingSeconds.isFinite, remainingSeconds > 0,
              remainingSeconds <= Double(executionBudget.maximumRunSeconds),
              arguments.count <= (toolMode == .dataPreparation ? 128 : (toolMode == .modelPreview || toolMode == .modelTraining) ? 64 : toolMode == .learningAssessment ? 8 : 6), arguments.enumerated().allSatisfy({ index, value in !value.isEmpty && !value.contains("\0") && value.utf8.count <= (toolMode == .sourceInspection && index == 3 && arguments.count == 4 && ["--table-options", "--file-options"].contains(arguments[2]) ? 32_768 : 4096) }) else {
            throw WorkflowFileLeaseError.unsafePath
        }
        switch toolMode {
        case .mixtureValidation:
            guard remainingSeconds<=30,arguments.count==2,arguments[0]=="--validate-mixture",inputFiles.count==1,inputFiles[0].path=="input/mixture.json",(1...100_000).contains(inputFiles[0].bytes),WorkflowArchiveCodec.isHash(inputFiles[0].sha256) else{throw WorkflowFileLeaseError.unsafePath}
            try validateInputPath(arguments[1],name:"mixture.json")
        case .selfPlay:
            try SelfPlayToolContract.validate(arguments:arguments,files:inputFiles,directory:nil)
            try validateInputPath(arguments[1],name:"self-play.json")
        case .gameSession:
            try GameSessionToolContract.validate(arguments:arguments,files:inputFiles,directory:nil)
            try validateInputPath(arguments[1],name:"session.json")
        case .modelAsset:
            try ModelAssetToolContract.validate(arguments:arguments,files:inputFiles,directory:nil)
            try validateInputPath(arguments[1],name:"model-request.json")
        case .textGeneration:
            try WorkflowGenerationToolContract.validate(arguments:arguments,files:inputFiles,directory:nil)
            try validateInputPath(arguments[1],name:"generation.json")
        case .programExport:
            try WorkflowProgramExportToolContract.validate(arguments:arguments,files:inputFiles,directory:nil)
            try validateInputPath(arguments[1],name:"program-export.json")
        case .backendFit:
            try BackendFitOperationToolContract.validate(arguments: arguments, files: inputFiles, directory: nil, workload: executionBudget.resources?.workloadPolicy)
            try validateInputPath(arguments[1], name: "backend-request.json")
        case .runtimeCapabilities:
            guard arguments == ["--capabilities"], inputFiles.isEmpty else { throw WorkflowFileLeaseError.unsafePath }
        case .tableViewQuery:
            guard remainingSeconds <= 30 else { throw WorkflowFileLeaseError.unsafePath }
            try WorkflowTableViewQueryToolContract.validate(arguments: arguments, files: inputFiles, directory: nil, workload: executionBudget.resources?.workloadPolicy)
            try validateInputPath(arguments[1], name: "view-query.json")
        case .preparationInspection:
            try WorkflowPreparationInspectionToolContract.validate(arguments: arguments, files: inputFiles, directory: nil, workload: executionBudget.resources?.workloadPolicy)
            try validateInputPath(arguments[1], name: "preparation/manifest.json")
        case .sourceInspection:
            try WorkflowSourceInspectionToolContract.validate(arguments: arguments, files: inputFiles, directory: nil)
            if arguments[2] == "--file-options" {
                // The source contract has already required one exact, hash-pinned collection manifest.
                guard let manifest = inputFiles.first(where: {
                    $0.path.hasPrefix("input/file-collections/") && $0.path.hasSuffix("/manifest.json")
                }) else { throw WorkflowFileLeaseError.unsafePath }
                try validateInputPath(arguments[1], name: String(manifest.path.dropFirst("input/".count)))
            } else {
                try validateInputPath(arguments[1], name: "data/" + URL(fileURLWithPath: inputFiles[0].path).lastPathComponent)
            }
        case .dataPreparation:
            try WorkflowPreparationToolContract.validate(arguments: arguments, files: inputFiles, directory: nil, workload: executionBudget.resources?.workloadPolicy)
            guard let graphIndex = arguments.firstIndex(of: "--execute-dataflow"), graphIndex + 1 < arguments.count else { throw WorkflowFileLeaseError.unsafePath }
            try validateInputPath(arguments[graphIndex + 1], name: "graph.json")
        case .modelTraining:
            try WorkflowModelTrainingToolContract.validate(arguments: arguments, files: inputFiles, directory: nil, workload: executionBudget.resources?.workloadPolicy)
            guard let graphIndex = arguments.firstIndex(of: "--graph"), graphIndex + 1 < arguments.count else { throw WorkflowFileLeaseError.unsafePath }
            try validateInputPath(arguments[graphIndex + 1], name: "graph.json")
        case .modelPreview:
            try WorkflowPreviewToolContract.validate(arguments: arguments, files: inputFiles, directory: nil, workload: executionBudget.resources?.workloadPolicy)
            guard let graphIndex = arguments.firstIndex(of: "--graph"), graphIndex + 1 < arguments.count else { throw WorkflowFileLeaseError.unsafePath }
            try validateInputPath(arguments[graphIndex + 1], name: "graph.json")
        case .learningRuntime:
            guard remainingSeconds <= 10, arguments == ["--learning-runtime"], inputFiles.isEmpty else { throw WorkflowFileLeaseError.unsafePath }
        case .codeExport:
            guard remainingSeconds <= 30, inputFiles.count == 1, arguments.count == 3,
                  (0...2_000_000).contains(inputFiles[0].bytes), WorkflowArchiveCodec.isHash(inputFiles[0].sha256) else { throw WorkflowFileLeaseError.unsafePath }
            let workflow = arguments[0] == "--export-workflow-code"
            let name = workflow ? "workflow-code.json" : "graph.json"
            guard inputFiles[0].path == "input/" + name,
                  arguments[0] == (workflow ? "--export-workflow-code" : "--export-code"),
                  arguments[1] == (workflow ? "--workflow-code-plan" : "--graph") else { throw WorkflowFileLeaseError.unsafePath }
            try validateInputPath(arguments[2], name: name)
        case .learningAssessment:
            let capturedRegistry = inputFiles.contains { $0.path == "input/model-registry.json" }
            let expectedFiles: Set<String> = capturedRegistry ? ["input/assessment.json", "input/submission.json", "input/model-registry.json"] : ["input/assessment.json", "input/submission.json"]
            guard remainingSeconds <= 65, inputFiles.count == expectedFiles.count, Set(inputFiles.map(\.path)) == expectedFiles,
                  inputFiles.allSatisfy({ (0...2_000_000).contains($0.bytes) && WorkflowArchiveCodec.isHash($0.sha256) }),
                  arguments.count == (capturedRegistry ? 8 : 6), arguments[0] == "--learning-evidence-request",
                  arguments[2] == "--learning-submission", arguments[4] == "--learning-submission-sha256",
                  arguments[5] == inputFiles.first(where: { $0.path == "input/submission.json" })?.sha256 else {
                throw WorkflowFileLeaseError.unsafePath
            }
            if capturedRegistry {
                guard arguments[6] == "--registry" else { throw WorkflowFileLeaseError.unsafePath }
                try validateInputPath(arguments[7], name: "model-registry.json")
                guard URL(fileURLWithPath: arguments[7]).deletingLastPathComponent() == URL(fileURLWithPath: arguments[1]).deletingLastPathComponent() else { throw WorkflowFileLeaseError.unsafePath }
            }
            for (index, name) in [(1, "assessment.json"), (3, "submission.json")] {
                try validateInputPath(arguments[index], name: name)
            }
            guard URL(fileURLWithPath: arguments[1]).deletingLastPathComponent() == URL(fileURLWithPath: arguments[3]).deletingLastPathComponent() else {
                throw WorkflowFileLeaseError.unsafePath
            }
        }
    }
    private func validateInputPath(_ path: String, name: String) throws {
        let suffix = "/workflow-operations/\(projectID)/\(operationID)/input/\(name)"
        guard path.hasPrefix("/"), path.hasSuffix(suffix), !path.contains("//"),
              !path.split(separator: "/").contains("."), !path.split(separator: "/").contains("..") else {
            throw WorkflowFileLeaseError.unsafePath
        }
    }
    func validate(directory: URL, verifyInputs: Bool) throws {
        try validate()
        guard directory.isFileURL, directory.lastPathComponent == operationID,
              directory.deletingLastPathComponent().lastPathComponent == projectID,
              directory.deletingLastPathComponent().deletingLastPathComponent().lastPathComponent == "workflow-operations" else {
            throw WorkflowFileLeaseError.unsafePath
        }
        if toolMode == .selfPlay {
            try SelfPlayToolContract.validate(arguments:arguments,files:inputFiles,directory:directory)
            if verifyInputs {try SelfPlayToolContract.verifyInputs(files:inputFiles,directory:directory)}
            return
        }
        if toolMode == .gameSession {
            try GameSessionToolContract.validate(arguments:arguments,files:inputFiles,directory:directory)
            if verifyInputs {try GameSessionToolContract.verifyInputs(files:inputFiles,directory:directory)}
            return
        }
        if toolMode == .modelAsset {
            try ModelAssetToolContract.validate(arguments:arguments,files:inputFiles,directory:directory)
            if verifyInputs {try ModelAssetToolContract.verify(self,directory:directory)}
            return
        }
        if toolMode == .textGeneration {
            try WorkflowGenerationToolContract.validate(arguments:arguments,files:inputFiles,directory:directory)
            if verifyInputs {try WorkflowGenerationToolContract.verifyInputs(request:self,directory:directory)}
            return
        }
        if toolMode == .programExport {
            try WorkflowProgramExportToolContract.validate(arguments:arguments,files:inputFiles,directory:directory)
            if verifyInputs {try WorkflowProgramExportToolContract.verifyInputs(request:self,directory:directory)}
            return
        }
        if toolMode == .backendFit {
            try BackendFitOperationToolContract.validate(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
            if verifyInputs { try BackendFitOperationToolContract.verifyInputs(files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy) }
            return
        }
        if toolMode == .tableViewQuery {
            try WorkflowTableViewQueryToolContract.validate(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
            if verifyInputs { try WorkflowTableViewQueryToolContract.verifyInputs(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy) }
            return
        }
        if toolMode == .preparationInspection {
            try WorkflowPreparationInspectionToolContract.validate(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
            if verifyInputs { try WorkflowPreparationInspectionToolContract.verifyInputs(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy) }
            return
        }
        if toolMode == .sourceInspection {
            try WorkflowSourceInspectionToolContract.validate(arguments: arguments, files: inputFiles, directory: directory)
            if verifyInputs { try WorkflowSourceInspectionToolContract.verifyInputs(arguments:arguments,files: inputFiles, directory: directory) }
            return
        }
        if toolMode == .dataPreparation {
            try WorkflowPreparationToolContract.validate(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
            if verifyInputs { try WorkflowPreparationToolContract.verifyInputs(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy) }
            return
        }
        if toolMode == .modelTraining {
            try WorkflowModelTrainingToolContract.validate(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
            if verifyInputs { try WorkflowModelTrainingToolContract.verifyInputs(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy) }
            return
        }
        if toolMode == .modelPreview {
            try WorkflowPreviewToolContract.validate(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
            if verifyInputs {
                try WorkflowPreviewToolContract.verifyInputs(files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
                try WorkflowPreviewToolContract.verifyPreparedInventory(arguments: arguments, files: inputFiles, directory: directory, workload: executionBudget.resources?.workloadPolicy)
            }
            return
        }
        if toolMode == .mixtureValidation {guard arguments[1]==directory.appendingPathComponent("input/mixture.json").path else{throw WorkflowFileLeaseError.unsafePath}}
        if toolMode == .learningAssessment {
            guard arguments[1] == directory.appendingPathComponent("input/assessment.json").path,
                  arguments[3] == directory.appendingPathComponent("input/submission.json").path else { throw WorkflowFileLeaseError.unsafePath }
        }
        if toolMode == .codeExport {
            let name = arguments[0] == "--export-workflow-code" ? "workflow-code.json" : "graph.json"
            guard arguments[2] == directory.appendingPathComponent("input/" + name).path else { throw WorkflowFileLeaseError.unsafePath }
        }
        if verifyInputs {
            for file in inputFiles {
                let input = directory.appendingPathComponent("input")
                let fd = try WorkflowEvaluationOwnership.openDirectory(input)
                defer { Darwin.close(fd) }
                _ = try WorkflowEvaluationOwnership.directoryIdentity(fd)
                let data = try WorkflowEvaluationOwnership.readProtocol(input, name: URL(fileURLWithPath: file.path).lastPathComponent, limit: 2_000_000)
                guard data.count == file.bytes, WorkflowArchiveCodec.hash(data) == file.sha256 else { throw WorkflowFileLeaseError.unsafePath }
            }
        }
    }
    func encoded() throws -> Data { try validate(); return try WorkflowArchiveCodec.encode(self) }
    func requestSHA256() throws -> String { try WorkflowArchiveCodec.hash(encoded()) }
    static func decode(_ data: Data) throws -> Self {
        guard data.count <= 256_000, let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == ["schemaVersion", "kind", "operationID", "projectID", "requestIdentity", "toolMode", "arguments", "inputFiles", "executionBudget", "remainingSeconds"],
              let files = object["inputFiles"] as? [[String: Any]],
              files.allSatisfy({ Set($0.keys) == ["path", "bytes", "sha256"] }) else { throw WorkflowFileLeaseError.unsafePath }
        let request = try JSONDecoder().decode(Self.self, from: data)
        try request.validate()
        guard let original = object["executionBudget"] as? [String: Any],
              let normalized = try JSONSerialization.jsonObject(with: WorkflowArchiveCodec.encode(request.executionBudget)) as? [String: Any],
              Set(original.keys) == Set(normalized.keys) else { throw WorkflowFileLeaseError.unsafePath }
        return request
    }
}
