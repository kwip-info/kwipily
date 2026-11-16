import Foundation
import KwipMLCore

/// Writes into an exclusively owned run directory. Failure may leave partial
/// evidence; the execution host owns terminal reporting and directory lifetime.
enum WorkflowArchiveCaptureWriter {
    struct Result {
        let reference:WorkflowArchiveFile
        let sourceInputs:[String:String]
    }
    static func writeInBackground(request:WorkflowExecutionRequest,runID:UUID,
        sources:[WorkflowNodeAddress:WorkflowCapturedSource],directory:URL,
        modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,
        checkpointFiles:[WorkflowArchiveFile]?=nil) async throws->Result {
        try Task.checkCancellation()
        let worker=Task.detached(priority:.userInitiated) {
            try write(request:request,runID:runID,sources:sources,directory:directory,
                modelRegistry:modelRegistry,dataRegistry:dataRegistry,checkpointFiles:checkpointFiles)
        }
        let result=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
        try Task.checkCancellation()
        return result
    }
    static func write(request:WorkflowExecutionRequest,runID:UUID,
        sources:[WorkflowNodeAddress:WorkflowCapturedSource],directory:URL,
        modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry,
        checkpointFiles:[WorkflowArchiveFile]?=nil)throws->Result {
        try Task.checkCancellation()
        if let selected = request.plan.gradientPolicyRecipe {
            guard request.executionBudget != nil, let permission = request.derivativeCapture else {
                throw WorkflowPhaseError("Selected gradient policies require an explicit derivative permission and execution budget before archive publication.")
            }
            try permission.validate(gradientPolicy:selected.configuration)
        } else if request.derivativeCapture != nil {
            throw WorkflowPhaseError("Derivative permission requires a selected gradient policy.")
        }
        try WorkflowArchivePlan(request.plan, compilerContractVersion: request.archiveCompilerContractVersion).validateModelSnapshot(
            registry: modelRegistry, context: request.plan.lossRecipe == nil ? .preparedWorkflow : .explicitObjective)
        guard Set(sources.keys) == Set(request.plan.sourceAddresses) else {throw WorkflowPhaseError("Archive sources must match the captured plan.")}
        var sourceInputs:[String:String]=[:]
        guard sources.count <= 128 else { throw WorkflowPhaseError("Too many captured source snapshots.") }
        var declaredTotal = 0
        let aggregateLimit = WorkflowCapturedSourceArchive.aggregateLimit(hasCollections: sources.values.contains { $0.isFileCollection })
        for source in sources.values {
            let count = try WorkflowCapturedSourceArchive.declaredByteCount(source)
            guard count <= aggregateLimit - declaredTotal else {
                throw WorkflowPhaseError("Captured source snapshots exceed their byte budget.")
            }
            declaredTotal += count
        }
        func publish(_ bytes: Data, path: String, limit: Int) throws -> WorkflowArchiveFile {
            try Task.checkCancellation()
            guard bytes.count <= limit else { throw WorkflowPhaseError("Captured archive file exceeds its byte limit.") }
            try DiagnosticFileIO.writeImmutable(bytes, root: directory, parts: path.split(separator: "/").map(String.init), limit: limit)
            return .init(path: path, bytes: bytes.count, sha256: WorkflowArchiveCodec.hash(bytes))
        }
        let workflow = try publish(WorkflowCodec.encode(request.document), path: "workflow.json", limit: 2_000_000)
        let modelRegistryFile = try publish(WorkflowArchiveCodec.encode(modelRegistry), path: "model-registry.json", limit: 2_000_000)
        let dataRegistryFile = try publish(WorkflowArchiveCodec.encode(dataRegistry), path: "data-registry.json", limit: 2_000_000)
        guard try WorkflowArchiveCodec.registryHash(modelRegistry) == request.modelRegistrySHA256,
              try WorkflowArchiveCodec.registryHash(dataRegistry) == request.dataRegistrySHA256 else {
            throw WorkflowPhaseError("Captured registries differ from the original request.")
        }
        let runtimeEvidence = try publish(WorkflowArchiveCodec.encode(request.runtimeEvidence), path: "runtime-evidence.json", limit: 256_000)
        var archivedSources: [WorkflowArchiveSource] = []
        for (index, address) in request.plan.sourceAddresses.sorted(by: { $0.localID < $1.localID }).enumerated() {
            let source = sources[address]!
            let file = try WorkflowCapturedSourceArchive.publish(source, address: address, index: index, root: directory, prefix: "inputs/")
            archivedSources.append(file)
            sourceInputs[address.localID] = directory.appendingPathComponent(file.path).path
        }
        let capture = WorkflowArchiveCapture(schemaVersion: request.document.schemaVersion, compilerContractVersion: request.archiveCompilerContractVersion, projectID: request.lease.projectID, directoryID: directory.lastPathComponent,
            runID: runID.uuidString, requestIdentity: request.requestIdentity, revision: request.document.revision, train: request.train,
            workflow: workflow, modelRegistry: modelRegistryFile, dataRegistry: dataRegistryFile,
            modelRegistrySHA256: request.modelRegistrySHA256, dataRegistrySHA256: request.dataRegistrySHA256,
            runtimeEvidence: runtimeEvidence, sources: archivedSources, plan: try WorkflowArchivePlan(request.plan, compilerContractVersion: request.archiveCompilerContractVersion), priorCache: request.priorCache?.reference, checkpointParent: request.checkpointParent, checkpointFiles: checkpointFiles,
            captureFormatVersion: [13,14,15,16,17,18,19,20,21,22].contains(request.document.schemaVersion) ? 8 : request.document.schemaVersion == 12 ? 7 : (request.document.schemaVersion == 11 ? 6 : (request.plan.gradientPolicyRecipe != nil ? 5 : (request.preparationCacheLimitBytes != nil ? 4 : (request.sampleCapture != nil ? 3 : (request.executionBudget == nil ? nil : 2))))),
            executionBudget: request.executionBudget, sampleCapture: request.sampleCapture, preparationCacheLimitBytes: request.preparationCacheLimitBytes, derivativeCapture: request.derivativeCapture, moduleCompositionDescriptor: request.moduleCompositionDescriptor)
        let reference = try publish(WorkflowArchiveCodec.encodeCapture(capture), path: "capture.json", limit: 8_000_000)
        try Task.checkCancellation()
        return .init(reference:reference,sourceInputs:sourceInputs)
    }
}
