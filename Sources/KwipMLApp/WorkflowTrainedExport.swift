import Foundation
import KwipMLCore

struct WorkflowTrainedExportPackage {
    let reference: WorkflowHistoryReference
    let runID: String
    let requestIdentity: String
    let modelManifestSHA256: String
    let directory: URL
    let files: [WorkflowArchiveFile]
    let manifestBytes: Data
    let inputContract: TrainedModelInputContract
    var storageLifetime: WorkflowExportStorageLifetime? = nil
    func remove() {
        if let storageLifetime { storageLifetime.remove() }
        else { try? FileManager.default.removeItem(at: directory) }
    }

    static func destination(in parent: URL, folderName: String) throws -> URL {
        try WorkflowExportPublication.destination(in:parent,folderName:folderName)
    }
    func publish(to destination: URL) throws {
        try WorkflowExportPublication.publish(directory:directory,files:files,manifestName:"export-manifest.json",manifestBytes:manifestBytes,to:destination)
    }

}

struct WorkflowTrainedExportManifest: Encodable {
    let schemaVersion = 1
    let kind = "kwipmlTrainedModel"
    let runID: String
    let requestIdentity: String
    let sourceRevision: UInt64
    let modelManifestSHA256: String
    let checkpointContents = "model weights, optimizer state, RNG state and provenance hashes"
    let preprocessing = "Already-prepared inputs required; source rows and fitted preprocessing are not included."
    let files: [WorkflowArchiveFile]
    /// Omitted for historical exports without a scheduler; retains qualified canvas provenance.
    var schedulerRecipe: WorkflowArchiveSchedulerRecipe? = nil
    /// A fresh starting-value policy; learned-state inference never applies it.
    var initializationRecipe: WorkflowArchiveInitializationRecipe? = nil
    var gradientPolicy: WorkflowExportGradientPolicy? = nil
    var trainability: WorkflowExportTrainability? = nil
}

/// Portable public policy provenance excludes selected derivative requests and upstream arrays.
struct WorkflowExportGradientPolicy: Encodable {
    let source: WorkflowNodeAddress
    let recipeSHA256: String
    let accumulationSteps: Int
    let gradientScale: String
    init(_ recipe: GradientPolicyRecipe) throws {
        source = recipe.source
        recipeSHA256 = WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(recipe.configuration.recipeValue))
        accumulationSteps = recipe.configuration.accumulationSteps
        gradientScale = recipe.configuration.gradientScale
    }
}
struct WorkflowExportTrainability: Encodable {
    let source: WorkflowNodeAddress
    let recipeSHA256: String
    let owners: [JSONValue]
    init(_ recipe: TrainabilityRecipe) throws {
        source = recipe.source
        recipeSHA256 = WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(recipe.configuration.recipeValue))
        owners = recipe.configuration.owners
    }
}
