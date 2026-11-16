import Foundation
import KwipMLCore

/// Self-consistency of a separately pinned parent manifest. This is deliberately
/// weaker than WorkflowDataLoadingVerifier's actual prepared-artifact binding:
/// it does not attest source permission, prepared table content or Torch state.
/// The worker validates current preparation/runtime/binary state before resume.
enum WorkflowDataLoadingParentIdentity {
    private static func require(_ value:Bool)throws {guard value else {throw WorkflowPhaseError("Stored parent data loading identity is inconsistent.")}}
    private static func object(_ raw:JSONValue?)throws->[String:JSONValue] {guard case .object(let value)=raw else {throw WorkflowPhaseError("Stored parent data loading metadata is missing.")};return value}
    private static func ids(_ raw:JSONValue?)throws->[String] {
        guard case .array(let values)=raw,(1...100000).contains(values.count) else {throw WorkflowPhaseError("Stored parent row identities are incomplete.")}
        let strings=try values.map {value->String in guard case .string(let id)=value,!id.isEmpty,id.utf8.count<=1024,!id.contains("\0") else {throw WorkflowPhaseError("Stored parent row identity is invalid.")};return id}
        try require(Set(strings).count==strings.count);return strings
    }
    private static func same(_ raw:JSONValue?,_ expected:JSONValue)throws->Bool {
        guard let raw else {return false};return try WorkflowPhaseArtifacts.encode(raw)==WorkflowPhaseArtifacts.encode(expected)
    }
    static func validate(_ manifest:[String:JSONValue])throws->WorkflowDataLoadingIdentity? {
        guard manifest["dataLoading"] != nil else {
            // Absence preserves minimal historical parent envelopes while the
            // shared guard still rejects any reserved nested loading metadata.
            try WorkflowDataLoadingVerifier.validateAttachment(manifest:manifest,identity:nil);return nil
        }
        let settings=try object(manifest["settings"]),provenance=try object(manifest["provenance"]),checkpoint=try object(manifest["checkpoint"]),contract=try object(checkpoint["contract"]),schedule=try object(contract["schedule"])
        let dataset=try settings["datasetRecipe"].map {try DatasetConfiguration.decodeRecipe($0)}
        let loader=try settings["loaderRecipe"].map {try DataLoaderConfiguration.decodeRecipe($0)}
        try require(dataset != nil || loader != nil)
        let training=try ids(provenance["trainingRowIDs"]),validation=try ids(provenance["validationRowIDs"]),original=training.count
        try require(Set(training).isDisjoint(with:validation) && provenance["trainingSamples"] == .integer(Int64(original)) && provenance["validationSamples"] == .integer(Int64(validation.count)))
        guard case .integer(let rawBatch)=settings["requestedBatchSize"],(1...100000).contains(rawBatch),case .integer(let epochs)=settings["epochs"],(1...10000).contains(epochs) else {throw WorkflowPhaseError("Stored parent batching or epoch range is invalid.")}
        let batch=Int(rawBatch)
        if let loader {try require(loader.batchSize==batch)}
        let positions:[Int]
        if dataset?.kind == "subset" {positions=dataset!.indices}
        else if dataset?.kind == "boundedIterable" {positions=Array(0..<min(original,dataset!.maximumRows))}
        else {positions=Array(0..<original)}
        try require(!positions.isEmpty && positions.allSatisfy({(0..<original).contains($0)}))
        let selected=positions.count,retained=loader?.dropLast == true ? selected/batch*batch:selected
        try require(retained>0)
        let dropped=selected-retained,m=(retained-1)/batch+1,effective=min(retained,batch),last=retained%batch == 0 ? effective:retained%batch
        let digest=try WorkflowDataLoadingVerifier.selectionSHA256(positions:positions,rowIDs:positions.map {training[$0]})
        let expected:JSONValue = .object(["schemaVersion":1,"datasetRecipe":dataset?.recipeValue ?? .null,"loaderRecipe":loader?.recipeValue ?? .null,"orderVersion":.string(WorkflowDataLoadingVerifier.orderVersion),"originalTrainingRows":.integer(Int64(original)),"selectedRows":.integer(Int64(selected)),"retainedRows":.integer(Int64(retained)),"droppedRows":.integer(Int64(dropped)),"batchSize":.integer(rawBatch),"batchesPerEpoch":.integer(Int64(m)),"selectionSHA256":.string(digest)])
        try require(try same(manifest["dataLoading"],expected) && same(provenance["dataLoading"],expected) && same(contract["dataLoading"],expected))
        try require(settings["batchSize"] == .integer(Int64(effective)) && settings["batchesPerEpoch"] == .integer(Int64(m)) && settings["lastBatchSize"] == .integer(Int64(last)) && schedule["shuffle"] == .bool(loader?.shuffle ?? false) && schedule["dropLast"] == .bool(loader?.dropLast ?? false))
        try require(manifest["schemaVersion"] == .integer(1) && manifest["status"] == .string("completed") && checkpoint["completedEpoch"] == .integer(epochs))
        return .init(originalRows:original,selectedRows:selected,retainedRows:retained,droppedRows:dropped,requestedBatchSize:batch,effectiveBatchSize:effective,lastBatchSize:last,batchesPerEpoch:m,selectionSHA256:digest,value:expected)
    }
}
