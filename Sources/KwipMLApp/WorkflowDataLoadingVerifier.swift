import Foundation
import KwipMLCore

/// Verified JSON population identity over already pinned prepared rows. This does
/// not deserialize Torch checkpoints; binary epoch/RNG state remains the bounded
/// Python checkpoint loader's separate authority.
struct WorkflowDataLoadingIdentity:Equatable {
    let originalRows:Int
    let selectedRows:Int
    let retainedRows:Int
    let droppedRows:Int
    let requestedBatchSize:Int
    let effectiveBatchSize:Int
    let lastBatchSize:Int
    let batchesPerEpoch:Int
    let selectionSHA256:String
    let value:JSONValue
    func optimizerUpdatesPerEpoch(accumulationSteps:Int)throws->Int {
        guard (1...32).contains(accumulationSteps) else {throw WorkflowPhaseError("Use a supported accumulation window.")}
        return (batchesPerEpoch-1)/accumulationSteps+1
    }
}

enum WorkflowDataLoadingVerifier {
    /// Only explicitly captured inline roles may share this exact Train origin.
    static func validateSources(dataset:WorkflowNodeAddress?,loader:WorkflowNodeAddress?,trainOrigin:WorkflowNodeAddress?,inlineRoles:Set<String>) throws {
        try require(inlineRoles.isSubset(of:["dataset","loader"]),"Unknown inline reader role.")
        for (role,source) in [("dataset",dataset),("loader",loader)] {
            if inlineRoles.contains(role) {try require(source != nil && source == trainOrigin && trainOrigin?.domain == .control,"Inline reader origin differs from its captured Train.")}
            else if let source,let trainOrigin {try require(source != trainOrigin,"A legacy reader cannot impersonate Train.")}
        }
        if let dataset,let loader,dataset == loader {
            try require(inlineRoles == ["dataset","loader"] && dataset == trainOrigin,"Dataset and DataLoader require distinct sources unless both are captured inline on Train.")
        }
    }
    /// Pure admission calculation for live contexts, not evidence of execution.
    static func retainedTrainingRows(originalRows:Int,datasetRecipe:DatasetRecipe?,loaderRecipe:DataLoaderRecipe?,capturedBatchSize:Int,trainOrigin:WorkflowNodeAddress?=nil,inlineRoles:Set<String>=[])throws->Int {
        let positions=try selectedPositions(originalRows:originalRows,datasetRecipe:datasetRecipe,loaderRecipe:loaderRecipe,capturedBatchSize:capturedBatchSize,trainOrigin:trainOrigin,inlineRoles:inlineRoles)
        let retained=loaderRecipe?.configuration.dropLast == true ? positions.count/capturedBatchSize*capturedBatchSize:positions.count
        try require(retained>0,"The final-group policy removes every selected row; reduce batch size or retain the short group.")
        return retained
    }
    private static func selectedPositions(originalRows:Int,datasetRecipe:DatasetRecipe?,loaderRecipe:DataLoaderRecipe?,capturedBatchSize:Int,trainOrigin:WorkflowNodeAddress?=nil,inlineRoles:Set<String>=[])throws->[Int] {
        try require((1...100000).contains(originalRows) && (1...100000).contains(capturedBatchSize),"Data loading requires bounded original rows and the captured batch size.")
        try datasetRecipe?.validate();try loaderRecipe?.validate()
        try validateSources(dataset:datasetRecipe?.source,loader:loaderRecipe?.source,trainOrigin:trainOrigin,inlineRoles:inlineRoles)
        if let loaderRecipe {try require(loaderRecipe.configuration.batchSize==capturedBatchSize,"Captured batch size differs from the selected DataLoader.")}
        let dataset=datasetRecipe?.configuration
        let positions:[Int]
        if dataset?.kind == "subset" {positions=dataset!.indices}
        else if dataset?.kind == "boundedIterable" {positions=Array(0..<min(originalRows,dataset!.maximumRows))}
        else {positions=Array(0..<originalRows)}
        try require(!positions.isEmpty && positions.allSatisfy({(0..<originalRows).contains($0)}) && Set(positions).count==positions.count,"Selected indices must address unique original training rows.")
        return positions
    }
    /// Consumers accept only the identity already derived from their prepared
    /// source and bind it to this manifest; absence cannot acquire policy fields.
    static func validateAttachment(manifest:[String:JSONValue],identity:WorkflowDataLoadingIdentity?)throws {
        guard let identity else {
            // Historical parent records need not contain modern settings or a
            // complete training manifest. Only this family's reserved fields
            // are inspected here; existing parent validators keep their scope.
            func fields(_ raw:JSONValue?)->[String:JSONValue] {if case .object(let value)=raw {return value};return [:]}
            let settings=fields(manifest["settings"]),provenance=fields(manifest["provenance"]),checkpoint=fields(manifest["checkpoint"]),contract=fields(checkpoint["contract"])
            try require(manifest["dataLoading"] == nil && provenance["dataLoading"] == nil && contract["dataLoading"] == nil && checkpoint["dataLoadingState"] == nil && settings["datasetRecipe"] == nil && settings["loaderRecipe"] == nil,"Selected data loading requires verified prepared population evidence.")
            return
        }
        let settings=try object(manifest["settings"],"settings"),provenance=try object(manifest["provenance"],"provenance"),checkpoint=try object(manifest["checkpoint"],"checkpoint"),contract=try object(checkpoint["contract"],"checkpoint contract")
        let value=try object(identity.value,"verified identity")
        let dataset=value["datasetRecipe"] == .null ? nil:value["datasetRecipe"]
        let loader=value["loaderRecipe"] == .null ? nil:value["loaderRecipe"]
        try require(try same(settings["datasetRecipe"],dataset) && same(settings["loaderRecipe"],loader),"Saved data loading controls differ from the verified population identity.")
        try require(try same(manifest["dataLoading"],identity.value) && same(provenance["dataLoading"],identity.value) && same(contract["dataLoading"],identity.value),"Data loading identity belongs to different saved metadata.")
    }
    static let orderVersion="kwip.data-loader.epoch-order.v1"
    private static func require(_ valid:Bool,_ message:String)throws {
        guard valid else {throw WorkflowPhaseError(message)}
    }
    private static func object(_ raw:JSONValue?,_ name:String)throws->[String:JSONValue] {
        guard case .object(let fields)=raw else {throw WorkflowPhaseError("Saved data loading \(name) is missing.")};return fields
    }
    private static func same(_ raw:JSONValue?,_ expected:JSONValue?)throws->Bool {
        guard let expected else {return raw == nil}
        guard let raw else {return false}
        return try WorkflowPhaseArtifacts.encode(raw) == WorkflowPhaseArtifacts.encode(expected)
    }
    static func validate(manifest:[String:JSONValue],datasetRecipe:DatasetRecipe?,loaderRecipe:DataLoaderRecipe?,prepared:WorkflowFailedGradientEvidence.PreparedRowCounts,capturedBatchSize:Int,expectedEpochs:Int,trainOrigin:WorkflowNodeAddress?=nil,inlineRoles:Set<String>=[])throws->WorkflowDataLoadingIdentity? {
        let settings=try object(manifest["settings"],"settings"),provenance=try object(manifest["provenance"],"provenance"),checkpoint=try object(manifest["checkpoint"],"checkpoint"),contract=try object(checkpoint["contract"],"checkpoint contract")
        guard datasetRecipe != nil || loaderRecipe != nil else {
            try require(manifest["dataLoading"] == nil && provenance["dataLoading"] == nil && contract["dataLoading"] == nil && checkpoint["dataLoadingState"] == nil && settings["datasetRecipe"] == nil && settings["loaderRecipe"] == nil,"Unselected data loading metadata cannot be attached to this run.")
            return nil
        }
        try datasetRecipe?.validate();try loaderRecipe?.validate()
        try validateSources(dataset:datasetRecipe?.source,loader:loaderRecipe?.source,trainOrigin:trainOrigin,inlineRoles:inlineRoles)
        try require((1...100000).contains(capturedBatchSize) && expectedEpochs>0,"Data loading needs the actual captured batch size and epoch count.")
        let original=prepared.trainingRows,ids=prepared.trainingRowIDs
        try require((1...100000).contains(original) && ids.count==original && Set(ids).count==original && Set(ids).isDisjoint(with:prepared.validationRowIDs),"Prepared training identities are incomplete or overlap held-out rows.")
        try require(manifest["schemaVersion"] == .integer(1) && manifest["status"] == .string("completed") && settings["epochs"] == .integer(Int64(expectedEpochs)),"Data loading is not the requested completed run.")
        try require(provenance["trainingRowIDs"] == .array(ids.map(JSONValue.string)) && provenance["validationRowIDs"] == .array(prepared.validationRowIDs.map(JSONValue.string)) && provenance["trainingSamples"] == .integer(Int64(original)) && provenance["validationSamples"] == .integer(Int64(prepared.validationRows)),"Data loading used different original prepared rows.")
        try require(try same(settings["datasetRecipe"],datasetRecipe?.configuration.recipeValue) && same(settings["loaderRecipe"],loaderRecipe?.configuration.recipeValue),"Saved data loading recipes differ from the captured controls.")
        let dataset=datasetRecipe?.configuration,loader=loaderRecipe?.configuration
        if let loader {try require(loader.batchSize==capturedBatchSize,"Captured batch size differs from the selected DataLoader.")}
        let positions=try selectedPositions(originalRows:original,datasetRecipe:datasetRecipe,loaderRecipe:loaderRecipe,capturedBatchSize:capturedBatchSize,trainOrigin:trainOrigin,inlineRoles:inlineRoles)
        let selected=positions.count,batch=capturedBatchSize
        let retained=loader?.dropLast == true ? selected/batch*batch:selected
        try require(retained>0,"The final-group policy removes every selected row; reduce batch size or retain the short group.")
        let dropped=selected-retained,m=(retained-1)/batch+1,effective=min(batch,retained),last=retained%batch == 0 ? effective:retained%batch
        let digest=try selectionSHA256(positions:positions,rowIDs:positions.map {ids[$0]})
        let expected:JSONValue = .object(["schemaVersion":1,"datasetRecipe":dataset?.recipeValue ?? .null,"loaderRecipe":loader?.recipeValue ?? .null,"orderVersion":.string(orderVersion),"originalTrainingRows":.integer(Int64(original)),"selectedRows":.integer(Int64(selected)),"retainedRows":.integer(Int64(retained)),"droppedRows":.integer(Int64(dropped)),"batchSize":.integer(Int64(batch)),"batchesPerEpoch":.integer(Int64(m)),"selectionSHA256":.string(digest)])
        try require(try same(manifest["dataLoading"],expected) && same(provenance["dataLoading"],expected) && same(contract["dataLoading"],expected),"Saved data loading population or identity differs from the captured prepared selection.")
        try require(settings["requestedBatchSize"] == .integer(Int64(batch)) && settings["batchSize"] == .integer(Int64(effective)) && settings["batchesPerEpoch"] == .integer(Int64(m)) && settings["lastBatchSize"] == .integer(Int64(last)),"Saved batch geometry differs from actual retained training rows.")
        let schedule=try object(contract["schedule"],"schedule")
        try require(schedule["shuffle"] == .bool(loader?.shuffle ?? false) && schedule["dropLast"] == .bool(loader?.dropLast ?? false),"Checkpoint ordering or final-group policy differs from the selected loader.")
        // dataLoadingState is in checkpoint.pt, not this JSON manifest. Its exact
        // epoch boundary is checked by the bounded Python loader before restore.
        return .init(originalRows:original,selectedRows:selected,retainedRows:retained,droppedRows:dropped,requestedBatchSize:batch,effectiveBatchSize:effective,lastBatchSize:last,batchesPerEpoch:m,selectionSHA256:digest,value:expected)
    }
    /// Exact Python ensure_ascii=False compact JSON for this one fixed digest
    /// tuple. No general JSON normalization or parsed-file authentication.
    static func selectionSHA256(positions:[Int],rowIDs:[String])throws->String {
        try require((1...100000).contains(positions.count) && positions.count==rowIDs.count && positions.allSatisfy({(0..<100000).contains($0)}) && Set(positions).count==positions.count && Set(rowIDs).count==rowIDs.count,"Selected row identities are invalid.")
        try require(rowIDs.allSatisfy({!$0.isEmpty && $0.utf8.count<=1024 && !$0.contains("\0")}),"Selected row IDs must be bounded nonempty identifiers.")
        func quoted(_ value:String)->String {
            var result="\""
            for scalar in value.unicodeScalars {
                switch scalar.value {
                case 34:result+="\\\""
                case 92:result+="\\\\"
                case 8:result+="\\b"
                case 9:result+="\\t"
                case 10:result+="\\n"
                case 12:result+="\\f"
                case 13:result+="\\r"
                case 0...31:result+=String(format:"\\u%04x",scalar.value)
                default:result+=String(scalar)
                }
            }
            return result+"\""
        }
        let json="[["+positions.map(String.init).joined(separator:",")+"],["+rowIDs.map(quoted).joined(separator:",")+"]]"
        return WorkflowArchiveCodec.hash(Data(json.utf8))
    }
}
