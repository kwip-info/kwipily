import Foundation
import KwipMLCore

/// Native JSON/graph evidence only. Binary update counters remain worker-validated.
struct WorkflowGradientPolicyIdentity:Equatable {
    let gradientPolicy:GradientPolicyConfiguration?
    let trainability:TrainabilityConfiguration?
    let fullOwnership:[JSONValue]
    let optimizerOwnership:[JSONValue]
    let microbatchesPerEpoch:Int
    let optimizerUpdatesPerEpoch:Int
}

enum WorkflowGradientPolicyVerifier {
    private static func fail()->WorkflowPhaseError {WorkflowPhaseError("Gradient policy, trainability, native ownership or update cadence differs from the captured execution.")}
    private static func require(_ value:Bool)throws {guard value else {throw fail()}}
    private static func object(_ value:JSONValue?)throws->[String:JSONValue] {guard case .object(let o)=value else {throw fail()};return o}
    private static func integer(_ value:JSONValue?,maximum:Int=100_000_000)throws->Int {guard case .integer(let n)=value,n>0,n<=maximum else {throw fail()};return Int(n)}

    /// Captured graph/registry supplies native names/shapes, never manifest-authored ownership.
    static func validate(manifest:[String:JSONValue],gradientPolicy:GradientPolicyConfiguration?,trainability:TrainabilityConfiguration?,derivativeCapture:DerivativeCapturePolicy?,graph:GraphDocument,trainingRows:Int,batchSize:Int,context:GraphValidationContext,capturedRegistry:ComponentRegistry?,dataLoading:WorkflowDataLoadingIdentity? = nil)throws->WorkflowGradientPolicyIdentity {
        let native=try WorkflowLifecycleVerifier.optimizerParameterOwnership(graph,context:context,capturedRegistry:capturedRegistry,initializationOwnership:true)
        let full=try flags(native,graph:graph,trainability:trainability)
        return try verify(manifest:manifest,gradientPolicy:gradientPolicy,trainability:trainability,derivativeCapture:derivativeCapture,fullOwnership:full,trainingRows:trainingRows,batchSize:batchSize,dataLoading:dataLoading)
    }
    private static func flags(_ native:[JSONValue],graph:GraphDocument,trainability:TrainabilityConfiguration?)throws->[JSONValue] {
        var modes:[String:String]=[:]
        for raw in trainability?.owners ?? [] {let owner=try object(raw);let node=owner["node"]!.stringValue!,parameter=owner["parameter"]!.stringValue!;modes[node+"\0"+parameter]=owner["mode"]!.stringValue!}
        try require(Set(graph.nodes.map(\.id)).count == graph.nodes.count)
        let nodes=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,$0)})
        var seen=Set<String>(),names=Set<String>(),elements=0,result:[JSONValue]=[]
        for raw in native {
            var row=try object(raw)
            try require(Set(row.keys)==["node","parameter","name","shape","dtype"])
            guard let node=row["node"]?.stringValue,let parameter=row["parameter"]?.stringValue,let name=row["name"]?.stringValue,let source=nodes[node],case .array(let shape)=row["shape"] else {throw fail()}
            try require(!node.isEmpty && !parameter.isEmpty && !name.isEmpty && !node.contains("\0") && !parameter.contains("\0") && names.insert(name).inserted && row["dtype"] == .string("float32") && !shape.isEmpty && shape.count<=8)
            var count=1
            for raw in shape {let d=try integer(raw,maximum:200_000);try require(count<=200_000/d);count*=d}
            try require(elements<=200_000-count);elements+=count
            let key=node+"\0"+parameter;try require(seen.insert(key).inserted)
            var enabled = !(["embedding", "embedding_bag"].contains(source.type) && parameter == "weight" && source.config["freeze"] == .bool(true))
            if let mode=modes[key],mode != "inherit" {enabled=mode == "trainable"}
            row["requiresGrad"] = .bool(enabled);result.append(.object(row))
        }
        try require(Set(modes.keys).isSubset(of:seen) && !result.isEmpty)
        return result
    }
    /// Parent metadata is pinned by the checkpoint inventory. It is checked without today's
    /// registry. Its native tensors/membership are revalidated by the worker on restoration.
    static func storedIdentity(_ manifest:[String:JSONValue],dataLoading:WorkflowDataLoadingIdentity? = nil)throws->WorkflowGradientPolicyIdentity? {
        func optionalObject(_ value:JSONValue?)->[String:JSONValue] {if case .object(let o)=value {return o};return [:]}
        let settings=optionalObject(manifest["settings"]),checkpoint=try object(manifest["checkpoint"]),contract=try object(checkpoint["contract"])
        let provenance=optionalObject(manifest["provenance"]),runtime=optionalObject(manifest["runtime"])
        if contract["gradientPolicy"] == nil && contract["trainability"] == nil {
            try require(settings["gradientPolicyRecipe"] == nil && settings["trainabilityRecipe"] == nil && provenance["gradientPolicy"] == nil && provenance["trainability"] == nil && runtime["updateCadence"] == nil && settings["microbatchesPerEpoch"] == nil && settings["optimizerUpdatesPerEpoch"] == nil && manifest["derivativeCapture"] == nil && manifest["derivativeResults"] == nil && runtime["gradientSummaryContext"] == nil)
            return nil
        }
        let p=try contract["gradientPolicy"].map {try GradientPolicyConfiguration.decodeRecipe(requiredValue(object($0)["recipe"]))}
        let t=try contract["trainability"].map {try TrainabilityConfiguration.decodeRecipe(requiredValue(object($0)["recipe"]))}
        guard p != nil || t != nil else {
            try require(settings["gradientPolicyRecipe"] == nil && settings["trainabilityRecipe"] == nil)
            let provenance=try object(manifest["provenance"]),runtime=try object(manifest["runtime"])
            try require(provenance["gradientPolicy"] == nil && provenance["trainability"] == nil && runtime["updateCadence"] == nil && settings["microbatchesPerEpoch"] == nil && settings["optimizerUpdatesPerEpoch"] == nil && manifest["derivativeCapture"] == nil && manifest["derivativeResults"] == nil)
            return nil
        }
        let graph=try JSONDecoder().decode(GraphDocument.self,from:WorkflowPhaseArtifacts.encode(try requiredValue(manifest["graph"])))
        let full:[JSONValue]
        if t != nil {
            let item=try object(contract["trainability"])
            guard case .array(let inventory)=item["parameterOwnership"] else {throw fail()}
            let withoutFlags=try inventory.map {raw->JSONValue in var row=try object(raw);try require(Set(row.keys)==["node","parameter","name","shape","dtype","requiresGrad"] && row["requiresGrad"]?.boolValue != nil);row.removeValue(forKey:"requiresGrad");return .object(row)}
            full=try flags(withoutFlags,graph:graph,trainability:t)
            try require(full==inventory)
        } else {
            // No flag policy: parent optimizer inventory remains the worker-owned native subset.
            full=[]
        }
        let capture=try manifest["derivativeCapture"].map {try DerivativeCapturePolicy(jsonValue:$0)}
        let rows=try integer(provenance["trainingSamples"]),batch=try integer(settings["batchSize"])
        return try verify(manifest:manifest,gradientPolicy:p,trainability:t,derivativeCapture:capture,fullOwnership:full,trainingRows:rows,batchSize:batch,parentOnly:true,dataLoading:dataLoading)
    }
    private static func requiredValue(_ value:JSONValue?)throws->JSONValue {guard let value else {throw fail()};return value}
    private static func verify(manifest:[String:JSONValue],gradientPolicy:GradientPolicyConfiguration?,trainability:TrainabilityConfiguration?,derivativeCapture:DerivativeCapturePolicy?,fullOwnership:[JSONValue],trainingRows:Int,batchSize:Int,parentOnly:Bool=false,dataLoading:WorkflowDataLoadingIdentity? = nil)throws->WorkflowGradientPolicyIdentity {
        try require((1...100_000).contains(trainingRows) && (1...100_000).contains(batchSize))
        let settings=try object(manifest["settings"]),provenance=try object(manifest["provenance"]),runtime=try object(manifest["runtime"]),checkpoint=try object(manifest["checkpoint"]),contract=try object(checkpoint["contract"])
        guard let torch=runtime["torch"]?.stringValue,!torch.isEmpty,torch.utf8.count<=128 else {throw fail()}
        try require(["cpu","mps"].contains(runtime["device"]?.stringValue ?? ""))
        let storedRuntime=try object(contract["runtime"])
        try require(storedRuntime["torch"] == .string(torch) && storedRuntime["device"] == runtime["device"])
        try WorkflowDataLoadingVerifier.validateAttachment(manifest:manifest,identity:dataLoading)
        if let dataLoading {try require(dataLoading.originalRows==trainingRows && (parentOnly || dataLoading.requestedBatchSize==batchSize))}
        let m=dataLoading?.batchesPerEpoch ?? (trainingRows-1)/batchSize+1,k=gradientPolicy?.accumulationSteps ?? 1,u=(m-1)/k+1
        try require(settings["batchSize"] == .integer(Int64(dataLoading?.effectiveBatchSize ?? min(batchSize,trainingRows))) && settings["batchesPerEpoch"] == .integer(Int64(m)))
        if let p=gradientPolicy {
            guard let permission=derivativeCapture else {throw fail()};try permission.validate(gradientPolicy:p)
            let permissionValue=try JSONDecoder().decode(JSONValue.self,from:WorkflowPhaseArtifacts.encode(permission))
            let schedule:JSONValue = .object(["microbatchesPerEpoch":.integer(Int64(m)),"optimizerUpdatesPerEpoch":.integer(Int64(u)),"accumulationSteps":.integer(Int64(k)),"gradientScale":.string(p.gradientScale)])
            try require(settings["gradientPolicyRecipe"]==p.recipeValue && settings["microbatchesPerEpoch"] == .integer(Int64(m)) && settings["optimizerUpdatesPerEpoch"] == .integer(Int64(u)))
            try require(contract["gradientPolicy"] == .object(["schemaVersion":1,"recipe":p.recipeValue,"schedule":schedule,"checkpointBoundary":.string("completedEpochNoPendingGradients")]))
            try require(provenance["gradientPolicy"] == .object(["schemaVersion":1,"recipe":p.recipeValue,"implementation":.string("kwip.runner.gradient_policy"),"torchVersion":.string(torch),"schedule":schedule]))
            try require(runtime["updateCadence"] == .object(["schemaVersion":1,"unit":.string("optimizerUpdate"),"microbatchesPerEpoch":.integer(Int64(m)),"optimizerUpdatesPerEpoch":.integer(Int64(u)),"accumulationSteps":.integer(Int64(k))]) && manifest["derivativeCapture"]==permissionValue)
            if case .object(let objective)=contract["objective"],let raw=objective["recipe"] {let loss=try LossConfiguration.decodeRecipe(raw);try p.validateObjectiveCompatibility(reduction:loss.values["reduction"]!.stringValue!,scalarization:objective["scalarization"]?.stringValue ?? "mean")} else {try p.validateObjectiveCompatibility(reduction:"mean")}
            if p.requests.isEmpty {try require(manifest["derivativeResults"] == nil)}
        } else {
            try require(derivativeCapture == nil && settings["gradientPolicyRecipe"] == nil && settings["microbatchesPerEpoch"] == nil && settings["optimizerUpdatesPerEpoch"] == nil && contract["gradientPolicy"] == nil && provenance["gradientPolicy"] == nil && runtime["updateCadence"] == nil && manifest["derivativeCapture"] == nil && manifest["derivativeResults"] == nil && runtime["gradientSummaryContext"] == nil)
        }
        var optimizerOwnership:[JSONValue]=[]
        for raw in fullOwnership {var row=try object(raw);if row["requiresGrad"] == .bool(true) {row.removeValue(forKey:"node");row.removeValue(forKey:"parameter");row.removeValue(forKey:"requiresGrad");optimizerOwnership.append(.object(row))}}
        if let t=trainability {
            try require(!optimizerOwnership.isEmpty)
            let names=try optimizerOwnership.map {raw->JSONValue in try requiredValue(object(raw)["name"])}
            let inventory:JSONValue = .array(fullOwnership),nameValues:JSONValue = .array(names)
            try require(settings["trainabilityRecipe"]==t.recipeValue && contract["trainability"] == .object(["schemaVersion":1,"recipe":t.recipeValue,"parameterOwnership":inventory,"optimizerParameterNames":nameValues]))
            try require(provenance["trainability"] == .object(["schemaVersion":1,"recipe":t.recipeValue,"implementation":.string("kwip.runner.trainability"),"torchVersion":.string(torch),"parameterOwnership":inventory,"optimizerParameterNames":nameValues]))
            let optimizer=try object(contract["optimizer"])
            if optimizer["recipe"] != nil {try require(optimizer["parameterOwnership"] == .array(optimizerOwnership))}
        } else {
            try require(settings["trainabilityRecipe"] == nil && contract["trainability"] == nil && provenance["trainability"] == nil)
            if parentOnly,case .object(let opt)=contract["optimizer"],case .array(let ownership)=opt["parameterOwnership"] {optimizerOwnership=ownership}
        }
        return .init(gradientPolicy:gradientPolicy,trainability:trainability,fullOwnership:fullOwnership,optimizerOwnership:optimizerOwnership,microbatchesPerEpoch:m,optimizerUpdatesPerEpoch:u)
    }
}
