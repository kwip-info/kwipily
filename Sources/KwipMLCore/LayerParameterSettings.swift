import Foundation

/// Layer-owned starting values and update flags, separate from operator geometry.
/// Compiled into the reviewed recipe envelopes; never sent as arbitrary module kwargs.
public enum LayerParameterSettings {
    public static func validate(_ settings:[String:JSONValue])throws {
        guard settings.count<=128 else {throw GraphConfigurationError("A layer supports at most 128 parameter settings.")}
        for (parameter,value) in settings {
            guard !parameter.isEmpty,parameter.utf8.count<=256,!parameter.contains("\0"),case .object(let row)=value,
                  !row.isEmpty,Set(row.keys).isSubset(of:["initialization","trainability"]) else {throw GraphConfigurationError("Choose starting values and trainability for an exact layer parameter.")}
            if let mode=row["trainability"] {guard let mode=mode.stringValue,["inherit","trainable","frozen"].contains(mode) else {throw GraphConfigurationError("Choose Default, Trainable or Frozen.")}}
            if let raw=row["initialization"] {
                guard case .object(let start)=raw,Set(start.keys)==["algorithm","settings"] else {throw GraphConfigurationError("Starting values need their method and settings.")}
                _ = try InitializationConfiguration(values:["seed":0,"rules":.array([.object(["owner":.object(["node":.string("layer"),"parameter":.string(parameter)]),"algorithm":start["algorithm"]!,"settings":start["settings"]!])])])
            }
        }
        guard try JSONEncoder().encode(settings).count<=32768 else {throw GraphConfigurationError("Layer parameter settings exceed 32 KiB.")}
    }
    public static func settings(for origin:CompositeNodeOrigin,in document:WorkflowDocument)->[String:JSONValue]? {
        if let reference=origin.definition {return document.definitions.first {$0.reference==reference}?.body.nodes.first {$0.id==origin.localNodeID}?.parameterSettings}
        return document.nodes.first {$0.address == .init(domain:.model,localID:origin.localNodeID)}?.parameterSettings
    }
    public static func stripped(_ definitions:[CompositeDefinition])->[CompositeDefinition] {
        definitions.map {definition in var copy=definition;for index in copy.body.nodes.indices {copy.body.nodes[index].parameterSettings=nil};return copy}
    }
    public static func recipes(document:WorkflowDocument,model:ModelExecutionSnapshot,registry:ComponentRegistry,seed:Int,
                               initialization:InitializationRecipe?,trainability:TrainabilityRecipe?,train:WorkflowNodeAddress)throws->(InitializationRecipe?,TrainabilityRecipe?,Set<String>) {
        var starts:[JSONValue]=[],flags:[JSONValue]=[]
        for node in model.graph.nodes {
            guard let origin=model.origins[node.id],let settings=settings(for:origin,in:document) else {continue}
            try validate(settings)
            let choices=try LayerParameterInventory.choices(type:node.type,config:node.config,version:model.graph.schemaVersion,registry:registry)
            for parameter in settings.keys.sorted() {
                guard let choice=choices.first(where:{$0.id==parameter}),case .object(let row)=settings[parameter] else {throw GraphConfigurationError("Layer \(node.name.isEmpty ? node.type:node.name) has a saved override for a parameter that is no longer present: \(parameter).")}
                if case .object(let start)=row["initialization"],let method=start["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(method),choice.shape.count<2 {throw GraphConfigurationError("Xavier and Kaiming require multidimensional weights. Use Normal or Zeros for biases.")}
                if case .object(let start)=row["initialization"] {
                    starts.append(.object(["owner":.object(["node":.string(node.id),"parameter":.string(parameter)]),"algorithm":start["algorithm"]!,"settings":start["settings"]!]))
                }
                if let mode=row["trainability"],mode != .string("inherit") {flags.append(.object(["node":.string(node.id),"parameter":.string(parameter),"mode":mode]))}
            }
        }
        var roles=Set<String>()
        var effectiveStart=initialization,effectiveFlags=trainability
        if !starts.isEmpty {
            let legacy:[JSONValue];if case .array(let rows)=initialization?.configuration.values["rules"] {legacy=rows}else{legacy=[]}
            let overridden=Set(legacy.compactMap {value->String? in guard case .object(let row)=value,case .object(let owner)=row["owner"] else{return nil};return identity(owner)})
            starts.removeAll {value in guard case .object(let row)=value,case .object(let owner)=row["owner"] else{return false};return overridden.contains(identity(owner))}
            let config=try InitializationConfiguration(values:["seed":.integer(initialization?.configuration.seed ?? Int64(seed)),"rules":.array(starts+legacy)])
            // Legacy-only authority retains its source; a combined recipe is derived at Train.
            if !starts.isEmpty {effectiveStart = .init(configuration:config,source:train);roles.insert("initialization")}
        }
        if !flags.isEmpty {
            let legacy=trainability?.configuration.owners ?? []
            let overridden=Set(legacy.compactMap {value->String? in guard case .object(let row)=value else{return nil};return identity(row)})
            flags.removeAll {value in guard case .object(let row)=value else{return false};return overridden.contains(identity(row))}
            if !flags.isEmpty {effectiveFlags = .init(configuration:try TrainabilityConfiguration(values:["owners":.array(flags+legacy)]),source:train);roles.insert("trainability")}
        }
        return(effectiveStart,effectiveFlags,roles)
    }
    private static func identity(_ row:[String:JSONValue])->String {
        let node=row["node"]?.stringValue ?? "",parameter=row["parameter"]?.stringValue ?? ""
        return "\(node.utf8.count):\(node)\(parameter.utf8.count):\(parameter)"
    }
}
