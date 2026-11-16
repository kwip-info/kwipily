import Foundation
import KwipMLCore

struct WorkflowArchiveGradientPolicyRecipe:Codable,Equatable {
    let source:WorkflowNodeAddress
    let configuration:JSONValue
    init(_ recipe:GradientPolicyRecipe)throws {
        source=recipe.source;configuration=recipe.configuration.recipeValue;try validate()
    }
    init(from decoder:Decoder)throws {
        let fields=try GradientArchiveRecipes.envelope(JSONValue(from:decoder))
        source=try JSONDecoder().decode(WorkflowNodeAddress.self,from:WorkflowArchiveCodec.encode(fields["source"]!))
        let raw=fields["configuration"]!,parsed=try GradientPolicyConfiguration.decodeRecipe(raw)
        guard try WorkflowArchiveCodec.encode(raw)==WorkflowArchiveCodec.encode(parsed.recipeValue) else {throw WorkflowArchiveError("Captured gradient policy must retain canonical complete controls and request order.")}
        configuration=parsed.recipeValue;try validate()
    }
    func validate()throws {
        try GradientArchiveRecipes.source(source)
        _ = try GradientPolicyConfiguration.decodeRecipe(configuration)
    }
}
struct WorkflowArchiveTrainabilityRecipe:Codable,Equatable {
    let source:WorkflowNodeAddress
    let configuration:JSONValue
    init(_ recipe:TrainabilityRecipe)throws {
        source=recipe.source;configuration=recipe.configuration.recipeValue;try validate()
    }
    init(from decoder:Decoder)throws {
        let fields=try GradientArchiveRecipes.envelope(JSONValue(from:decoder))
        source=try JSONDecoder().decode(WorkflowNodeAddress.self,from:WorkflowArchiveCodec.encode(fields["source"]!))
        let raw=fields["configuration"]!,parsed=try TrainabilityConfiguration.decodeRecipe(raw)
        guard try WorkflowArchiveCodec.encode(raw)==WorkflowArchiveCodec.encode(parsed.recipeValue) else {throw WorkflowArchiveError("Captured trainability must retain canonical complete controls and owner order.")}
        configuration=parsed.recipeValue;try validate()
    }
    func validate()throws {
        try GradientArchiveRecipes.source(source)
        _ = try TrainabilityConfiguration.decodeRecipe(configuration)
    }
}
/// Structural captured identity only. No today's built-in registry fallback:
/// configured ports/dtypes are verified by compilation with the pinned registry,
/// and actual native parameter membership/aliases remain worker-owned facts.
enum GradientArchiveRecipes {
    static func envelope(_ raw:JSONValue)throws->[String:JSONValue] {
        guard case .object(let fields)=raw,Set(fields.keys)==["source","configuration"],
              case .object(let address)=fields["source"],Set(address.keys)==["domain","localID"] else {
            throw WorkflowArchiveError("Captured gradient/trainability recipes require exact qualified source and configuration fields.")
        }
        return fields
    }
    static func source(_ address:WorkflowNodeAddress)throws {
        guard address.domain == .control,!address.localID.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,
              !address.localID.contains("\0"),address.localID.utf8.count<=128 else {
            throw WorkflowArchiveError("Captured gradient/trainability recipes require a bounded qualified control source.")
        }
    }
    static func validate(policy:WorkflowArchiveGradientPolicyRecipe?,trainability:WorkflowArchiveTrainabilityRecipe?,graph:GraphDocument,origins:[WorkflowArchiveOrigin],otherSources:[WorkflowNodeAddress],objectiveReduction:String,scalarization:String,inlineTrain:WorkflowNodeAddress?=nil,inlineRoles:Set<String>=[])throws {
        guard policy != nil || trainability != nil else {return}
        let sources:[String:WorkflowNodeAddress?]=["gradientPolicy":policy?.source,"trainability":trainability?.source]
        var foreign=Set<WorkflowNodeAddress>()
        for (role,source) in sources {
            if inlineRoles.contains(role) {
                guard let source,let inlineTrain,source == inlineTrain else {throw WorkflowArchiveError("Inline gradient controls require their exact Train origin.")}
            } else if let source {
                guard source != inlineTrain,!otherSources.contains(source),foreign.insert(source).inserted else {
                    throw WorkflowArchiveError("Gradient/trainability controls require distinct qualified sources from Train and other selected recipes.")
                }
            }
        }
        try policy?.validate();try trainability?.validate()
        let ids=Set(graph.nodes.map(\.id)),originIDs=Set(origins.map(\.nodeID))
        guard ids.count==graph.nodes.count,originIDs.count==origins.count else {throw WorkflowArchiveError("Captured expanded model identities must be unique.")}
        func node(_ value:JSONValue?)throws->GraphNode {
            guard let id=value?.stringValue,originIDs.contains(id),let node=graph.nodes.first(where:{$0.id==id}) else {
                throw WorkflowArchiveError("Selected gradient/trainability owner is outside the captured expanded model and origin inventory.")
            }
            return node
        }
        if let trainability {
            let config=try TrainabilityConfiguration.decodeRecipe(trainability.configuration)
            for value in config.owners {
                guard case .object(let owner)=value else {throw WorkflowArchiveError("Captured owner must be explicit.")}
                _ = try node(owner["node"])
            }
        }
        if let policy {
            let config=try GradientPolicyConfiguration.decodeRecipe(policy.configuration)
            try config.validateObjectiveCompatibility(reduction:objectiveReduction,scalarization:scalarization)
            for value in config.requests {
                guard case .object(let request)=value,case .object(let output)=request["output"],
                      let port=output["port"]?.stringValue,!port.isEmpty,!port.contains("\0"),port.utf8.count<=32_768,
                      case .array(let targets)=request["targets"] else {throw WorkflowArchiveError("Captured derivative request requires explicit bounded endpoints and targets.")}
                _ = try node(output["node"])
                for value in targets {
                    guard case .object(let target)=value else {throw WorkflowArchiveError("Captured derivative target must be explicit.")}
                    let targetNode=try node(target["node"])
                    if target["kind"] == .string("input") {
                        guard ["input","index_input","bag_input"].contains(targetNode.type),let port=target["port"]?.stringValue,
                              !port.isEmpty,!port.contains("\0"),port.utf8.count<=32_768 else {
                            throw WorkflowArchiveError("Input derivatives require a genuine captured model input and exact output endpoint.")
                        }
                        if targetNode.type == "bag_input" {
                            let configuration = try BagSourceConfiguration(values: targetNode.config)
                            guard configuration.outputPorts.contains(port) else {
                                throw WorkflowArchiveError("Bag input derivatives require an enabled captured coordinate port.")
                            }
                        }
                    }
                }
            }
        }
    }
}
