import Foundation
import KwipMLCore

/// A source-qualified reader policy, never authority to read another partition.
struct WorkflowDataLoadingArchiveRecipe:Codable,Equatable {
    let source:WorkflowNodeAddress
    let configuration:JSONValue
    init(_ recipe:DatasetRecipe)throws {try recipe.validate();source=recipe.source;configuration=recipe.configuration.recipeValue}
    init(_ recipe:DataLoaderRecipe)throws {try recipe.validate();source=recipe.source;configuration=recipe.configuration.recipeValue}
    init(from decoder:Decoder)throws {
        let raw=try JSONValue(from:decoder)
        guard case .object(let fields)=raw,Set(fields.keys)==["source","configuration"],case .object(let address)=fields["source"],Set(address.keys)==["domain","localID"] else {throw WorkflowArchiveError("Reader policy requires exact source and configuration fields.")}
        source=try JSONDecoder().decode(WorkflowNodeAddress.self,from:WorkflowArchiveCodec.encode(fields["source"]!))
        let value=fields["configuration"]!
        guard case .object(let recipe)=value else {throw WorkflowArchiveError("Reader configuration must be a recipe object.")}
        if recipe["componentID"] == .string(DatasetConfiguration.componentID) {configuration=try DatasetConfiguration.decodeRecipe(value).recipeValue}
        else if recipe["componentID"] == .string(DataLoaderConfiguration.componentID) {configuration=try DataLoaderConfiguration.decodeRecipe(value).recipeValue}
        else {throw WorkflowArchiveError("Unknown captured reader policy.")}
        try validate()
    }
    func validate(componentID:String?=nil)throws {
        guard source.domain == .control,!source.localID.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,!source.localID.contains("\0"),source.localID.utf8.count<=128,case .object(let fields)=configuration,let id=fields["componentID"]?.stringValue,componentID == nil || componentID==id else {throw WorkflowArchiveError("Reader policy requires its own bounded control source and exact recipe identity.")}
        switch id {
        case DatasetConfiguration.componentID:_ = try DatasetConfiguration.decodeRecipe(configuration)
        case DataLoaderConfiguration.componentID:_ = try DataLoaderConfiguration.decodeRecipe(configuration)
        default:throw WorkflowArchiveError("Unknown captured reader policy.")
        }
    }
}
