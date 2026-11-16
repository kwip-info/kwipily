import Foundation

/// Explicit current-tool authoring for newly opened practices, never a saved-project migration.
enum CurrentPracticeConfiguration {
    static func columns(_ ordered: [(String,String)], missingPolicy: String = "error") -> [String:JSONValue] {
        ["rows":.array(ordered.map { .object(["source":.string($0.0),"included":.bool(true),"outputName":.string($0.0),"targetType":.string($0.1)]) }),
         "newColumns":.string("keep"),"castFailure":.string("error"),"trimText":.bool(false),"missingPolicy":.string(missingPolicy)]
    }
    static func scaledPartitions(_ names: [String]) -> [String:JSONValue] {
        PreprocessConfiguration(columns:names.map { name in
            var column=PreprocessConfiguration.Column(column:name)
            column.missing="error";column.scaling="standard"
            return column
        }).values
    }
    static func setTrain(_ role:String, value:JSONValue, in document:inout WorkflowDocument) throws {
        document=try WorkflowTrainRecipes.migrateAuthoringSchema(document)
        guard let index=document.nodes.firstIndex(where:{$0.address == WorkflowExamples.trainAddress}),case .train(var config)=document.nodes[index].payload else {throw DataFlowContractError("Practice Train is missing.")}
        config[role]=value;document.nodes[index].payload = .train(config:config)
    }
}
