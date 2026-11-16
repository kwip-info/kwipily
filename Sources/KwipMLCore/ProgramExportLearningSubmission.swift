import Foundation

public struct ProgramExportLearningSubmission:Equatable,Codable {
    public let schemaVersion:Int
    public let kind:String
    public let taskID:String
    public let recipe:JSONValue
    public init(task:ProgramExportLearningTask,values:[String:JSONValue])throws {
        let configuration=try ProgramExportConfiguration(values:values)
        schemaVersion=1;kind="programExportLearningAction";taskID=task.rawValue;recipe=configuration.recipeValue
    }
    public var configuration:ProgramExportConfiguration {get throws {try .decodeRecipe(recipe)}}
    public func encoded()throws->Data {let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return try encoder.encode(self)}
    public static func decode(_ data:Data,task:ProgramExportLearningTask)throws->Self {
        guard data.count<=32768 else {throw LearningContractError("Program export lesson controls exceed 32 KiB.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let envelope=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","taskID","recipe"])
        guard envelope["schemaVersion"] == .integer(1),envelope["kind"] == .string("programExportLearningAction"),envelope["taskID"] == .string(task.rawValue) else {throw LearningContractError("Use the requested versioned executable-program lesson.")}
        let configuration=try ProgramExportConfiguration.decodeRecipe(envelope["recipe"]!)
        return try .init(task:task,values:configuration.values)
    }
}
