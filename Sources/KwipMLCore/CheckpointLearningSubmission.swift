import Foundation

/// The learner selects lifecycle controls; the checker owns every saved artifact and training fixture.
public struct CheckpointLearningSubmission:Codable,Equatable {
    public let schemaVersion:Int
    public let kind:String
    public let taskID:String
    public let configuration:CheckpointLearningConfiguration
    public init(configuration:CheckpointLearningConfiguration,task:CheckpointLearningTask)throws {
        try configuration.validate()
        guard configuration.trainingMode == .fresh || configuration.parent == .savedLessonRun else {throw LearningContractError("Choose the saved lesson run before restoring or continuing.")}
        schemaVersion=1;kind="checkpointLearning";taskID=task.rawValue;self.configuration=configuration
    }
    public func encoded()throws->Data {let e=JSONEncoder();e.outputFormatting=[.sortedKeys,.withoutEscapingSlashes];return try e.encode(self)}
    public static func decode(_ data:Data,task:CheckpointLearningTask)throws->Self {
        guard data.count<=4096 else {throw LearningContractError("Checkpoint lesson controls exceed their bounded format.")}
        let raw=try JSONDecoder().decode(JSONValue.self,from:data)
        let object=try TensorLearningSubmission.checkedObject(raw,required:["schemaVersion","kind","taskID","configuration"])
        guard object["schemaVersion"] == .integer(1),object["kind"] == .string("checkpointLearning"),object["taskID"] == .string(task.rawValue) else {throw LearningContractError("Use the requested checkpoint lesson controls, without answers, paths or state.")}
        let values=try TensorLearningSubmission.checkedObject(object["configuration"]!,required:["trainingMode","parent","epochs","batchSize"])
        guard let mode=values["trainingMode"]?.stringValue.flatMap(WorkflowTrainingMode.init(rawValue:)),let parent=values["parent"]?.stringValue.flatMap(CheckpointLearningParent.init(rawValue:)),case .integer(let epochs)=values["epochs"],case .integer(let batch)=values["batchSize"],let epochCount=Int(exactly:epochs),let batchCount=Int(exactly:batch) else {throw LearningContractError("Use supported lifecycle choices and whole-number epochs/batch size.")}
        return try Self(configuration:.init(trainingMode:mode,parent:parent,epochs:epochCount,batchSize:batchCount),task:task)
    }
}
