import Foundation

public enum CheckpointLearningParent:String,Codable,CaseIterable {case none,savedLessonRun}
public struct CheckpointLearningConfiguration:Codable,Equatable {
    public var trainingMode:WorkflowTrainingMode
    public var parent:CheckpointLearningParent
    public var epochs:Int
    public var batchSize:Int
    public init(trainingMode:WorkflowTrainingMode = .fresh,parent:CheckpointLearningParent = .none,epochs:Int = 1,batchSize:Int = 4)throws {
        self.trainingMode=trainingMode;self.parent=parent;self.epochs=epochs;self.batchSize=batchSize;try validate()
    }
    public init(from decoder:Decoder)throws {
        let value=try JSONValue(from:decoder)
        let fields=try TensorLearningSubmission.checkedObject(value,required:["trainingMode","parent","epochs","batchSize"])
        guard let mode=fields["trainingMode"]?.stringValue.flatMap(WorkflowTrainingMode.init(rawValue:)),let parent=fields["parent"]?.stringValue.flatMap(CheckpointLearningParent.init(rawValue:)),case .integer(let rawEpochs)=fields["epochs"],let epochs=Int(exactly:rawEpochs),case .integer(let rawBatch)=fields["batchSize"],let batch=Int(exactly:rawBatch) else {throw LearningContractError("Use lifecycle choices and whole-number lesson controls.")}
        try self.init(trainingMode:mode,parent:parent,epochs:epochs,batchSize:batch)
    }
    public func validate()throws {
        guard (1...8).contains(epochs),(1...8).contains(batchSize) else {throw LearningContractError("Choose whole-number epochs and batch size from 1 to 8.")}
    }
}
