import Foundation

/// Expected differences are zero; actual values come from independently measured worker comparisons.
public struct InitializationLessonReference {
    public let expected:[String:LearningTensor]
    public init(inputs:[String:LearningTensor])throws {
        guard Set(inputs.keys) == ["x","target","firstWeight","firstBias","secondWeight","secondBias"],inputs.values.allSatisfy({$0.wellFormed && $0.values.allSatisfy({abs($0)<=1_000_000})}),let x=inputs["x"],x.shape.count == 2,let first=inputs["firstWeight"],first.shape.count == 2,let second=inputs["secondWeight"],second.shape.count == 2 else {throw LearningContractError("Use the six supplied finite input and learned-array tables.")}
        let i=x.shape[1],h=first.shape[0],o=second.shape[0]
        guard InitializationLearningTask.widths.contains([i,h,o]),x.shape == [6,i],inputs["target"]?.shape == [4,o],first.shape == [h,i],inputs["firstBias"]?.shape == [h],second.shape == [o,h],inputs["secondBias"]?.shape == [o] else {throw LearningContractError("The supplied model and training/held-out rows must use a reviewed exercise shape.")}
        let count=h*i+h+o*h+o
        let shapes=["startingSettings":[count],"predictions":[6,o],"firstAdjustment":[2,count],"repeatIsolation":[2*count+1],"savedReuse":[count+6*o+1]]
        expected=shapes.mapValues {shape in LearningTensor(shape:shape,values:Array(repeating:0,count:shape.reduce(1,*)))}
    }
    public static func label(_ key:String)->String {switch key {case "startingSettings":return "Starting learned arrays";case "predictions":return "Predictions on training and held-out rows";case "firstAdjustment":return "First gradient and adjustment";case "repeatIsolation":return "Repeatability and training random stream";case "savedReuse":return "Saved learned arrays and predictions";default:return key}}
}
