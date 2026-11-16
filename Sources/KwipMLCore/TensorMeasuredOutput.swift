import Foundation

/// Observation tensors and shared parameter tensors have distinct display axes.
public enum TensorMeasuredOutput:String,CaseIterable {
    case sourceWeightsGradient,indices,kernelGradient,output,inputGradient,leftGradient,rightGradient,weightGradient,scaleGradient,biasGradient,tableGradient,tableAfterForward
    public var isParameter:Bool {self == .kernelGradient || self == .weightGradient || self == .scaleGradient || self == .biasGradient || self == .tableGradient || self == .tableAfterForward}
    public var title:String {
        switch self {
        case .sourceWeightsGradient:return "Item-weight influence"
        case .indices:return "Winning input positions"
        case .kernelGradient:return "Shared kernel influence"
        case .output:return "Output values"
        case .inputGradient:return "Input gradient"
        case .leftGradient:return "Left operand gradient"
        case .rightGradient:return "Right operand gradient"
        case .weightGradient:return "Weight gradient"
        case .scaleGradient:return "Scale gradient"
        case .biasGradient:return "Bias gradient"
        case .tableGradient:return "Table gradient"
        case .tableAfterForward:return "Table after forward"
        }
    }
    public func displayTensor(_ tensor:LearningTensor,observation:Int)->LearningTensor? {
        guard tensor.wellFormed else {return nil}
        return isParameter ? tensor : tensor.batchObservation(at:observation)
    }
    public static func available(in fixture:TensorFormativeFixture)->[Self] {
        allCases.filter {fixture.measured[$0.rawValue] != nil && fixture.expected[$0.rawValue] != nil}
    }
}
