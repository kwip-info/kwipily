import Foundation

/// Independent scalar mathematics for the declared task; the worker evaluates the constructed graph.
public struct GELULessonReference:Equatable {
    public let approximateTanh:Bool
    public init(approximateTanh:Bool) {self.approximateTanh=approximateTanh}
    public var formula:String {
        approximateTanh ? "GELU(x) ≈ 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))" : "GELU(x) = 0.5 x (1 + erf(x/√2))"
    }
    public var instructions:String {
        "Construct the elementwise \(approximateTanh ? "tanh-approximate":"exact erf") GELU map from input [batch,2] to output [batch,2]. Use at least one GELU; every GELU must have Tanh approximation \(approximateTanh ? "on":"off"). Equivalent structures are accepted, not one preferred topology. GELU preserves shape, has no trainable parameters, and can produce negative outputs; it is neither ReLU nor Tanh. The current graph adapter requires a Linear component: an identity Linear is one possible scaffold, not a mathematical requirement of GELU. Supply its parameters explicitly. This check evaluates constructed behavior, not training or optimization mastery."
    }
    public var taskText:String {instructions + " " + formula + ". Values near zero have approximately half their input magnitude; the negative branch is not simply clipped to zero."}
    public func apply(_ x:Double)throws->Double {
        guard x.isFinite else {throw LearningContractError("GELU reference input must be finite.")}
        let gate:Double
        if approximateTanh {gate=tanh(sqrt(2/Double.pi)*(x+0.044715*x*x*x))}
        else {gate=erf(x/sqrt(2))}
        let result=0.5*x*(1+gate)
        guard result.isFinite else {throw LearningContractError("GELU reference exceeded finite bounds.")}
        return result
    }
}
extension LearningExamples {
    public static func geluReference(approximateTanh:Bool)->GELULessonReference {GELULessonReference(approximateTanh:approximateTanh)}
}
