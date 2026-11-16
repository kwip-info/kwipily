import Foundation

/// Independent scalar formulas; no backend implementation supplies expected answers.
public struct ConfigurableActivationLessonReference {
    public let output:LearningTensor
    private let input:LearningTensor,task:ConfigurableActivationLearningTask
    public init(input:LearningTensor,task:ConfigurableActivationLearningTask)throws {
        guard input.wellFormed,!input.shape.isEmpty,input.shape == [input.shape[0]]+task.sampleShape,(1...2).contains(input.shape[0]) else {throw LearningContractError("Activation input must match the public two-row, three-column array.")}
        self.input=input;self.task=task
        output = .init(shape:input.shape,values:input.values.map {x in
            switch task {
            case .silu:return x*Self.sigmoid(x)
            case .leakyReLU:return x>0 ? x:0.2*x
            case .elu:return x>0 ? x:2*expm1(x)
            }
        })
    }
    private static func sigmoid(_ x:Double)->Double {x>=0 ? 1/(1+exp(-x)):exp(x)/(1+exp(x))}
    public func inputGradient(for upstream:LearningTensor)throws->LearningTensor {
        guard upstream.wellFormed,upstream.shape == input.shape else {throw LearningContractError("Activation influence must match the input array.")}
        return .init(shape:input.shape,values:zip(input.values,upstream.values).map {x,g in
            switch task {
            case .silu:let s=Self.sigmoid(x);return g*(s+x*s*(1-s))
            case .leakyReLU:return g*(x>0 ? 1:0.2)
            case .elu:return g*(x>0 ? 1:2*exp(x))
            }
        })
    }
    public static let explanation="Each activation acts on one number x without mixing positions. Let g be the measured influence arriving from the output. SiLU uses s=1/(1+exp(−x)), output x·s and input influence g·[s+x·s·(1−s)]; exp is the exponential function, evaluated with stable branches for large signed inputs. LeakyReLU uses x for x>0 and a·x otherwise, where a is Negative slope 0.2. Its influence is g for x>0 and a·g otherwise. ELU uses x for x>0 and a·(exp(x)−1) otherwise, where a is Alpha 2. Its influence is g for x>0 and a·exp(x)·g otherwise. At exactly zero the reviewed PyTorch convention selects the negative-branch influence for LeakyReLU and ELU; SiLU’s influence factor is one half. None of these arrays are normalized probabilities, and no weights are learned. Lesson measurements use float64; production activations retain their reviewed float32 contract."
}
