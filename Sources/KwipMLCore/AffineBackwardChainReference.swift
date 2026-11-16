import Foundation

/// Public forward coefficients and the analytical vector-Jacobian product, never an inverse map.
public struct AffineBackwardChainReference:Equatable {
    public let forward:AffineCompositionReference
    public init(forward:AffineCompositionReference) {self.forward=forward}
    public var instructions:String {
        "For the fixed forward computation y = W2(W1x + b1) + b2, build the backward map from upstream gradient g [batch,\(forward.second.outputDimension)] to input gradient [batch,\(forward.first.inputDimension)]. In row convention the result is g W2 W1. Traverse the second map before the first; PyTorch Linear uses row inputs times weight transpose, so choose transposes accordingly. Forward bias shifts y but its derivative with respect to x is zero: do not add bias to the gradient. This is a vector-Jacobian product, not an inverse or a training update. Any equivalent affine construction is accepted."
    }
    public var taskText:String {
        func matrix(_ t:LearningTensor)->String {(0..<t.shape[0]).map{Array(t.values[($0*t.shape[1])..<(($0+1)*t.shape[1])]).description}.joined(separator:", ")}
        return instructions + " W1 [\(forward.first.outputDimension),\(forward.first.inputDimension)] = [\(matrix(forward.first.weight))], b1 [\(forward.first.outputDimension)] = \(forward.first.bias.values); W2 [\(forward.second.outputDimension),\(forward.second.inputDimension)] = [\(matrix(forward.second.weight))], b2 [\(forward.second.outputDimension)] = \(forward.second.bias.values)."
    }
    public func apply(_ upstream:[Double])throws->[Double] {
        guard upstream.count == forward.second.outputDimension,upstream.allSatisfy(\.isFinite) else {throw LearningContractError("Upstream gradient must match the forward output width and contain finite values.")}
        let w2=forward.second.weight,w1=forward.first.weight
        let hidden=(0..<forward.second.inputDimension).map {column in
            (0..<forward.second.outputDimension).reduce(0.0){$0+upstream[$1]*w2.values[$1*forward.second.inputDimension+column]}
        }
        let result=(0..<forward.first.inputDimension).map {column in
            (0..<forward.first.outputDimension).reduce(0.0){$0+hidden[$1]*w1.values[$1*forward.first.inputDimension+column]}
        }
        guard result.allSatisfy(\.isFinite) else {throw LearningContractError("Backward reference exceeded finite bounds.")}
        return result
    }
}
extension LearningExamples {
    public static let affineBackwardReference=AffineBackwardChainReference(forward:affineReference)
}
