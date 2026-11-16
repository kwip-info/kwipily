import Foundation

/// Small independent row-major reference for teaching values and the transpose
/// of the transform's Jacobian. No PyTorch execution or storage-stride claims.
public struct TensorTransformLessonReference {
    public let output:LearningTensor
    private let inputShape:[Int]
    private let sourceIndices:[Int]
    public init(input:LearningTensor,operation:String,configuration:[String:JSONValue])throws {
        let concrete=try TensorLayout(input.shape)
        guard input.shape.count>=2,concrete.elementCount<=4096,input.values.count == concrete.elementCount,input.values.allSatisfy(\.isFinite) else {
            throw LearningContractError("Transform reference requires a finite batched tensor with at most 4096 values and an exact shape/value count.")
        }
        let config=try TensorTransformConfiguration(operation:operation,values:configuration)
        let layout=try config.outputLayout(for:BatchedTensorLayout(sample:Array(input.shape.dropFirst())))
        let shape=[input.shape[0]]+layout.sample.dimensions
        var mapping=Array(0..<input.values.count)
        if operation == "transpose" {
            let rank=input.shape.count
            let rawA=config.values["dim0"]!.integerValue!,rawB=config.values["dim1"]!.integerValue!
            let a=rawA<0 ? rawA+rank:rawA,b=rawB<0 ? rawB+rank:rawB
            // Decode each output coordinate, swap its axes back, then encode
            // the source coordinate. Distinct values expose permutation mistakes.
            for flat in mapping.indices {
                var remainder=flat,coordinates=Array(repeating:0,count:rank)
                for axis in (0..<rank).reversed() {coordinates[axis]=remainder%shape[axis];remainder/=shape[axis]}
                coordinates.swapAt(a,b)
                var source=0
                for axis in 0..<rank {source=source*input.shape[axis]+coordinates[axis]}
                mapping[flat]=source
            }
        }
        self.inputShape=input.shape;self.sourceIndices=mapping
        self.output=LearningTensor(shape:shape,values:mapping.map {input.values[$0]})
    }
    public func inputGradient(for upstream:LearningTensor)throws->LearningTensor {
        guard upstream.shape == output.shape,upstream.values.count == sourceIndices.count,upstream.values.allSatisfy(\.isFinite) else {
            throw LearningContractError("Upstream gradient must match the output shape with finite values.")
        }
        var gradient=Array(repeating:0.0,count:sourceIndices.count)
        for index in sourceIndices.indices {gradient[sourceIndices[index]]=upstream.values[index]}
        return LearningTensor(shape:inputShape,values:gradient)
    }
    public static let explanation="Reshape and flatten preserve logical row-major element order while changing how indices group those values. Transpose swaps axes and generally changes that order. Equal output shapes do not imply equal values. Each operation here is a bijection y = P x on flattened coordinates, with no learned weights. Backpropagation routes an upstream gradient by Pᵀ: each input receives the gradient of its corresponding output. The batch axis is preserved, so observations never mix. Logical ordering does not promise a view, shared storage or a contiguous result; reshape may copy a noncontiguous tensor."
}
