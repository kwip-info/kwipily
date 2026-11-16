import Foundation

/// An unknown positive batch N followed by a concrete sample shape.
/// Axis arguments use full-tensor indexing; axis zero cannot be folded into a sample.
public struct BatchedTensorLayout:Equatable,Sendable {
    public let sample:TensorLayout
    public var rank:Int {sample.dimensions.count+1}
    public var displayDimensions:[String] {["N"]+sample.dimensions.map(String.init)}
    public init(sample:[Int])throws {
        guard sample.count<TensorLayout.maximumRank else {throw TensorLayout.Failure(message:"A batched tensor requires room for its leading batch axis.")}
        self.sample=try TensorLayout(sample)
    }
    private func sampleAxis(_ axis:Int)throws->Int {
        guard axis>=(-rank),axis<rank else {throw TensorLayout.Failure(message:"Axis \(axis) is outside tensor rank \(rank).")}
        let normalized=axis<0 ? axis+rank:axis
        guard normalized>0 else {throw TensorLayout.Failure(message:"This operation preserves observations independently. Axis 0 is the batch; choose a sample axis.")}
        return normalized-1
    }
    public func softmax(axis:Int)throws->Self {
        _ = try sampleAxis(axis)
        return self
    }
    public func reshapedSample(_ dimensions:[Int])throws->Self {
        try Self(sample:sample.reshaped(dimensions).dimensions)
    }
    public func transposed(_ first:Int,_ second:Int)throws->Self {
        try Self(sample:sample.transposed(sampleAxis(first),sampleAxis(second)).dimensions)
    }
    public func flattened(from first:Int,through last:Int)throws->Self {
        try Self(sample:sample.flattened(from:sampleAxis(first),through:sampleAxis(last)).dimensions)
    }
    public func linear(inputFeatures:Int,outputFeatures:Int)throws->Self {
        guard sample.dimensions.last == inputFeatures else {throw TensorLayout.Failure(message:"Linear expects last-axis width \(inputFeatures), received \(sample.dimensions.last!).")}
        var result=sample.dimensions;result[result.count-1]=outputFeatures
        return try Self(sample:result)
    }
    /// Independent per-observation bilinear maps. Unlike matrix multiplication,
    /// neither leading sample axes nor the observation batch may broadcast.
    public func bilinear(with other: Self, input1Features: Int, input2Features: Int, outputFeatures: Int) throws -> Self {
        guard (1...512).contains(input1Features), (1...512).contains(input2Features), (1...512).contains(outputFeatures),
              sample.dimensions.last == input1Features, other.sample.dimensions.last == input2Features else {
            throw TensorLayout.Failure(message: "Bilinear expects first-input width \(input1Features) and second-input width \(input2Features); received \(sample.dimensions.last!) and \(other.sample.dimensions.last!).")
        }
        guard sample.dimensions.dropLast().elementsEqual(other.sample.dimensions.dropLast()) else {
            throw TensorLayout.Failure(message: "Bilinear requires identical leading sample dimensions on both inputs. Only the final feature widths may differ; no axes broadcast.")
        }
        return try Self(sample: Array(sample.dimensions.dropLast()) + [outputFeatures])
    }
    public func concatenated(with other:Self,axis:Int)throws->Self {
        let selected=try sampleAxis(axis)
        guard rank == other.rank else {throw TensorLayout.Failure(message:"Concatenate requires equal ranks; received \(rank) and \(other.rank).")}
        for axis in sample.dimensions.indices where axis != selected && sample.dimensions[axis] != other.sample.dimensions[axis] {
            throw TensorLayout.Failure(message:"Axis \(axis+1) differs: \(sample.dimensions[axis]) versus \(other.sample.dimensions[axis]). Match the other dimensions before joining.")
        }
        var result=sample.dimensions
        // Each concrete dimension is bounded by maximumElements, so this sum cannot overflow Int.
        result[selected]+=other.sample.dimensions[selected]
        return try Self(sample:result)
    }
    public func adding(_ other:Self)throws->Self {
        guard self == other else {throw TensorLayout.Failure(message:"Add requires identical sample shapes; implicit broadcasting is not enabled.")}
        return self
    }
    public func adding(_ other: Self, broadcastingSamples: Bool) throws -> Self {
        guard broadcastingSamples else { return try adding(other) }
        let count = max(sample.dimensions.count, other.sample.dimensions.count)
        let left = Array(repeating: 1, count: count - sample.dimensions.count) + sample.dimensions
        let right = Array(repeating: 1, count: count - other.sample.dimensions.count) + other.sample.dimensions
        var result: [Int] = []
        for (index, pair) in zip(left, right).enumerated() {
            let (a,b)=pair
            guard a == b || a == 1 || b == 1 else { throw TensorLayout.Failure(message: "Sample dimension \(index+1) differs: \(a) versus \(b). Use matching sizes or a dimension of 1; batch N never repeats.") }
            result.append(max(a, b))
        }
        return try Self(sample: result)
    }
    public func stacked(with other: Self, axis: Int) throws -> Self {
        guard self == other, rank < TensorLayout.maximumRank else { throw TensorLayout.Failure(message: "Stack needs matching sample shapes and room for a new direction.") }
        let outputRank = rank + 1
        guard (-outputRank..<outputRank).contains(axis) else { throw TensorLayout.Failure(message: "The new direction is outside the output rank.") }
        let normalized = axis < 0 ? axis + outputRank : axis
        guard normalized > 0 else { throw TensorLayout.Failure(message: "Stack preserves observations; choose a sample direction.") }
        var result = sample.dimensions
        result.insert(2, at: normalized - 1)
        return try Self(sample: result)
    }
    public func permuted(_ order: [Int]) throws -> Self {
        guard order.count == rank, order.allSatisfy({ (-rank..<rank).contains($0) }) else { throw TensorLayout.Failure(message: "Direction order must match the incoming rank.") }
        let normalized = order.map { $0 < 0 ? $0 + rank : $0 }
        guard normalized.first == 0, Set(normalized).count == rank else { throw TensorLayout.Failure(message: "Keep observations first and include every direction exactly once.") }
        return try Self(sample: Array(normalized.dropFirst()).map { sample.dimensions[$0 - 1] })
    }
    /// Matrix product within each observation. Only leading sample axes broadcast;
    /// the runtime must align those axes after N, never broadcast the observation axis.
    /// Vector products require a separate scalar/vector contract.
    public func matrixMultiplied(by other:Self)throws->Self {
        let a=sample.dimensions,b=other.sample.dimensions
        guard a.count>=2,b.count>=2 else {
            throw TensorLayout.Failure(message:"Matrix multiplication requires two matrix axes per sample. Reshape vectors explicitly before connecting them.")
        }
        guard a.last == b[b.count-2] else {
            throw TensorLayout.Failure(message:"Matrix multiplication contracts the left final axis (\(a.last!)) with the right penultimate axis (\(b[b.count-2])); these widths must match.")
        }
        let leadingCount=max(a.count,b.count)-2
        let left=Array(repeating:1,count:leadingCount-(a.count-2))+Array(a.dropLast(2))
        let right=Array(repeating:1,count:leadingCount-(b.count-2))+Array(b.dropLast(2))
        var result:[Int]=[]
        for (l,r) in zip(left,right) {
            guard l == r || l == 1 || r == 1 else {
                throw TensorLayout.Failure(message:"Matrix multiplication sample axes must match or be 1 for broadcasting; received \(l) and \(r). The observation batch never broadcasts.")
            }
            result.append(max(l,r))
        }
        return try Self(sample:result+[a[a.count-2],b.last!])
    }
}
