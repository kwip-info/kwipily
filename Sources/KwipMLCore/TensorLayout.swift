import Foundation

/// Concrete, bounded shape algebra. This describes logical dimensions, not strides or memory contiguity.
/// Symbolic batches, scalars and empty tensors require separate contracts; no dimensions are guessed.
public struct TensorLayout:Equatable,Sendable {
    public let dimensions:[Int]
    public let elementCount:Int
    public static let maximumRank=8
    public static let maximumElements=100_000_000

    public struct Failure:LocalizedError {
        public let message:String
        public var errorDescription:String? {message}
    }
    public init(_ dimensions:[Int])throws {
        guard (1...Self.maximumRank).contains(dimensions.count) else {
            throw Failure(message:"Tensor layout requires 1 to \(Self.maximumRank) concrete axes.")
        }
        self.elementCount=try Self.product(dimensions)
        self.dimensions=dimensions
    }
    private static func product(_ dimensions:[Int])throws->Int {
        var total=1
        for value in dimensions {
            guard value>0,value<=maximumElements/total else {
                throw Failure(message:"Tensor dimensions must be positive and contain at most \(maximumElements) elements.")
            }
            total*=value
        }
        return total
    }
    private func axis(_ value:Int)throws->Int {
        guard value>=(-dimensions.count),value<dimensions.count else {
            throw Failure(message:"Axis \(value) is outside rank \(dimensions.count). Use -\(dimensions.count) through \(dimensions.count-1).")
        }
        return value<0 ? value+dimensions.count:value
    }
    public func reshaped(_ target:[Int])throws->Self {
        guard (1...Self.maximumRank).contains(target.count),target.filter({$0 == -1}).count<=1 else {
            throw Failure(message:"Reshape requires 1 to \(Self.maximumRank) axes and at most one inferred (-1) dimension.")
        }
        var result=target
        let known=try Self.product(target.filter {$0 != -1})
        if let inferred=target.firstIndex(of:-1) {
            guard elementCount%known == 0 else {throw Failure(message:"Reshape cannot infer an integer dimension from \(elementCount) elements and known product \(known).")}
            result[inferred]=elementCount/known
        }
        let layout=try Self(result)
        guard layout.elementCount == elementCount else {
            throw Failure(message:"Reshape must preserve \(elementCount) elements; the requested shape contains \(layout.elementCount).")
        }
        return layout
    }
    public func transposed(_ first:Int,_ second:Int)throws->Self {
        let a=try axis(first),b=try axis(second)
        var result=dimensions;result.swapAt(a,b)
        return try Self(result)
    }
    public func permuted(_ order:[Int])throws->Self {
        let normalized=try order.map(axis)
        guard normalized.count == dimensions.count,Set(normalized).count == dimensions.count else {
            throw Failure(message:"Permutation must contain each axis exactly once.")
        }
        return try Self(normalized.map {dimensions[$0]})
    }
    public func flattened(from start:Int,through end:Int)throws->Self {
        let first=try axis(start),last=try axis(end)
        guard first<=last else {throw Failure(message:"Flatten's first axis must precede or equal its last axis.")}
        let merged=try Self.product(Array(dimensions[first...last]))
        return try Self(Array(dimensions[..<first])+[merged]+Array(dimensions[(last+1)...]))
    }
}
