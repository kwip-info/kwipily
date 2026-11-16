import Foundation

/// Spatial settings follow the named axis order; geometry uses the shared pooling contract.
public struct MaxPool3dConfiguration: Equatable, Sendable {
    private let spatial:SpatialMaxPoolConfiguration
    public init(values:[String:JSONValue]) throws {
        spatial=try .init(values:values,axisNames:["depth","height","width"])
    }
    public var kernelSize:[Int] {spatial.kernelSize}
    public var stride:[Int]? {spatial.stride}
    public var padding:[Int] {spatial.padding}
    public var dilation:[Int] {spatial.dilation}
    public var ceilMode:Bool {spatial.ceilMode}
    public var returnIndices:Bool {spatial.returnIndices}
    public var effectiveStride:[Int] {spatial.effectiveStride}
    public var values:[String:JSONValue] {spatial.values}
    public struct Resolved:Equatable,Sendable {
        public let outputLayout:BatchedTensorLayout
        public let indicesLayout:BatchedTensorLayout?
    }
    public func resolved(input:BatchedTensorLayout) throws -> Resolved {
        let result=try spatial.resolved(input:input)
        return .init(outputLayout:result.outputLayout,indicesLayout:result.indicesLayout)
    }
}
