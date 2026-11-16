import Foundation

/// Explicit learned-slope, logarithmic-axis and parent-mode dropout controls.
/// Observation batches and source types are never implicitly converted.
public struct RemainingActivationConfiguration:Equatable,Sendable {
    public static let componentIDs:Set<String>=["prelu","log_softmax","dropout"]
    public let componentID:String
    public let normalizedValues:[String:JSONValue]
    public var values:[String:JSONValue] {normalizedValues}
    public let numParameters:Int?
    public let initialSlope:Double?
    public let dim:Int?
    public let probability:Double?
    public init(componentID:String,values:[String:JSONValue])throws {
        guard Self.componentIDs.contains(componentID) else {throw GraphConfigurationError("Unknown activation component.")}
        self.componentID=componentID
        var count:Int?,slope:Double?,axis:Int?,chance:Double?,canonical:[String:JSONValue]
        switch componentID {
        case "prelu":
            guard Set(values.keys).isSubset(of:["num_parameters","init"]),
                  let n=(values["num_parameters"] ?? .integer(1)).integerValue,(1...4096).contains(n),
                  let initial=(values["init"] ?? .number(0.25)).doubleValue,
                  abs(initial)<=Double(Float.greatestFiniteMagnitude) else {
                throw GraphConfigurationError("Choose 1 to 4096 slopes and a finite 32-bit initial slope. The count must match channels or use one shared slope.")
            }
            count=n;slope=initial;canonical=["num_parameters":.integer(Int64(n)),"init":.number(initial == 0 ? 0:initial)]
        case "log_softmax":
            guard Set(values.keys).isSubset(of:["dim"]),
                  let a=(values["dim"] ?? .integer(-1)).integerValue,a != 0,(-8...7).contains(a) else {
                throw GraphConfigurationError("Choose an integer sample direction from -8 to 7; direction 0 holds independent observations.")
            }
            axis=a;canonical=["dim":.integer(Int64(a))]
        default:
            guard Set(values.keys).isSubset(of:["p"]),let p=(values["p"] ?? .number(0.5)).doubleValue,(0...1).contains(p) else {
                throw GraphConfigurationError("Drop probability must be a finite number from 0 through 1. Mode and seed belong to the run, not the block.")
            }
            chance=p;canonical=["p":.number(p == 0 ? 0:p)]
        }
        numParameters=count;initialSlope=slope;dim=axis;probability=chance;normalizedValues=canonical
    }
    public func resolved(layout:BatchedTensorLayout)throws->BatchedTensorLayout {
        guard (2...8).contains(layout.rank) else {throw GraphConfigurationError("Connect a nonempty batch with 1 to 7 sample directions.")}
        if componentID == "prelu" {
            guard numParameters == 1 || numParameters == layout.sample.dimensions.first else {
                throw GraphConfigurationError("Use one shared slope or \(layout.sample.dimensions[0]) channel slopes. Channels are the first sample direction, not the last columns.")
            }
        } else if componentID == "log_softmax" {
            return try layout.softmax(axis:dim!)
        }
        return layout
    }
    public func resolved(input:TypedTensorContract)throws->TypedTensorContract {
        guard input.dtype == .float32 else {throw GraphConfigurationError("This activation requires 32-bit decimal arrays; convert the source explicitly.")}
        return .init(layout:try resolved(layout:input.layout),dtype:input.dtype)
    }
    public func parameterCount(for layout:BatchedTensorLayout)throws->Int {
        _ = try resolved(layout:layout)
        return numParameters ?? 0
    }
}
