import Foundation

/// Real-valued paired rotations; no learned state, batch broadcasting or implicit attention mask.
public struct RotaryPositionConfiguration:Equatable,Sendable {
    public static let defaults:[String:JSONValue]=["sequence_axis":-2,"rotation":.string("full"),"rotary_dim":2,"pairing":.string("adjacent"),"base":10000,"start_position":0,"position_scale":1]
    public let values:[String:JSONValue]
    public init(values supplied:[String:JSONValue])throws {
        guard Set(supplied.keys).isSubset(of:Set(Self.defaults.keys)) else {throw GraphConfigurationError("Rotary position accepts only its declared axis, rotation and position controls.")}
        var v=Self.defaults.merging(supplied){_,new in new}
        for (key,choices) in [("rotation",["full","custom"]),("pairing",["adjacent","halves"])] {
            guard let text=v[key]?.stringValue,choices.contains(text) else {throw GraphConfigurationError("Choose a supported rotation, pairing or position source.")}
        }
        for (key,lo,hi) in [("sequence_axis",-3,2),("rotary_dim",2,4096),("start_position",0,1_000_000)] {
            guard let n=v[key]?.integerValue,(lo...hi).contains(n) else {throw GraphConfigurationError("\(key) requires a bounded whole number.")};v[key] = .integer(Int64(n))
        }
        guard ![0,-1].contains(v["sequence_axis"]!.integerValue!),v["rotary_dim"]!.integerValue!%2==0 else {throw GraphConfigurationError("Sequence axis must exclude observations/features; rotated width must be even.")}
        for (key,lo,hi) in [("base",1.0,1_000_000_000.0),("position_scale",0.0001,10000.0)] {
            guard let n=v[key]?.doubleValue,n.isFinite,(lo...hi).contains(n),key != "base" || n>1 else {throw GraphConfigurationError("Frequency base must exceed one; position scale must be positive and within its finite supported range.")};v[key] = .number(n)
        }
        values=v
    }
    public var configuredInputs:[String] {["query","key"]}
    public func resolved(layouts:[BatchedTensorLayout])throws->AttentionConfiguration.Resolved {
        guard layouts.count==configuredInputs.count else {throw GraphConfigurationError("Connect exactly the Query and Key tensors; positions belong to Configuration.")}
        let q=layouts[0].sample.dimensions,k=layouts[1].sample.dimensions,rank=q.count+1
        let axisValue=values["sequence_axis"]!.integerValue!,axis=(axisValue%rank+rank)%rank
        guard [2,3].contains(q.count),q.count==k.count,(1..<rank-1).contains(axis),[q,k].allSatisfy({$0.allSatisfy{(1...4096).contains($0)}}) else {throw GraphConfigurationError("RoPE needs rank3/4 tensors and a non-batch, non-feature sequence axis.")}
        let t=q[axis-1],width=q.last!,r=values["rotation"] == .string("full") ? width:values["rotary_dim"]!.integerValue!
        guard k[axis-1]==t,k.last==width,r<=width,r%2==0 else {throw GraphConfigurationError("Match Query/Key sequence and feature widths; the rotated prefix must fit and be even.")}
        if rank==4 {guard q[2-axis]<=32,k[2-axis]<=32 else {throw GraphConfigurationError("RoPE supports at most32 heads per operand.")}}
        do {
            let end=values["start_position"]!.integerValue!+t-1
            guard end<=1_000_000,Double(end)/values["position_scale"]!.doubleValue!<=1_000_000 else {throw GraphConfigurationError("Sequential/effective positions exceed one million.")}
        }
        let physical=layouts[0].sample.elementCount+layouts[1].sample.elementCount,work=8*(physical+t*r)
        guard physical<=AttentionConfiguration.maximumElements,work<=AttentionConfiguration.maximumElements else {throw GraphConfigurationError("Rotary position exceeds its bounded tensor/work allowance.")}
        return .init(outputs:["query":layouts[0],"key":layouts[1]],outputDTypes:["query":.float32,"key":.float32],outputElementsPerObservation:physical,scoreElementsPerObservation:0,maskElementsPerObservation:0,workPerObservation:work)
    }
}
