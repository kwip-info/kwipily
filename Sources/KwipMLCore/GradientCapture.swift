import Foundation
import CryptoKit

public struct GradientCaptureError:LocalizedError { public let message:String; public var errorDescription:String? {message} }
private func requireGradient(_ condition:Bool,_ message:String)throws {if !condition {throw GradientCaptureError(message:message)}}
private struct GradientKey:CodingKey {var stringValue:String;var intValue:Int? {nil};init?(stringValue:String){self.stringValue=stringValue};init?(intValue:Int){return nil}}
public struct GradientCaptureConfiguration:Codable,Equatable {
    public static let epochCadenceBounds=1...10000
    public static let batchBounds=1...8
    public static let captureBounds=1...1024
    public var schemaVersion:Int;public var everyEpochs:Int;public var maxBatchesPerEpoch:Int;public var maxCaptures:Int
    public init(schemaVersion:Int=1,everyEpochs:Int=1,maxBatchesPerEpoch:Int=1,maxCaptures:Int=256) {self.schemaVersion=schemaVersion;self.everyEpochs=everyEpochs;self.maxBatchesPerEpoch=maxBatchesPerEpoch;self.maxCaptures=maxCaptures}
    public func validate()throws {try requireGradient(schemaVersion == 1 && Self.epochCadenceBounds.contains(everyEpochs) && Self.batchBounds.contains(maxBatchesPerEpoch) && Self.captureBounds.contains(maxCaptures),"Invalid gradient capture configuration.")}
    public init(from decoder:Decoder)throws {
        let c=try decoder.container(keyedBy:GradientKey.self)
        try requireGradient(Set(c.allKeys.map(\.stringValue)) == ["schemaVersion","everyEpochs","maxBatchesPerEpoch","maxCaptures"],"Unknown or missing gradient configuration fields.")
        func integer(_ key:String)throws->Int {try c.decode(Int.self,forKey:GradientKey(stringValue:key)!)}
        self.init(schemaVersion:try integer("schemaVersion"),everyEpochs:try integer("everyEpochs"),maxBatchesPerEpoch:try integer("maxBatchesPerEpoch"),maxCaptures:try integer("maxCaptures"));try validate()
    }
}
public struct GradientCaptureParameter:Codable,Equatable {public var name:String;public var nodeID:String;public var shape:[Int];public var elementCount:Int}
public struct GradientCaptureInventory:Codable,Equatable {public var parameterCount:Int;public var elementCount:Int;public var parameters:[GradientCaptureParameter]}
public struct GradientBatchSummary:Codable,Equatable {
    public var batchIndex:Int;public var trainingRows:Int;public var presentParameterCount:Int;public var presentElementCount:Int;public var missingParameterCount:Int;public var missingElementCount:Int;public var finiteElementCount:Int;public var nonfiniteElementCount:Int;public var zeroElementCount:Int;public var l2Norm:Double?;public var maxAbs:Double?
}
public enum GradientCaptureStatus:String,Codable {case captured,notScheduled,budgetExhausted}
public struct GradientEpochSummary:Codable,Equatable {
    public var event:String;public var schemaVersion:Int;public var runID:String;public var epoch:Int;public var totalBatches:Int;public var scheduledBatchCount:Int;public var observedBatchCount:Int;public var totalTrainingRows:Int;public var observedTrainingRows:Int;public var captureBudgetRemaining:Int;public var status:GradientCaptureStatus;public var batches:[GradientBatchSummary]
}
public struct GradientCaptureStarted:Codable,Equatable {public var schemaVersion:Int;public var config:GradientCaptureConfiguration;public var initialEpoch:Int;public var inventory:GradientCaptureInventory}
public struct GradientCaptureArtifact:Codable,Equatable {
    public var schemaVersion:Int;public var id:String;public var runID:String;public var device:String;public var phase:String;public var config:GradientCaptureConfiguration;public var initialEpoch:Int;public var finalEpoch:Int;public var inventory:GradientCaptureInventory;public var summaries:[GradientEpochSummary]
}
public struct GradientCaptureMetadata:Codable,Equatable {
    public var schemaVersion:Int;public var file:String;public var sha256:String;public var bytes:Int;public var observedEpochCount:Int;public var capturedEpochCount:Int;public var observedBatchCount:Int;public var initialEpoch:Int;public var finalEpoch:Int
}

/// The codec is the untrusted wire boundary; plain Codable is provided for local typed encoding.
public enum GradientCaptureCodec {
    public static let maximumArtifactBytes=8_000_000
    private static func object(_ value:JSONValue,keys:Set<String>)throws->[String:JSONValue] {
        guard case .object(let result)=value,Set(result.keys)==keys else {throw GradientCaptureError(message:"Unknown or missing gradient fields.")};return result
    }
    private static func json(_ data:Data,limit:Int=maximumArtifactBytes)throws->JSONValue {try requireGradient(data.count<=limit,"Gradient document exceeds byte limit.");return try JSONDecoder().decode(JSONValue.self,from:data)}
    private static func array(_ value:JSONValue?)throws->[JSONValue] {guard case .array(let values)=value else {throw GradientCaptureError(message:"Gradient array is missing.")};return values}
    private static func inventoryShape(_ value:JSONValue)throws {
        let v=try object(value,keys:["parameterCount","elementCount","parameters"])
        for p in try array(v["parameters"]) {_ = try object(p,keys:["name","nodeID","shape","elementCount"])}
    }
    private static func eventShape(_ value:JSONValue)throws {
        let v=try object(value,keys:["event","schemaVersion","runID","epoch","totalBatches","scheduledBatchCount","observedBatchCount","totalTrainingRows","observedTrainingRows","captureBudgetRemaining","status","batches"])
        for b in try array(v["batches"]) {_ = try object(b,keys:["batchIndex","trainingRows","presentParameterCount","presentElementCount","missingParameterCount","missingElementCount","finiteElementCount","nonfiniteElementCount","zeroElementCount","l2Norm","maxAbs"])}
    }
    private static func validate(_ inventory:GradientCaptureInventory)throws {
        try requireGradient((1...256).contains(inventory.parameterCount) && inventory.parameterCount == inventory.parameters.count && (1...2_000_000).contains(inventory.elementCount),"Invalid gradient parameter inventory size.")
        var count=0;var names=Set<String>()
        for p in inventory.parameters {
            try requireGradient(!p.nodeID.isEmpty && p.nodeID.unicodeScalars.count<=128 && !p.nodeID.contains(".") && ["layers.\(p.nodeID).weight","layers.\(p.nodeID).bias"].contains(p.name) && names.insert(p.name).inserted,"Invalid or duplicate gradient parameter identity.")
            try requireGradient(p.shape.count == (p.name.hasSuffix(".weight") ? 2:1) && p.shape.allSatisfy{(1...4096).contains($0)},"Invalid gradient parameter shape.")
            try requireGradient(p.shape.reduce(1,*) == p.elementCount && p.elementCount<=2_000_000,"Gradient parameter shape/count differs.")
            count+=p.elementCount
        }
        try requireGradient(count == inventory.elementCount,"Gradient inventory element total differs.")
    }
    private static func validate(_ event:GradientEpochSummary,runID:String,config:GradientCaptureConfiguration,inventory:GradientCaptureInventory,previous:GradientEpochSummary?,initialEpoch:Int)throws {
        try config.validate();try validate(inventory)
        if let previous {try requireGradient(previous.runID == runID && previous.schemaVersion == 1 && previous.event == "gradientSummary" && (1...9999).contains(previous.epoch) && previous.epoch>initialEpoch && (0...config.maxCaptures).contains(previous.captureBudgetRemaining),"Invalid prior gradient coverage context.")}
        try requireGradient(UUID(uuidString:runID) != nil && event.runID == runID && event.schemaVersion == 1 && event.event == "gradientSummary" && (0..<10000).contains(initialEpoch) && event.epoch == (previous?.epoch ?? initialEpoch)+1 && event.epoch<=10000,"Gradient event identity or epoch sequence differs.")
        try requireGradient((1...100000).contains(event.totalTrainingRows) && (1...event.totalTrainingRows).contains(event.totalBatches),"Invalid gradient training counts.")
        let scheduled=(event.epoch-1)%config.everyEpochs == 0 ? min(config.maxBatchesPerEpoch,event.totalBatches):0
        let available=previous?.captureBudgetRemaining ?? config.maxCaptures
        let observed=min(scheduled,available)
        try requireGradient(event.scheduledBatchCount == scheduled && event.observedBatchCount == observed && event.batches.count == observed && event.captureBudgetRemaining == available-observed,"Gradient cadence, count or budget differs.")
        let status:GradientCaptureStatus=scheduled == 0 ? .notScheduled:(observed == 0 ? .budgetExhausted:.captured)
        try requireGradient(event.status == status,"Gradient coverage status differs.")
        var rows=0
        for (index,b) in event.batches.enumerated() {
            let expectedIndex=scheduled == 1 ? 0:index*(event.totalBatches-1)/(scheduled-1)
            try requireGradient(b.batchIndex == expectedIndex && (1...event.totalTrainingRows).contains(b.trainingRows),"Invalid gradient sampled minibatch.")
            let pc=[b.presentParameterCount,b.missingParameterCount],ec=[b.presentElementCount,b.missingElementCount,b.finiteElementCount,b.nonfiniteElementCount,b.zeroElementCount]
            try requireGradient(pc.allSatisfy{(0...inventory.parameterCount).contains($0)} && ec.allSatisfy{(0...inventory.elementCount).contains($0)},"Gradient count exceeds inventory.")
            try requireGradient(b.presentParameterCount+b.missingParameterCount == inventory.parameterCount && b.presentElementCount+b.missingElementCount == inventory.elementCount && b.finiteElementCount+b.nonfiniteElementCount == b.presentElementCount && b.zeroElementCount<=b.finiteElementCount && b.presentElementCount>=b.presentParameterCount && b.missingElementCount>=b.missingParameterCount && (b.presentParameterCount == 0) == (b.presentElementCount == 0) && (b.missingParameterCount == 0) == (b.missingElementCount == 0),"Gradient presence or finite counts differ.")
            if b.presentElementCount == 0 || b.nonfiniteElementCount>0 {try requireGradient(b.l2Norm == nil && b.maxAbs == nil,"Unobserved or nonfinite gradients require null norms.")}
            else {
                guard let l2=b.l2Norm,let maxAbs=b.maxAbs else {throw GradientCaptureError(message:"Finite observed gradients require norms.")}
                try requireGradient(l2.isFinite && maxAbs.isFinite && maxAbs>=0 && l2>=maxAbs && l2<=maxAbs*sqrt(Double(b.presentElementCount))*(1+1e-12) && (maxAbs == 0) == (b.zeroElementCount == b.finiteElementCount),"Invalid gradient norm.")
            }
            rows+=b.trainingRows
        }
        try requireGradient(rows == event.observedTrainingRows && rows<=event.totalTrainingRows,"Gradient observed row count differs.")
    }
    public static func decodeConfiguration(_ data:Data)throws->GradientCaptureConfiguration {try requireGradient(data.count<=1024,"Gradient config exceeds bounds.");return try JSONDecoder().decode(GradientCaptureConfiguration.self,from:data)}
    public static func decodeStarted(_ data:Data)throws->GradientCaptureStarted {
        let v=try object(json(data,limit:200_000),keys:["schemaVersion","config","initialEpoch","inventory"]);try inventoryShape(v["inventory"]!)
        let result=try JSONDecoder().decode(GradientCaptureStarted.self,from:data);try result.config.validate();try validate(result.inventory)
        try requireGradient(result.schemaVersion == 1 && (0..<10000).contains(result.initialEpoch),"Invalid gradient start epoch or schema.");return result
    }
    public static func decodeEvent(_ data:Data,runID:String,config:GradientCaptureConfiguration,inventory:GradientCaptureInventory,previous:GradientEpochSummary?=nil,initialEpoch:Int=0)throws->GradientEpochSummary {
        try eventShape(json(data,limit:100_000));let value=try JSONDecoder().decode(GradientEpochSummary.self,from:data)
        try validate(value,runID:runID,config:config,inventory:inventory,previous:previous,initialEpoch:initialEpoch);return value
    }
    public static func decodeMetadata(_ data:Data)throws->GradientCaptureMetadata {
        _ = try object(json(data,limit:4096),keys:["schemaVersion","file","sha256","bytes","observedEpochCount","capturedEpochCount","observedBatchCount","initialEpoch","finalEpoch"])
        let v=try JSONDecoder().decode(GradientCaptureMetadata.self,from:data)
        try requireGradient(v.schemaVersion == 1 && v.file == "gradient-summary.json" && v.sha256.count == 64 && v.sha256.allSatisfy{"0123456789abcdef".contains($0)} && (1...maximumArtifactBytes).contains(v.bytes) && (0..<10000).contains(v.initialEpoch) && (v.initialEpoch+1...10000).contains(v.finalEpoch) && v.observedEpochCount == v.finalEpoch-v.initialEpoch && (0...v.observedEpochCount).contains(v.capturedEpochCount) && v.observedBatchCount>=v.capturedEpochCount && v.observedBatchCount<=1024,"Invalid gradient artifact metadata.");return v
    }
    public static func decodeArtifact(_ data:Data,expectedRunID:String,expectedDevice:String,metadata:GradientCaptureMetadata?=nil)throws->GradientCaptureArtifact {
        let v=try object(json(data),keys:["schemaVersion","id","runID","device","phase","config","initialEpoch","finalEpoch","inventory","summaries"])
        try inventoryShape(v["inventory"]!);for event in try array(v["summaries"]) {try eventShape(event)}
        let result=try JSONDecoder().decode(GradientCaptureArtifact.self,from:data)
        try result.config.validate();try validate(result.inventory)
        try requireGradient(result.schemaVersion == 1 && result.id == "kwip.gradient-summary.v1" && result.phase == "afterBackwardBeforeStep" && result.runID == expectedRunID && UUID(uuidString:expectedRunID) != nil && result.device == expectedDevice && ["cpu","mps"].contains(result.device) && (0..<10000).contains(result.initialEpoch) && result.finalEpoch>result.initialEpoch && result.finalEpoch<=10000 && result.summaries.count == result.finalEpoch-result.initialEpoch,"Gradient artifact identity or epoch coverage differs.")
        var previous:GradientEpochSummary?
        for event in result.summaries {try validate(event,runID:expectedRunID,config:result.config,inventory:result.inventory,previous:previous,initialEpoch:result.initialEpoch);previous=event}
        if let metadata {
            _ = try decodeMetadata(JSONEncoder().encode(metadata))
            let hash=SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()
            try requireGradient(metadata.sha256 == hash && metadata.bytes == data.count && metadata.initialEpoch == result.initialEpoch && metadata.finalEpoch == result.finalEpoch && metadata.observedEpochCount == result.summaries.count && metadata.capturedEpochCount == result.summaries.filter{$0.status == .captured}.count && metadata.observedBatchCount == result.summaries.reduce(0){$0+$1.observedBatchCount},"Gradient metadata hash or coverage differs.")
        }
        return result
    }
}
