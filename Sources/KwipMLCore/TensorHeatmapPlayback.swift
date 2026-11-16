import Foundation

/// Authenticated, bounded summaries from actual validation passes; never tensor history.
public struct TensorHeatmapPlayback:Equatable {
    public struct Frame:Equatable {
        public let epoch:Int
        public let sourceRows:Int
        public let summary:TensorHeatmapSummary
    }
    public let frames:[Frame]
    public let reason:String?
    public init(value:JSONValue,maximumFrames:Int=24)throws {
        func invalid() -> GraphConfigurationError {GraphConfigurationError("Invalid or unrequested heatmap playback evidence.")}
        guard case .object(let raw)=value,Set(raw.keys).isSubset(of:["schemaVersion","source","frames","reason"]),raw["schemaVersion"] == .integer(1),raw["source"] == .string("validationFirstBatch"),case .array(let entries)?=raw["frames"],entries.count<=min(24,maximumFrames),maximumFrames>0 else {throw invalid()}
        var result:[Frame]=[]
        for entry in entries {
            guard case .object(let fields)=entry,Set(fields.keys)==["epoch","sourceRows","summary"],let epoch=fields["epoch"]?.integerValue,epoch>0,epoch<=1_000_000,epoch>Int64(result.last?.epoch ?? 0),let rows=fields["sourceRows"]?.integerValue,(1...1024).contains(rows),let summary=fields["summary"] else {throw invalid()}
            let measured=try TensorHeatmapSummary(value:summary)
            guard measured.available else {throw invalid()}
            result.append(.init(epoch:Int(epoch),sourceRows:Int(rows),summary:measured))
        }
        if let reason=raw["reason"] {guard case .string(let text)=reason,!text.isEmpty,text.count<=200,entries.isEmpty else {throw invalid()};self.reason=text}
        else {reason=nil}
        frames=result
    }
}
