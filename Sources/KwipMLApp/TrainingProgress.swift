import Foundation
import KwipMLCore

/// Observation cadence is a local display preference, not a scientific run setting.
/// Its resolved value is pinned by the immutable worker launch and progress receipt.
struct TrainingReportingPolicy:Codable,Equatable {
    var everyBatches:Int=10
    var everySeconds:Double=1
    static func decode(_ bytes:Data)throws->Self {
        guard bytes.count<=256,let fields=try JSONSerialization.jsonObject(with:bytes) as? [String:Any],Set(fields.keys)==["everyBatches","everySeconds"] else {throw WorkflowPhaseError("Invalid training reporting fields.")}
        let result=try JSONDecoder().decode(Self.self,from:bytes)
        guard (1...10000).contains(result.everyBatches),result.everySeconds.isFinite,(0.1...60).contains(result.everySeconds) else {throw WorkflowPhaseError("Invalid training reporting cadence.")}
        return result
    }
    static var current:Self {
        guard let bytes=UserDefaults.standard.data(forKey:"trainingReportingPolicy"),let value=try? decode(bytes) else {return .init()};return value
    }
    func save() {if let bytes=try? JSONEncoder().encode(self), (try? Self.decode(bytes)) != nil {UserDefaults.standard.set(bytes,forKey:"trainingReportingPolicy")}}
}

/// Decode live canonical epochs with the same floating-point reader used by
/// saved metrics and progress. JSONSerialization's NSNumber route can round
/// the identical JSON number differently on macOS.
struct TrainingEpochObservation:Decodable {
    let epoch:Int
    let trainLoss:Double
    let validationLoss:Double
    let accuracy:Double
    static func decode(_ bytes:Data)throws->Self {
        let value=try JSONDecoder().decode(Self.self,from:bytes)
        guard value.epoch>0,value.trainLoss.isFinite,value.validationLoss.isFinite,value.accuracy.isFinite,(0...1).contains(value.accuracy) else {throw WorkflowPhaseError("Invalid epoch measurement.")}
        return value
    }
}

struct MoETrainingObjective:Codable,Equatable {
    let balancePenalty:Double
    let routerZPenalty:Double
    let trainingObjective:Double
    func validate(task:Double?)throws {
        guard let task,balancePenalty.isFinite,routerZPenalty.isFinite,trainingObjective.isFinite,balancePenalty>=0,routerZPenalty>=0,
              abs(trainingObjective-task-balancePenalty-routerZPenalty)<=1e-10*max(1,abs(trainingObjective)) else {throw WorkflowPhaseError("MoE objective differs from the captured task loss and weighted penalties.")}
    }
}

struct TrainingProgressPoint:Codable,Equatable,Identifiable {
    let event:String
    let schemaVersion:Int
    let runID:String
    let sequence:Int
    let epoch:Int
    let epochBatch:Int
    let epochBatches:Int
    let batches:Int
    let updates:Int
    let elapsedSeconds:Double
    let kind:String
    let trainingLoss:Double?
    var moe:MoETrainingObjective?=nil
    let numerator:Double?
    let denominator:Double?
    let validationLoss:Double?
    let accuracy:Double?
    private enum CodingKeys:String,CodingKey {case moe,event,schemaVersion,runID,sequence,epoch,epochBatch,epochBatches,batches,updates,elapsedSeconds,kind,trainingLoss,numerator,denominator,validationLoss,accuracy}
    func encode(to encoder:Encoder)throws {
        var c=encoder.container(keyedBy:CodingKeys.self)
        try c.encode(event,forKey:.event);try c.encode(schemaVersion,forKey:.schemaVersion);try c.encode(runID,forKey:.runID)
        try c.encode(sequence,forKey:.sequence);try c.encode(epoch,forKey:.epoch);try c.encode(epochBatch,forKey:.epochBatch);try c.encode(epochBatches,forKey:.epochBatches)
        try c.encode(batches,forKey:.batches);try c.encode(updates,forKey:.updates);try c.encode(elapsedSeconds,forKey:.elapsedSeconds);try c.encode(kind,forKey:.kind)
        try c.encode(trainingLoss,forKey:.trainingLoss)
        try c.encodeIfPresent(moe,forKey:.moe)
        if kind=="training" {try c.encode(numerator,forKey:.numerator);try c.encode(denominator,forKey:.denominator)}
        else {try c.encode(validationLoss,forKey:.validationLoss);try c.encode(accuracy,forKey:.accuracy)}
    }
    var id:Int {sequence}
    static func decode(_ bytes:Data)throws->Self {
        let fields=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        let base:Set<String>=["event","schemaVersion","runID","sequence","epoch","epochBatch","epochBatches","batches","updates","elapsedSeconds","kind","trainingLoss"]
        let training=fields["kind"] == .string("training")
        guard Set(fields.keys)==base.union(training ? ["numerator","denominator"]:["validationLoss","accuracy"]).union(fields["schemaVersion"]==2 ? ["moe"]:[]) else {throw WorkflowPhaseError("Unexpected training progress fields.")}
        let row=try JSONDecoder().decode(Self.self,from:bytes)
        guard row.event=="trainingProgress",[1,2].contains(row.schemaVersion),UUID(uuidString:row.runID) != nil,
              (1...1_000_000_000).contains(row.sequence),(1...10000).contains(row.epoch),
              row.epochBatches>0,row.epochBatches<=1_000_000_000,(1...row.epochBatches).contains(row.epochBatch),
              row.batches>0,row.batches<=1_000_000_000,row.updates>=0,row.updates<=row.batches,
              row.elapsedSeconds.isFinite,row.elapsedSeconds>=0,
              row.trainingLoss.map(\.isFinite) ?? true,
              row.validationLoss.map(\.isFinite) ?? true,
              row.accuracy.map({$0.isFinite && (0...1).contains($0)}) ?? true else {throw WorkflowPhaseError("Invalid training progress values.")}
        if row.schemaVersion==2 {guard training,let moe=row.moe,case .object(let values)=fields["moe"],Set(values.keys)==["balancePenalty","routerZPenalty","trainingObjective"] else {throw WorkflowPhaseError("MoE progress requires exact training objective fields.")};try moe.validate(task:row.trainingLoss)}
        if training {
            guard let numerator=row.numerator,let denominator=row.denominator,numerator.isFinite,denominator.isFinite,denominator>=0,
                  (denominator==0 && row.trainingLoss==nil) || (denominator>0 && row.trainingLoss==numerator/denominator) else {throw WorkflowPhaseError("Progress loss differs from its weighted interval.")}
        } else {
            guard row.kind=="validation",row.epochBatch==row.epochBatches,row.trainingLoss != nil,row.validationLoss != nil else {throw WorkflowPhaseError("Invalid validation observation.")}
        }
        return row
    }
}

/// Bounded retained observations. Decimation is explicit; canonical epoch metrics
/// and checkpoint/replay evidence are independent and never downsampled here.
struct TrainingProgressLedger {
    static let maximumPoints=8192
    private(set) var points:[TrainingProgressPoint]=[]
    private(set) var observed=0
    private(set) var last:TrainingProgressPoint?
    mutating func append(_ row:TrainingProgressPoint,worker:String,maximumEpoch:Int,initialEpoch:Int=0)throws {
        guard row.runID==worker,row.epoch<=maximumEpoch,row.epoch>initialEpoch,
              row.sequence==observed+1 else {throw WorkflowPhaseError("Training progress belongs to another run or sequence.")}
        if let last {
            guard row.elapsedSeconds>=last.elapsedSeconds,row.batches>=last.batches,row.updates>=last.updates,row.updates-last.updates<=row.batches-last.batches,
                  row.epoch==last.epoch || (last.kind=="validation" && row.epoch==last.epoch+1),
                  row.kind=="validation" ? (last.kind=="training" && row.batches==last.batches && row.updates==last.updates && row.epoch==last.epoch && last.epochBatch==last.epochBatches) : (row.batches>last.batches && row.epochBatches==(row.epoch==last.epoch ? last.epochBatches:row.epochBatches)) else {throw WorkflowPhaseError("Training progress counters moved backwards or skipped validation.")}
        } else {guard row.kind=="training",row.epoch==initialEpoch+1 else {throw WorkflowPhaseError("Progress started outside the accepted epoch.")}}
        last=row;observed+=1;points.append(row)
        if points.count>Self.maximumPoints {points=points.enumerated().filter {$0.offset==0 || $0.offset%2==0 || $0.offset==points.count-1}.map(\.element)}
    }
}

struct TrainingProgressArchive:Codable {
    static let path="diagnostics/training-progress.json"
    static let maximumBytes=8_000_000
    let schemaVersion:Int
    let requestIdentity:String
    let workerRunID:String
    let policy:TrainingReportingPolicy
    let observed:Int
    let points:[TrainingProgressPoint]
    static func decode(_ bytes:Data)throws->Self {
        guard bytes.count<=maximumBytes,Set(try JSONDecoder().decode([String:JSONValue].self,from:bytes).keys)==["schemaVersion","requestIdentity","workerRunID","policy","observed","points"] else {throw WorkflowPhaseError("Invalid progress receipt fields or size.")}
        return try JSONDecoder().decode(Self.self,from:bytes)
    }
    func validate(request:String,worker:String,metrics:[EpochMetric])throws {
        guard schemaVersion==1,requestIdentity==request,workerRunID==worker,observed>=points.count,points.count<=TrainingProgressLedger.maximumPoints,observed<=1_000_000_000 else {throw WorkflowPhaseError("Progress receipt identity or bounds differ.")}
        _=try TrainingReportingPolicy.decode(JSONEncoder().encode(policy))
        let epochs=Dictionary(uniqueKeysWithValues:metrics.map {($0.epoch,$0)})
        var previous:TrainingProgressPoint?
        for point in points {
            let row=try TrainingProgressPoint.decode(JSONEncoder().encode(point))
            guard row.runID==worker,row.sequence<=observed,previous.map({$0.sequence<row.sequence && $0.batches<=row.batches && $0.updates<=row.updates && $0.elapsedSeconds<=row.elapsedSeconds && $0.epoch<=row.epoch}) ?? true else {throw WorkflowPhaseError("Invalid retained progress order.")}
            if row.kind=="validation" {
                guard let canonical=epochs[row.epoch],canonical.trainLoss==row.trainingLoss,canonical.validationLoss==row.validationLoss,canonical.accuracy==row.accuracy else {throw WorkflowPhaseError("Progress validation differs from verified epoch metrics.")}
            }
            previous=row
        }
        guard points.last?.sequence==observed,points.last?.kind=="validation",points.last?.epoch==metrics.last?.epoch else {throw WorkflowPhaseError("Progress receipt has no verified terminal measurement.")}
    }
}
