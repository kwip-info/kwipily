import Foundation
import KwipMLCore

/// A parent schedule owns its trace, independently of the child's selected schedule.
enum WorkflowCheckpointInventory {
    static let traceName="rate-trace.json"
    static let traceLimit=4_000_000
    static func structuralLimits(names:Set<String>)throws->[String:Int] {
        var limits=WorkflowCheckpointSnapshot.artifactLimits
        if names.contains(traceName) {limits[traceName]=traceLimit}
        guard names == Set(limits.keys) else {throw WorkflowArchiveError("Checkpoint parent inventory is incomplete or contains unknown files.")}
        return limits
    }
    static func limits(manifest:Data)throws->[String:Int] {
        let raw=try JSONDecoder().decode([String:JSONValue].self,from:manifest)
        var limits=WorkflowCheckpointSnapshot.artifactLimits
        if try WorkflowLifecycleParentRecord.hasScheduler(manifest) {
            guard case .object(let trace)=raw["rateTrace"],Set(trace.keys)==["schemaVersion","path","bytes","sha256","records","timing"],trace["schemaVersion"] == 1,trace["path"] == .string(traceName),let bytes=trace["bytes"]?.integerValue,(1...traceLimit).contains(bytes),let hash=trace["sha256"]?.stringValue,WorkflowArchiveCodec.isHash(hash),let records=trace["records"]?.integerValue,(0...10000).contains(records),case .object(let settings)=raw["settings"],let recipe=settings["schedulerRecipe"] else {throw WorkflowArchiveError("Scheduled parent requires an exact bounded rate trace pin.")}
            let configuration=try SchedulerConfiguration.decodeRecipe(recipe)
            let timing=configuration.componentID == "scheduler_one_cycle_lr" ? "afterOptimizerBatch":"afterCompletedEpochValidation"
            guard trace["timing"] == .string(timing) else {throw WorkflowArchiveError("Parent trace timing differs from its scheduler.")}
            limits[traceName]=traceLimit
        } else if raw["rateTrace"] != nil {throw WorkflowArchiveError("A parent without a scheduler cannot contain a rate trace.")}
        return limits
    }
    static func verify(_ contents:[String:Data])throws {
        guard let manifest=contents["manifest.json"] else {throw WorkflowArchiveError("Parent manifest is missing.")}
        let limits=try limits(manifest:manifest)
        guard Set(contents.keys)==Set(limits.keys),contents.allSatisfy({!$0.value.isEmpty && $0.value.count<=limits[$0.key]!}) else {throw WorkflowArchiveError("Checkpoint parent requires its exact bounded dependent artifacts.")}
        if let bytes=contents[traceName] {
            let raw=try JSONDecoder().decode([String:JSONValue].self,from:manifest)
            guard case .object(let pin)=raw["rateTrace"],pin["bytes"]?.integerValue == bytes.count,pin["sha256"] == .string(WorkflowArchiveCodec.hash(bytes)),case .object(let settings)=raw["settings"] else {throw WorkflowArchiveError("Parent rate trace differs from its manifest pin.")}
            let trace=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
            guard Set(trace.keys)==["schemaVersion","recipe","timing","initialization","records"],trace["schemaVersion"] == 1,trace["recipe"] == settings["schedulerRecipe"],trace["timing"] == pin["timing"],case .array(let records)=trace["records"],records.count==pin["records"]?.integerValue else {throw WorkflowArchiveError("Parent rate trace identity or count differs from its manifest.")}
        }
    }
}
