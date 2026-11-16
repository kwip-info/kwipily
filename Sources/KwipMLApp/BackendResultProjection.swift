import Foundation
import KwipMLCore

/// Accepted backend evidence, distinct from optional preparation/node samples.
/// Never serializes a worker manifest, status prose, artifact paths or labels.
struct BackendResultProjection {
    static let metricNames:Set<String> = ["count","samples","accuracy","logLoss","loss","macroPrecision","macroRecall","macroF1","mae","rmse","rSquared","clusters","noiseRows","inertia","silhouette","silhouetteSampleRows","silhouettePopulationRows"]

    static func numericMetrics(_ manifest:[String:JSONValue]) -> [String:JSONValue] {
        guard case .object(let metrics)=manifest["evaluation"] else {return [:]}
        return metrics.filter { key,value in
            guard metricNames.contains(key) else {return false}
            switch value {case .integer:return true;case .number(let n):return n.isFinite;default:return false}
        }
    }

    let runID:UUID
    let capturedRevision:UInt64
    let phase:WorkflowExecutionPhase
    let operation:String
    let backendID:String
    let currentConstruction:Bool
    let currentBindings:Bool
    let accepted:Bool
    let tablePorts:[String]
    let manifest:[String:JSONValue]

    func response(includeValues:Bool)->[String:JSONValue] {
        let completed=phase == .completed && accepted
        var result:[String:JSONValue] = ["available":.bool(completed),"resultKind":.string("backend"),
            "runID":.string(runID.uuidString),"capturedRevision":.integer(Int64(capturedRevision)),
            "phase":.string(phase.rawValue),"operation":.string(["fit","predict","evaluate"].contains(operation) ? operation:"unknown"),
            "backendID":.string(["sklearn","tensorflow"].contains(backendID) ? backendID:"unknown"),
            "currentConstruction":.bool(currentConstruction),"currentBindings":.bool(currentBindings),
            "ports":.array(completed ? tablePorts.filter{["predictions","profiles","confusion"].contains($0)}.sorted().map(JSONValue.string):[])]
        result["scope"] = .string("Current-session backend evidence; freshness flags apply. Tables use explicit value-authorized paging. No model publication is implied.")
        if completed && includeValues {result["evaluation"] = .object(Self.numericMetrics(manifest))}
        return result
    }
}

extension WorkflowWorkspaceSession {
    func backendResultSummary(for address:WorkflowNodeAddress,includeValues:Bool)->[String:JSONValue]? {
        guard let backend=currentBackendFit,backend.plan.estimator==address else {return nil}
        let construction=WorkflowOutputConstruction.matches(owner.document,captured:backend.document,provider:address)
        let bindingsMatch=bindings.revision==backend.bindingRevision && sourceProvider.observedSourcesMatch(backend.preparationOutputs)
        let ports=["predictions","profiles","confusion"].filter {backend.predictionArtifact(for:.init(address:address,port:$0)) != nil}
        return BackendResultProjection(runID:backend.runID,capturedRevision:backend.document.revision,phase:backend.phase,
            operation:backend.plan.configuration.operation,backendID:backend.plan.configuration.backendID,
            currentConstruction:construction,currentBindings:bindingsMatch,accepted:backend.manifest != nil && backend.output != nil,
            tablePorts:ports,manifest:backend.manifest ?? [:]).response(includeValues:includeValues)
    }
}
