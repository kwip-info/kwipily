import Foundation
import KwipMLCore

/// Geometry resolved from the captured graph and its actual native parameter inventory.
struct DerivativeTargetGeometry {
    let target:JSONValue
    let name:String
    let shape:[Int]
    let dtype:ContractDType
    let requiresGrad:Bool
}
struct DerivativeRequestGeometry {
    let requestID:String
    let outputShape:[Int]
    let targets:[DerivativeTargetGeometry]
}
/// The caller must obtain these bytes and row identities through owned, verified
/// run/preparation artifacts. Artifact-authored hashes are not an authority.
struct DerivativeResultsContext {
    let policy:GradientPolicyConfiguration
    let capture:DerivativeCapturePolicy
    let runID:String
    let device:ContractDevice
    let torchVersion:String
    let graph:Data
    let checkpoint:Data
    let preparationManifest:Data
    let preparationGraphDigest:String
    let partitionRowIDs:[String:[String]]
    let geometry:[DerivativeRequestGeometry]
    var bag:WorkflowBagDerivativeContext? = nil
}
enum DerivativeReportedStatus:String {case observed,unavailable}
struct DerivativeTargetResult {
    let target:JSONValue
    let name:String
    let shape:[Int]
    let dtype:ContractDType
    let status:DerivativeReportedStatus
    let values:[Double]?
    let reportedCause:String?
}
struct DerivativeRequestResult {
    let request:JSONValue
    let outputShape:[Int]
    let targets:[DerivativeTargetResult]
}
struct DerivativeResults {
    let runID:String
    let device:ContractDevice
    let requests:[DerivativeRequestResult]
    /// Validates provenance/structure and reported statuses; does not independently
    /// prove the worker's numerical derivative or a reported disconnected route.
    private static func failure(_ text:String)->WorkflowPhaseError {WorkflowPhaseError("Derivative results: "+text)}
    private static func object(_ value:JSONValue?,keys:Set<String>)throws->[String:JSONValue] {
        guard case .object(let v)=value,Set(v.keys)==keys else {throw failure("missing or unknown fields.")};return v
    }
    private static func integer(_ value:JSONValue?,minimum:Int=0,maximum:Int=Int.max)throws->Int {
        guard case .integer(let n)=value,n>=Int64(minimum),n<=Int64(maximum) else {throw failure("invalid integer count.")};return Int(n)
    }
    private static func array(_ value:JSONValue?)throws->[JSONValue] {
        guard case .array(let v)=value else {throw failure("expected a bounded array.")};return v
    }
    private static func shape(_ value:JSONValue?,allowEmpty:Bool=false)throws->[Int] {
        let v=try array(value);guard v.count<=8 else {throw failure("shape rank exceeds eight directions.")}
        let result=try v.map {try integer($0,minimum:allowEmpty ? 0:1,maximum:100_000_000)}
        guard !result.contains(0) || allowEmpty && result == [0] else {throw failure("zero shape is not a verified packed coordinate.")};return result
    }
    private static func elements(_ shape:[Int],maximum:Int,allowEmpty:Bool=false)throws->Int {
        if allowEmpty && shape == [0] {return 0}
        var n=1
        for d in shape {guard d>0,d<=maximum,n<=maximum/d else {throw failure("shape exceeds its element budget.")};n *= d};return n
    }
    private static func hash(_ value:String)->Bool {value.count==64 && value.utf8.allSatisfy {($0>=48 && $0<=57)||($0>=97 && $0<=102)}}
    private static func normalizedRequest(_ value:JSONValue,policy:GradientPolicyConfiguration)throws->JSONValue {
        var config=policy.values;config["requests"] = .array([value])
        return try GradientPolicyConfiguration(values:config).requests[0]
    }
    static func verify(_ data:Data,metadata:JSONValue,context c:DerivativeResultsContext)throws->Self {
        try c.capture.validate(gradientPolicy:c.policy)
        guard c.capture.values,!c.policy.requests.isEmpty,!data.isEmpty,data.count<=8_388_608,
              !c.runID.isEmpty,!c.torchVersion.isEmpty,[ContractDevice.cpu,.mps].contains(c.device),
              !c.graph.isEmpty,!c.checkpoint.isEmpty,!c.preparationManifest.isEmpty,hash(c.preparationGraphDigest),
              c.geometry.count==c.policy.requests.count else {throw failure("selected permission, verified source or bounded request context is absent.")}
        let schema = c.bag == nil ? 1:2
        if let bag=c.bag {
            guard bag.preparation.selection.manifestSHA256 == WorkflowPhaseArtifacts.hash(c.preparationManifest),
                  bag.preparation.selection.graphDigest == c.preparationGraphDigest,
                  c.partitionRowIDs == ["training":bag.preparation.training.rowIDs,"validation":bag.preparation.validation.rowIDs] else {throw failure("bag context differs from pinned preparation and row identities.")}
        }
        let m=try object(metadata,keys:["schemaVersion","path","bytes","sha256","requests","elements","phase"])
        guard try integer(m["schemaVersion"],minimum:schema,maximum:schema)==schema,
              m["path"]?.stringValue=="derivative-results.json",m["phase"]?.stringValue=="finalSnapshotEvaluation",
              try integer(m["bytes"],minimum:1,maximum:8_388_608)==data.count,
              m["sha256"]?.stringValue==WorkflowPhaseArtifacts.hash(data),
              try integer(m["requests"],minimum:1,maximum:32)==c.policy.requests.count else {throw failure("artifact inventory differs from the selected immutable bytes.")}
        let declaredElements=try integer(m["elements"],maximum:65_536)
        let root=try object(JSONDecoder().decode(JSONValue.self,from:data),keys:Set(["schemaVersion","runID","phase","device","dtype","torchVersion","recipe","source","requests"]).union(c.bag == nil ? []:["modelInput"]))
        guard try integer(root["schemaVersion"],minimum:schema,maximum:schema)==schema,root["runID"]?.stringValue==c.runID,
              root["phase"]?.stringValue=="finalSnapshotEvaluation",root["device"]?.stringValue==c.device.rawValue,
              root["dtype"]?.stringValue=="float32",root["torchVersion"]?.stringValue==c.torchVersion,
              let recipe=root["recipe"],try GradientPolicyConfiguration.decodeRecipe(recipe)==c.policy else {throw failure("run, runtime or selected recipe differs.")}
        if let bag=c.bag {guard root["modelInput"] == bag.modelInput else {throw failure("bag source contract differs from the captured model.")}}
        let source=try object(root["source"],keys:["graphSHA256","checkpointSHA256","preparationManifestSHA256","preparationGraphDigest"])
        let pins=["graphSHA256":WorkflowPhaseArtifacts.hash(c.graph),"checkpointSHA256":WorkflowPhaseArtifacts.hash(c.checkpoint),"preparationManifestSHA256":WorkflowPhaseArtifacts.hash(c.preparationManifest),"preparationGraphDigest":c.preparationGraphDigest]
        guard pins.allSatisfy({source[$0.key]?.stringValue==$0.value}) else {throw failure("saved graph, learned state or preparation source changed.")}
        let records=try array(root["requests"]);guard records.count==c.policy.requests.count else {throw failure("request count differs.")}
        var results:[DerivativeRequestResult]=[],returnedElements=0,appearanceElements=0
        for (index,value) in records.enumerated() {
            let record=try object(value,keys:["request","sample","output","targets"])
            guard let submitted=record["request"],try normalizedRequest(submitted,policy:c.policy)==c.policy.requests[index],
                  case .object(let request)=c.policy.requests[index],let id=request["id"]?.stringValue,
                  c.geometry[index].requestID==id,let partition=request["partition"]?.stringValue,
                  let rowIDs=c.partitionRowIDs[partition] else {throw failure("request order or captured partition identity differs.")}
            let start=try integer(request["row_start"]),count=try integer(request["row_count"],minimum:1,maximum:32)
            guard start<=rowIDs.count,count<=rowIDs.count-start else {throw failure("requested rows exceed the actual captured partition.")}
            let sample=try object(record["sample"],keys:Set(["partition","rowStart","rowCount","rowIDsSHA256"]).union(c.bag == nil ? []:["bagItemCounts"]))
            let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
            let rowHash=WorkflowPhaseArtifacts.hash(try encoder.encode(Array(rowIDs[start..<start+count])))
            guard sample["partition"]?.stringValue==partition,try integer(sample["rowStart"])==start,
                  try integer(sample["rowCount"],minimum:1,maximum:32)==count,sample["rowIDsSHA256"]?.stringValue==rowHash else {throw failure("selected row identity or order changed.")}
            let bagCounts=try c.bag?.selectedCounts(c.policy.requests[index])
            if let bagCounts {guard sample["bagItemCounts"] == .array(bagCounts.map { .integer(Int64($0)) }) else {throw failure("bag item counts differ from the independently verified selected observations.")}}
            let output=try object(record["output"],keys:["node","port","shape","dtype"])
            guard case .object(let endpoint)=request["output"],output["node"]==endpoint["node"],output["port"]==endpoint["port"],output["dtype"]?.stringValue=="float32" else {throw failure("output endpoint or dtype differs.")}
            let outputShape=try shape(output["shape"])
            guard outputShape==c.geometry[index].outputShape,outputShape.first==count,(1...8).contains(outputShape.count),outputShape.dropFirst().allSatisfy({$0<=4096}) else {throw failure("produced output geometry differs from the captured graph.")}
            _ = try elements(outputShape,maximum:100_000_000)
            if case .object(let upstream)=request["upstream"] {guard try shape(upstream["shape"])==outputShape else {throw failure("upstream shape differs from actual selected output.")}}
            let targets=try array(record["targets"]),selectedTargets=try array(request["targets"])
            guard targets.count==selectedTargets.count,targets.count==c.geometry[index].targets.count else {throw failure("target count differs from native ownership.")}
            var parsed:[DerivativeTargetResult]=[],names=Set<String>()
            for (targetIndex,value) in targets.enumerated() {
                guard case .object(let target)=value,let statusName=target["status"]?.stringValue,let status=DerivativeReportedStatus(rawValue:statusName) else {throw failure("unknown reported derivative status.")}
                let base:Set<String>=["target","name","shape","dtype","status"]
                guard Set(target.keys)==base.union(status == .observed ? ["values"]:["cause"]) else {throw failure("observed values and unavailable reasons must remain distinct.")}
                let expected=c.geometry[index].targets[targetIndex]
                guard case .object(let identity)=expected.target,let kind=identity["kind"]?.stringValue else {throw failure("missing captured target kind.")}
                var coordinateShape:[Int]?
                if kind == "input",let bag=c.bag,let bagCounts,let node=identity["node"]?.stringValue,let port=identity["port"]?.stringValue {
                    coordinateShape=try bag.coordinateShape(node:node,port:port,counts:bagCounts)
                }
                let empty=coordinateShape == [0] && expected.shape == [0]
                let targetShape=try shape(target["shape"],allowEmpty:empty),n=try elements(targetShape,maximum:65_536,allowEmpty:empty)
                guard expected.target==selectedTargets[targetIndex],target["target"]==expected.target,target["name"]?.stringValue==expected.name,
                      !expected.name.isEmpty,names.insert(expected.name).inserted,targetShape==expected.shape,
                      target["dtype"]?.stringValue==expected.dtype.rawValue else {throw failure("target identity, order, aliases or native geometry differs.")}
                guard appearanceElements<=65_536-n else {throw failure("aggregate requested derivatives exceed 65536 elements.")};appearanceElements += n
                if kind=="input" {
                    guard [ContractDType.float32,.int64].contains(expected.dtype) else {throw failure("input target dtype differs.")}
                    if let coordinateShape {
                        guard targetShape==coordinateShape,let port=identity["port"]?.stringValue,
                              expected.dtype == (port == "per_sample_weights" ? .float32:.int64) else {throw failure("bag coordinate target geometry or dtype differs.")}
                    } else {guard c.bag == nil,targetShape.first==count else {throw failure("input target geometry differs.")}}
                }
                else {guard kind=="parameter",expected.dtype == .float32 else {throw failure("native parameter target must be float32.")}}
                let intrinsicallyUnavailable=expected.dtype == .int64 || kind=="parameter" && !expected.requiresGrad
                var values:[Double]?,cause:String?
                if status == .observed {
                    guard !intrinsicallyUnavailable else {throw failure("integer or frozen target cannot claim observed derivatives.")}
                    let raw=try array(target["values"]);guard raw.count==n else {throw failure("derivative values differ from native target shape.")}
                    values=try raw.map {v in guard let d=v.doubleValue,d.isFinite,abs(d)<=Double(Float.greatestFiniteMagnitude),(d==0 || Float(d) != 0) else {throw failure("derivative values must be finite float32 numbers, never Booleans.")};return d}
                    returnedElements += n
                } else {
                    guard request["unavailable"]?.stringValue=="reportUnavailable",let reported=target["cause"]?.stringValue else {throw failure("selected policy requires an actual derivative.")}
                    if expected.dtype == .int64 {guard reported=="integerInput",kind=="input" else {throw failure("unavailable cause contradicts integer input.")}}
                    else if kind=="parameter" && !expected.requiresGrad {guard reported=="frozenParameter" else {throw failure("unavailable cause contradicts native trainability.")}}
                    else {guard ["disconnected","sparseGradient"].contains(reported) else {throw failure("unsupported cause for a recorded final-snapshot derivative.")}}
                    cause=reported
                }
                parsed.append(.init(target:expected.target,name:expected.name,shape:targetShape,dtype:expected.dtype,status:status,values:values,reportedCause:cause))
            }
            results.append(.init(request:c.policy.requests[index],outputShape:outputShape,targets:parsed))
        }
        guard returnedElements==declaredElements else {throw failure("manifest derivative element count differs from observed values.")}
        return .init(runID:c.runID,device:c.device,requests:results)
    }
}
