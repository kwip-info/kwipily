import Foundation

struct DiagnosticContractError:LocalizedError {let message:String;init(_ message:String){self.message=message};var errorDescription:String?{message}}
enum DiagnosticSampling:String,Codable {case unspecified,complete,sampled}
struct DiagnosticInputContext:Codable,Equatable {
    var runID:String?
    var sampling:DiagnosticSampling
    init(runID:String?=nil,sampling:DiagnosticSampling = .unspecified){self.runID=runID;self.sampling=sampling}
    func validate()throws {if let runID {guard !runID.isEmpty,runID.count<=128 else {throw DiagnosticContractError("Invalid diagnostic run reference.")}}}
}
struct DiagnosticObservationReference:Codable,Equatable {var inputIndex:Int;var epoch:Int}
struct DiagnosticProvenance:Codable,Equatable {
    var registryID:String
    var registryVersion:Int
    var ruleVersion:Int
    var inputSHA256:String
    var context:DiagnosticInputContext
    var observedInputCount:Int
    var observations:[DiagnosticObservationReference]
    var usesRawError:Bool
    var derivedSignals:[String:Double]
}
struct RunDiagnostic:Identifiable,Codable,Equatable {
    let id:String
    let title:String
    let evidence:String
    let suggestion:String
    var schemaVersion=1
    var provenance:DiagnosticProvenance?
    init(id:String,title:String,evidence:String,suggestion:String,provenance:DiagnosticProvenance?=nil) {
        self.id=id;self.title=title;self.evidence=evidence;self.suggestion=suggestion;self.provenance=provenance
    }
}
enum DiagnosticEvaluationStatus:String,Codable {case emitted,notTriggered,withheld}
enum DiagnosticEvaluationReason:String,Codable {case matched,thresholdNotMet,noObservations,missingError,earlierErrorClassifier,insufficientObservations,nonfiniteMeasurements,invalidMetricDomain,nonfiniteDerivedSignals}
struct DiagnosticEvaluation:Codable,Equatable {var ruleID:String;var status:DiagnosticEvaluationStatus;var reason:DiagnosticEvaluationReason}
struct DiagnosticReport:Codable,Equatable {
    var schemaVersion=1
    var registryID:String
    var registryVersion:Int
    var inputSHA256:String
    var context:DiagnosticInputContext
    var observedInputCount:Int
    var findings:[RunDiagnostic]
    var evaluations:[DiagnosticEvaluation]
    var limitations:[String]
    func validate(registry:DiagnosticRuleRegistry = .builtIn)throws {
        try registry.validate();try context.validate()
        let ids=registry.rules.map(\.id)
        guard schemaVersion==1,registryID==registry.id,registryVersion==registry.version,
              inputSHA256.count==64,inputSHA256.utf8.allSatisfy({(48...57).contains($0)||(97...102).contains($0)}),observedInputCount>=0,
              evaluations.map(\.ruleID)==ids,findings.count<=ids.count,Set(findings.map(\.id)).count==findings.count,
              Set(evaluations.filter {$0.status == .emitted}.map(\.ruleID))==Set(findings.map(\.id)),
              limitations==RunDiagnostics.limitations else {throw DiagnosticContractError("Invalid or unsupported diagnostic report provenance.")}
        for (evaluation,rule) in zip(evaluations,registry.rules) {
            let allowed:Set<DiagnosticEvaluationReason>
            switch rule.evaluator {
            case .nonfinite: allowed=[.noObservations]
            case .generalizationGap,.lossGrowth: allowed=[.insufficientObservations,.nonfiniteMeasurements,.invalidMetricDomain,.nonfiniteDerivedSignals]
            case .deviceError: allowed=[.missingError]
            case .shapeError,.dataError,.workerError: allowed=[.missingError,.earlierErrorClassifier]
            }
            guard (evaluation.status == .emitted && evaluation.reason == .matched) ||
                  (evaluation.status == .notTriggered && evaluation.reason == .thresholdNotMet && rule.evaluator != .workerError) ||
                  (evaluation.status == .withheld && allowed.contains(evaluation.reason)) else {
                throw DiagnosticContractError("Diagnostic evaluation status contradicts its rule or reason.")
            }
        }
        for finding in findings {
            guard let rule=registry.rules.first(where:{$0.id==finding.id}),let p=finding.provenance,
                  finding.schemaVersion==1,finding.title==rule.title,finding.suggestion==rule.suggestion,
                  p.registryID==registryID,p.registryVersion==registryVersion,p.ruleVersion==rule.version,
                  p.inputSHA256==inputSHA256,p.context==context,p.observedInputCount==observedInputCount,
                  p.observations.count<=20,Set(p.observations.map(\.inputIndex)).count==p.observations.count,
                  p.observations.allSatisfy({$0.inputIndex>=0 && $0.inputIndex<observedInputCount}),
                  p.derivedSignals.values.allSatisfy(\.isFinite) else {throw DiagnosticContractError("Invalid diagnostic finding schema or evidence references.")}
            switch rule.evaluator {
            case .deviceError,.shapeError,.dataError,.workerError:
                guard p.usesRawError,p.observations.isEmpty,p.derivedSignals.isEmpty,!finding.evidence.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty else {throw DiagnosticContractError("Invalid raw-error evidence declaration.")}
            case .nonfinite:
                guard !p.usesRawError,!p.observations.isEmpty,p.observations.count<=8,
                      Set(p.derivedSignals.keys)==["affectedObservationCount"],let count=p.derivedSignals["affectedObservationCount"],
                      count.rounded()==count,count>=Double(p.observations.count),count<=Double(observedInputCount) else {throw DiagnosticContractError("Invalid non-finite observation evidence.")}
            case .generalizationGap,.lossGrowth:
                let keys:Set<String>=["trainingStartMean","trainingEndMean","validationStartMean","validationEndMean","gapIncrease"]
                guard !p.usesRawError,(10...20).contains(p.observations.count),Set(p.derivedSignals.keys)==keys,
                      p.observations.allSatisfy({$0.epoch>0}),zip(p.observations,p.observations.dropFirst()).allSatisfy({$0.epoch<$1.epoch}),
                      p.derivedSignals.filter({$0.key != "gapIncrease"}).values.allSatisfy({$0>=0}) else {throw DiagnosticContractError("Invalid trend evidence or undeclared derived signal.")}
                let values=p.derivedSignals
                guard values["gapIncrease"]! == (values["validationEndMean"]!-values["trainingEndMean"]!)-(values["validationStartMean"]!-values["trainingStartMean"]!) else {throw DiagnosticContractError("Inconsistent derived gap signal.")}
            }
        }
    }
    static func decode(_ data:Data)throws->Self {
        let object=try DiagnosticSchema.object(data)
        try DiagnosticSchema.keys(object,["schemaVersion","registryID","registryVersion","inputSHA256","context","observedInputCount","findings","evaluations","limitations"])
        try DiagnosticSchema.context(object["context"])
        for finding in object["findings"] as? [[String:Any]] ?? [] {
            try DiagnosticSchema.keys(finding,["id","title","evidence","suggestion","schemaVersion","provenance"])
            if let provenance=finding["provenance"] as? [String:Any] {
                try DiagnosticSchema.keys(provenance,["registryID","registryVersion","ruleVersion","inputSHA256","context","observedInputCount","observations","usesRawError","derivedSignals"])
                try DiagnosticSchema.context(provenance["context"])
                for reference in provenance["observations"] as? [[String:Any]] ?? [] {try DiagnosticSchema.keys(reference,["inputIndex","epoch"])}
            }
        }
        for evaluation in object["evaluations"] as? [[String:Any]] ?? [] {try DiagnosticSchema.keys(evaluation,["ruleID","status","reason"])}
        let value=try JSONDecoder().decode(Self.self,from:data);try value.validate();return value
    }
}
enum DiagnosticSchema {
    static func object(_ data:Data)throws->[String:Any] {
        guard data.count<=8_000_000,let value=try JSONSerialization.jsonObject(with:data) as? [String:Any] else {throw DiagnosticContractError("Diagnostic document must be an object within 8 MB.")};return value
    }
    static func keys(_ object:[String:Any],_ allowed:Set<String>)throws {guard Set(object.keys).isSubset(of:allowed) else {throw DiagnosticContractError("Unknown diagnostic schema field.")}}
    static func context(_ object:Any?)throws {if let value=object as? [String:Any] {try keys(value,["runID","sampling"])}}
}
