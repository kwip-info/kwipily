import Foundation

enum DiagnosticEvaluator:String,Codable {case nonfinite,generalizationGap,lossGrowth,deviceError,shapeError,dataError,workerError}
struct DiagnosticRule:Codable,Equatable {
    var id:String
    var version:Int
    var evaluator:DiagnosticEvaluator
    var title:String
    var suggestion:String
    var prerequisites:[String]
    var thresholds:[String:Double]
    var terms:[String]
    var confounders:[String]
    var missingEvidence:String
}
struct DiagnosticRuleRegistry:Codable,Equatable {
    var schemaVersion=1
    var id="kwip.run-diagnostics"
    var version=1
    var rules:[DiagnosticRule]
    static var builtIn:Self {Self(rules:[
.init(id:"measurement.nonfinite.v1",version:1,evaluator:.nonfinite,title:"Non-finite measurement",suggestion:"Inspect source values and numerical ranges, then review the failing operation and precision settings. Trend analysis is withheld because these measurements are not finite.",prerequisites:["At least one metric observation"],thresholds:[:],terms:[],confounders:["Transport or source corruption; no failing operation is inferred."],missingEvidence:"No observations: not evaluated."),
.init(id:"trend.generalization-gap.v1",version:1,evaluator:.generalizationGap,title:"Possible overfitting pattern",suggestion:"Compare earlier checkpoints and repeat with a verified holdout split. Consider less training or a smaller model if the pattern persists. This divergence is a heuristic, not proof of overfitting; data shift or evaluation differences can also cause it.",prerequisites:["All provided measurements finite", "Latest 20 positive distinct epochs; last duplicate wins", "At least 10 observations; nonnegative losses and accuracy in [0,1]"],thresholds:["minimumObservations": 10, "windowObservations": 20, "meanObservations": 5, "minimumAbsoluteChange": 0.02, "relativeChange": 0.1, "minimumGapIncrease": 0.05],terms:[],confounders:["Data shift", "Different evaluation procedures", "Sampling, resumed runs, and changed partitions are not verified"],missingEvidence:"Withhold when prerequisites fail; absence is not evidence of model health."),
.init(id:"trend.loss-growth.v1",version:1,evaluator:.lossGrowth,title:"Sharp observed loss growth",suggestion:"Inspect feature ranges, the learning rate and recent architecture changes; compare a controlled shorter run. Gradient norms were not measured, so this does not establish exploding gradients.",prerequisites:["All provided measurements finite", "Latest 20 positive distinct epochs; last duplicate wins", "At least 10 observations; nonnegative losses and accuracy in [0,1]"],thresholds:["minimumObservations": 10, "windowObservations": 20, "meanObservations": 5, "baselineFloor": 0.01, "growthFactor": 2, "minimumAbsoluteGrowth": 0.5],terms:[],confounders:["Loss-scale or evaluation changes", "Gradient norms are not measured", "Sampling and continuation boundaries are not verified"],missingEvidence:"Withhold when prerequisites fail; no gradient diagnosis."),
.init(id:"error.device.v1",version:1,evaluator:.deviceError,title:"Compute backend needs attention",suggestion:"Check Project settings for available devices and the exact operation named in the error. Select a supported backend explicitly and retry; keep this error with the run record. No fallback is applied automatically.",prerequisites:["Nonblank raw runtime error"],thresholds:[:],terms:["mps", "cuda", "backend", "device unavailable", "unsupported device", "requested device"],confounders:["Substring matches may be incidental", "No hardware or operation probe is performed"],missingEvidence:"No error: not evaluated; lexical precedence device before shape before data before generic."),
.init(id:"error.shape.v1",version:1,evaluator:.shapeError,title:"Graph contract needs attention",suggestion:"Select the named component and compare each incoming tensor width with its declared input. Repair incompatible dimensions or connections, then use graph validation before retrying.",prerequisites:["Nonblank raw runtime error", "No earlier error classifier matches"],thresholds:[:],terms:["shape", "size mismatch", "receives", "expects", "width", "dimension", "input port", "connection", "cycle"],confounders:["Substring matches may be incidental", "No tensor contract is independently checked"],missingEvidence:"No error: not evaluated; earlier classifier match: withheld."),
.init(id:"error.data.v1",version:1,evaluator:.dataError,title:"Data source needs attention",suggestion:"Open Data, refresh the source preview and check the selected feature/target columns and cited rows. Fix the reported missing, invalid or duplicate values before retrying; do not silently drop records.",prerequisites:["Nonblank raw runtime error", "No earlier error classifier matches"],thresholds:[:],terms:["csv", "column", "source", "mapping", "dataset", "target must", "observations"],confounders:["Substring matches may be incidental", "Source records are not reread or diagnosed"],missingEvidence:"No error: not evaluated; earlier classifier match: withheld."),
.init(id:"error.worker.v1",version:1,evaluator:.workerError,title:"Worker reported an error",suggestion:"Review the exact message and run artifacts before retrying. This initial rule set cannot identify a specific cause from this error.",prerequisites:["Nonblank raw runtime error", "No specific classifier matches"],thresholds:[:],terms:[],confounders:["Cause remains unknown"],missingEvidence:"No error: not evaluated; earlier classifier match: withheld.")
    ])}
    /// Only this reviewed implementation version is executable; metadata is not a plugin language.
    func validate() throws {
        guard self == Self.builtIn else {throw DiagnosticContractError("Unsupported or modified diagnostic registry; use the pinned reviewed rule set.")}
    }
    static func decode(_ data:Data)throws->Self {
        let object=try DiagnosticSchema.object(data)
        try DiagnosticSchema.keys(object,["schemaVersion","id","version","rules"])
        for item in object["rules"] as? [[String:Any]] ?? [] {try DiagnosticSchema.keys(item,["id","version","evaluator","title","suggestion","prerequisites","thresholds","terms","confounders","missingEvidence"])}
        let value=try JSONDecoder().decode(Self.self,from:data);try value.validate();return value
    }
}
