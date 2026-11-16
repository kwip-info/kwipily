import Foundation
import KwipMLCore

/// Presentation-only tables. Truncation is explicit; structural geometry stays exact.
struct ObjectiveResultSection:Equatable,Identifiable {
    let id:String
    let title:String
    let caption:String
    let columns:[String]
    let rows:[[String]]
}
enum ObjectiveResultProjection {
    private static func formatted(_ values:[Double])->String {
        let prefix=values.prefix(16).map {String(format:"%.6g",$0)}.joined(separator:", ")
        return values.count>16 ? "\(prefix) … (first 16 of \(values.count))":prefix
    }
    private static func geometry(_ shape:[Int])->String {shape.isEmpty ? "scalar []":"shape [\(shape.map(String.init).joined(separator:", "))]"}
    static func sections(preview:ObjectivePredictionPreview,evaluation e:ObjectiveEvaluation)->[ObjectiveResultSection] {
        var summary=[["Result score",formatted([e.effectiveScalar])],["Observations",String(e.observations)],["Active positions",String(e.activeCount)],["Ignored positions",String(e.ignoredCount)],["Masked positions",String(e.maskedCount)],["Score total",formatted([e.aggregationNumerator])],["Divisor",formatted([e.aggregationDenominator])]]
        if e.sequenceObjective {
            summary=[["Validation loss / token",formatted([e.effectiveScalar])],["Perplexity",e.perplexity.map {formatted([$0])} ?? "Outside numeric range"],["Windows",String(e.observations)],["Valid tokens",String(e.activeCount)],["Padded tokens",String(e.ignoredCount)]]
        }
        if let accuracy=e.accuracy {summary.append([e.sequenceObjective ? "Token accuracy":"Accuracy",String(format:"%.2f%%",accuracy*100)])}
        var sections:[ObjectiveResultSection]=[.init(id:"summary",title:"Evaluation summary",caption:"The combined score uses the recorded score total and divisor. Lower scores usually mean closer agreement; each objective defines the comparison.",columns:["Measure","Value"],rows:summary)]
        func appendRawCheckpoint() {
        if let raw=e.rawPreview {
            sections.append(.init(id:"rawPreview",title:"Score checkpoint",caption:geometry(raw.shape)+" · \(raw.dtype.rawValue). This is a bounded prefix, not the complete tensor.",columns:["Capture","Values"],rows:[[raw.captureStatus == .captured ? "First \(raw.values.count) values":"Omitted by policy",formatted(raw.values)]]))
        }
        }
        if preview.rows.isEmpty {
            sections.append(.init(id:"capture",title:"Predicted values",caption:"Prediction rows were excluded by this run’s capture settings.",columns:["Capture"],rows:[["Omitted by policy"]]));appendRawCheckpoint();return sections
        }
        let displayed=Array(preview.rows.prefix(32)),shape=displayed[0].output.shape
        let countCaption="First \(displayed.count) of \(preview.evaluationCount) evaluation observations; \(geometry(shape))."
        if e.componentID == "loss_triplet_margin",shape.count == 2,shape[0] == 3 {
            let width=shape[1]
            sections.append(.init(id:"produced",title:"Three related examples",caption:countCaption+" Anchor is the example to compare; Match should be closer than Nonmatch.",columns:["Observation","Anchor","Match","Nonmatch"],rows:displayed.enumerated().map {index,row in [String(index+1)]+(0..<3).map {group in formatted(Array(row.output.values[group*width..<(group+1)*width]))}}))
        } else {
            sections.append(.init(id:"produced",title:"Predicted values",caption:countCaption,columns:["Observation","Values"],rows:displayed.enumerated().map {[String($0.offset+1),formatted($0.element.output.values)]}))
        }
        for role in displayed[0].targets.keys.sorted() {
            guard let tensor=displayed[0].targets[role] else {continue}
            let titles=["target":"Desired values","input_lengths":"Source sequence lengths","target_lengths":"Desired sequence lengths","weight":"Element importance","pos_weight":"Positive-label importance"]
            sections.append(.init(id:"target.\(role)",title:titles[role] ?? role,caption:geometry(tensor.shape)+" · \(tensor.dtype.rawValue). These values came from the selected source columns.",columns:["Observation","Desired values"],rows:displayed.enumerated().map {[String($0.offset+1),formatted($0.element.targets[role]!.values)]}))
        }
        if let tensor=displayed[0].decoded {
            sections.append(.init(id:"classSelection",title:"Selected classes",caption:geometry(tensor.shape)+" per observation. These recorded class indices do not name real-world labels.",columns:["Observation","Selected class indices"],rows:displayed.enumerated().map {[String($0.offset+1),formatted($0.element.decoded!.values)]}))
        }
        if let tensor=displayed[0].objectiveOutput {
            sections.append(.init(id:"individualScores",title:"Individual comparison scores",caption:geometry(tensor.shape)+" per observation. These scores were retained before the configured combination step.",columns:["Observation","Score values"],rows:displayed.enumerated().map {[String($0.offset+1),formatted($0.element.objectiveOutput!.values)]}))
        }
        appendRawCheckpoint()
        return sections
    }
}
