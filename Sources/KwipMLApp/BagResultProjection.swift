import Foundation
import KwipMLCore

/// Presentation of verified whole-bag observations, never a dense input substitute.
enum BagResultProjection {
    private static func formatted(_ values:[Double])->String {
        let shown=values.prefix(16).map {String(format:"%.6g",$0)}.joined(separator:", ")
        return values.count>16 ? "\(shown) … (first 16 of \(values.count))":shown
    }
    private static func ids(_ values:[Int64])->String {
        let shown=values.prefix(16).map(String.init).joined(separator:", ")
        return values.count>16 ? "\(shown) … (first 16 of \(values.count))":shown
    }
    private static func geometry(_ tensor:ObjectiveResultTensor)->String {
        (tensor.shape.isEmpty ? "scalar []":"shape [\(tensor.shape.map(String.init).joined(separator:", "))]")+" · "+tensor.dtype.rawValue
    }
    static func sections(preview:BagPredictionPreview,evaluation:ObjectiveEvaluation?=nil)->[ObjectiveResultSection] {
        var sections:[ObjectiveResultSection]=[]
        if let e=evaluation {
            var measures=[["Result score",formatted([e.effectiveScalar])],["Observations",String(e.observations)],["Active positions",String(e.activeCount)],["Ignored positions",String(e.ignoredCount)],["Masked positions",String(e.maskedCount)],["Score total",formatted([e.aggregationNumerator])],["Divisor",formatted([e.aggregationDenominator])]]
            if let accuracy=e.accuracy {measures.append(["Accuracy",String(format:"%.2f%%",accuracy*100)])}
            sections.append(.init(id:"summary",title:"Evaluation summary",caption:"Recorded comparison across all evaluated observations. The score total and divisor explain how the overall result was combined.",columns:["Measure","Value"],rows:measures))
        }
        func rawCheckpoint() {
            guard let raw=evaluation?.rawPreview else {return}
            let shape=raw.shape.isEmpty ? "scalar []":"shape [\(raw.shape.map(String.init).joined(separator:", "))]"
            sections.append(.init(id:"rawPreview",title:"Score checkpoint",caption:shape+" · \(raw.dtype.rawValue). A bounded value prefix; the full tensor may be larger.",columns:["Capture","Values"],rows:[[raw.captureStatus == .captured ? "First \(raw.values.count) values":"Omitted by policy",formatted(raw.values)]]))
        }
        guard preview.captureStatus == .captured,!preview.rows.isEmpty else {
            sections.append(.init(id:"capture",title:"Whole-bag observations",caption:"Observation rows were excluded by this run’s capture settings. Recorded aggregate evaluation remains available independently.",columns:["Capture"],rows:[[preview.captureStatus == .omittedByPolicy ? "Omitted by policy":"No captured observations"]]))
            rawCheckpoint();return sections
        }
        let rows=Array(preview.rows.prefix(32))
        let count="First \(rows.count) of \(preview.rows.count) captured observations; \(preview.evaluationCount) evaluated."
        let weighted=rows.contains {$0.input.perSampleWeights != nil}
        let columns=["Observation","Items","Item IDs"]+(weighted ? ["Floating weights"]:[])
        sections.append(.init(id:"inputs",title:"Whole-bag inputs",caption:count+" Each observation is an entire bag. Item IDs select lookup-table rows; optional weights scale their contribution. Empty bags contain no items.",columns:columns,rows:rows.enumerated().map {index,row in
            var cells=[String(index+1),String(row.input.indices.count),row.input.indices.isEmpty ? "Empty bag":ids(row.input.indices)]
            if weighted {cells.append(row.input.perSampleWeights.map {$0.isEmpty ? "Empty bag":formatted($0.map(Double.init))} ?? "Not recorded")}
            return cells
        }))
        if rows.contains(where:{$0.label != nil && $0.predicted != nil}) {
            sections.append(.init(id:"binary",title:"Binary class predictions",caption:count+" Expected and predicted values are recorded class IDs; names are defined by your source data.",columns:["Observation","Expected class","Predicted class","Match"],rows:rows.enumerated().map {index,row in
                [String(index+1),row.label.map(String.init) ?? "Not recorded",row.predicted.map(String.init) ?? "Not recorded",row.label.flatMap {label in row.predicted.map {label == $0 ? "Yes":"No"}} ?? "Not recorded"]
            }))
        }
        if let sample=rows.compactMap(\.output).first {
            sections.append(.init(id:"produced",title:"Predicted values",caption:count+" "+geometry(sample)+" per observation.",columns:["Observation","Values"],rows:rows.enumerated().map {[String($0.offset+1),$0.element.output.map {formatted($0.values)} ?? "Not recorded"]}))
        }
        let roles=Set(rows.flatMap {$0.targets.keys}).sorted()
        let titles=["target":"Desired values","input_lengths":"Source sequence lengths","target_lengths":"Desired sequence lengths","weight":"Element importance","pos_weight":"Positive-label importance"]
        for role in roles {
            guard let sample=rows.compactMap({$0.targets[role]}).first else {continue}
            sections.append(.init(id:"target."+role,title:titles[role] ?? role,caption:geometry(sample)+" per observation. Mapped source role: \(role).",columns:["Observation","Values"],rows:rows.enumerated().map {[String($0.offset+1),$0.element.targets[role].map {formatted($0.values)} ?? "Not recorded"]}))
        }
        if let sample=rows.compactMap(\.decoded).first {
            sections.append(.init(id:"classSelection",title:"Selected classes",caption:geometry(sample)+" per observation. These recorded class indices do not name real-world labels.",columns:["Observation","Selected class indices"],rows:rows.enumerated().map {[String($0.offset+1),$0.element.decoded.map {formatted($0.values)} ?? "Not recorded"]}))
        }
        if let sample=rows.compactMap(\.objectiveOutput).first {
            sections.append(.init(id:"individualScores",title:"Individual comparison scores",caption:geometry(sample)+" per observation. Recorded before the configured combination step.",columns:["Observation","Score values"],rows:rows.enumerated().map {[String($0.offset+1),$0.element.objectiveOutput.map {formatted($0.values)} ?? "Not recorded"]}))
        }
        rawCheckpoint();return sections
    }
}
