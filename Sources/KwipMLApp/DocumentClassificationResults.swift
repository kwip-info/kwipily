import SwiftUI
import KwipMLCore

/// Pure rendering of accepted aggregate evidence; no fitting/IO on canvas render.
struct DocumentClassificationResults:View {
    let manifest:[String:JSONValue]
    private var metrics:[String:JSONValue] {if case .object(let c)=manifest["evaluation"] {return c};return [:]}
    private var labels:[String] {if case .array(let c)=metrics["classOrder"] ?? manifest["classOrder"] {return c.compactMap(\.stringValue)};return []}
    private var matrix:[[JSONValue]] {if case .array(let rows)=metrics["confusionMatrix"] {return rows.compactMap {if case .array(let cells)=$0 {return cells};return nil}};return []}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            ResultsSection(title:manifest["operation"] == .string("predict") ? "Unseen document predictions":"Held-out classification",symbol:"doc.text.magnifyingglass",initiallyExpanded:true) {
                Text(manifest["operation"] == .string("evaluate") ? "Explicit labeled evaluation · population supplied on this run":manifest["operation"] == .string("predict") ? "Unlabeled documents · frozen vocabulary and coefficients · no accuracy is inferred":"Training fits the classifier; Validation evaluates. Test is not evaluated during fitting.").font(.caption).foregroundStyle(MLTheme.muted)
                LazyVGrid(columns:[GridItem(.adaptive(minimum:130),alignment:.leading)],alignment:.leading,spacing:10) {
                    metric("Documents",key:"count",percent:false)
                    if metrics["accuracy"] != nil {
                        metric("Accuracy",key:"accuracy");metric("Macro F1",key:"macroF1",percent:false);metric("Balanced accuracy",key:"balancedAccuracy");metric("Majority baseline",key:"majorityBaseline")
                    } else {metric("Classified",key:"classified",percent:false);metric("Extraction failures",key:"failed",percent:false)}
                }
                if let converged=manifest["solver"],case .object(let s)=converged,s["converged"] == false {Text("Solver reached its iteration limit. Review convergence before relying on this model.").foregroundStyle(MLTheme.rose)}
            }
            if !matrix.isEmpty {
                ResultsSection(title:"Confusion matrix",symbol:"square.grid.3x3",initiallyExpanded:true) {
                    Text("Actual ↓ · Predicted → · counts across all evaluated documents").font(.caption).foregroundStyle(MLTheme.muted)
                    ThemedScrollView(.horizontal) {
                        Grid(alignment:.leading,horizontalSpacing:12,verticalSpacing:8) {
                            GridRow {Text("Category");ForEach(Array(labels.prefix(12).enumerated()),id:\.offset) {_,label in Text(label).lineLimit(1)}}.fontWeight(.semibold)
                            ForEach(Array(matrix.prefix(12).enumerated()),id:\.offset) {i,row in
                                GridRow {Text(labels.indices.contains(i) ? labels[i]:"—");ForEach(Array(row.prefix(12).enumerated()),id:\.offset) {j,cell in
                                    Text(String(cell.integerValue ?? 0)).monospacedDigit().frame(minWidth:44,minHeight:26).background(ComponentVisualFamily.data.primary.opacity(i==j ? 0.15:0.03))
                                }}
                            }
                        }.font(.caption).textSelection(.enabled).padding(.vertical,6)
                    }
                    if labels.count>12 {Text("Showing the first 12 of \(labels.count) categories; aggregate metrics cover all categories.").font(.caption2).foregroundStyle(MLTheme.muted)}
                }
                ResultsSection(title:"Per-class metrics",symbol:"list.bullet") {
                    if case .array(let rows)=metrics["perClass"] {
                        ThemedScrollView(.horizontal) {Grid(alignment:.leading,horizontalSpacing:18,verticalSpacing:8) {
                            GridRow {Text("Category");Text("Precision");Text("Recall");Text("F1");Text("Support")}.fontWeight(.semibold)
                            ForEach(Array(rows.prefix(128).enumerated()),id:\.offset) {_,row in if case .object(let c)=row {GridRow {Text(c["category"]?.stringValue ?? "—");Text(format(c["precision"]));Text(format(c["recall"]));Text(format(c["f1"]));Text(String(c["support"]?.integerValue ?? 0))}}}
                        }.font(.caption).textSelection(.enabled)}
                    }
                    Text("Zero-division metrics are 0. Missing categories are reported; probabilities are not calibrated certainty.").font(.caption2).foregroundStyle(MLTheme.muted)
                }
            }

        }
    }
    private func metric(_ title:String,key:String,percent:Bool=true)->some View {VStack(alignment:.leading,spacing:3) {Text(title).font(.caption2).foregroundStyle(MLTheme.muted);Text(metrics[key]?.doubleValue.map {percent ? String(format:"%.1f%%",$0*100):String(format:"%.4g",$0)} ?? "—").font(.system(size:15,design:.monospaced))}}
    private func format(_ value:JSONValue?)->String {value?.doubleValue.map{String(format:"%.3f",$0)} ?? "—"}
}
