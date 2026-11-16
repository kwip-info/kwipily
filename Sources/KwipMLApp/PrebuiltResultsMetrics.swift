import SwiftUI
import KwipMLCore
struct PrebuiltResultsMetrics:View {
    let manifest:[String:JSONValue]
    private var metrics:[String:JSONValue] {if case .object(let m)=manifest["evaluation"] {return m};return [:]}
    private var numericMetrics:[String:JSONValue] {BackendResultProjection.numericMetrics(manifest)}
    private let fields:[(String,String)]=[("count","Rows"),("accuracy","Accuracy"),("logLoss","Log loss"),("macroPrecision","Macro precision"),("macroRecall","Macro recall"),("macroF1","Macro F1"),("mae","MAE"),("rmse","RMSE"),("rSquared","R²"),("clusters","Clusters"),("noiseRows","Noise rows"),("inertia","Fit inertia"),("silhouette","Silhouette")]
    var body:some View {
        ResultsSection(title:"Metrics",symbol:"chart.bar",initiallyExpanded:true) {
            LazyVGrid(columns:[GridItem(.adaptive(minimum:140),alignment:.leading)],alignment:.leading,spacing:12) {
                ForEach(fields.filter{numericMetrics[$0.0] != nil},id:\.0) {key,title in VStack(alignment:.leading,spacing:4){Text(title).font(.caption).foregroundStyle(MLTheme.muted);Text(numericMetrics[key]?.doubleValue.map{String(format:"%.6g",$0)} ?? "Unavailable").font(.system(size:14,design:.monospaced))}}
            }
            if case .array(let classes)=metrics["classOrder"] {Text("Class order · "+classes.map{$0.stringValue ?? $0.integerValue.map(String.init) ?? "?"}.joined(separator:", ")).font(.caption).foregroundStyle(MLTheme.muted)}
            if let sample=metrics["silhouetteSampleRows"]?.integerValue,let total=metrics["silhouettePopulationRows"]?.integerValue {Text("Silhouette · \(sample) sampled / \(total) population rows · seed 42 · noise excluded. Fit population was not sampled.").font(.caption).foregroundStyle(MLTheme.muted)}
            Text("Open the component Output or its connected View for complete tables. Cluster IDs and memberships are not churn probabilities. Model packages expect the saved prepared feature order.").font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
}
