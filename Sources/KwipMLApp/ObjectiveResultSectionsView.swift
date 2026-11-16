import SwiftUI

/// One presentation for current-session and independently verified archived objectives.
/// All numeric values arrive through the typed artifact decoder; this view never grades them.
struct ObjectiveResultSectionsView:View {
    let preview:ObjectivePredictionPreview
    let evaluation:ObjectiveEvaluation
    let provenance:String
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            Text(provenance).font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            ForEach(ObjectiveResultProjection.sections(preview:preview,evaluation:evaluation)) { section in
                ResultsSection(title:section.title,symbol:section.id == "rawPreview" ? "square.stack.3d.up":"tablecells",initiallyExpanded:section.id == "summary" || section.id == "produced" || section.id == "capture") {
                    VStack(alignment:.leading,spacing:8) {
                        Text(section.caption).font(.caption2).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                        if section.id == "rawPreview",let raw=evaluation.rawPreview {
                            TensorOutputPreviewView(preview:["shape":raw.shape,"dtype":raw.dtype.rawValue,"values":raw.values,"captureStatus":raw.captureStatus.rawValue])
                        } else {table(section)}
                    }.padding(.vertical,8)
                }
            }
        }.frame(maxWidth:.infinity,alignment:.leading)
            .accessibilityElement(children:.contain)
            .accessibilityLabel("Objective result tables")
    }
    private func table(_ section:ObjectiveResultSection)->some View {
        ThemedScrollView([.horizontal,.vertical]) {
            VStack(alignment:.leading,spacing:0) {
                HStack(spacing:0) {
                    ForEach(Array(section.columns.enumerated()),id:\.offset) {index,title in cell(title,index:index)}
                }.font(.system(size:11,weight:.semibold)).background(MLTheme.panel)
                ForEach(Array(section.rows.enumerated()),id:\.offset) {_,row in
                    Rectangle().fill(MLTheme.rule).frame(height:1)
                    HStack(alignment:.top,spacing:0) {
                        ForEach(Array(row.enumerated()),id:\.offset) {index,value in cell(value,index:index)}
                    }.font(.system(size:11,design:.monospaced)).textSelection(.enabled)
                }
            }.overlay(Rectangle().stroke(MLTheme.rule,lineWidth:1))
        }.frame(height:min(260,CGFloat(section.rows.count+1)*34+2))
    }
    private func cell(_ value:String,index:Int)->some View {
        Text(value).lineLimit(2).help(value).padding(.horizontal,10).padding(.vertical,8)
            .frame(width:index == 0 ? 150:260,alignment:.leading)
    }
}
