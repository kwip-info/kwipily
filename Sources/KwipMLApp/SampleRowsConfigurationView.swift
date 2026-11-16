import SwiftUI
import KwipMLCore

/// Retains the shared numeric edit/validation lifecycle and saved sampling contract.
struct SampleRowsConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    @Environment(\.colorScheme) private var colorScheme
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            VStack(alignment:.leading,spacing:7) {
                Label("Sample size",systemImage:"tablecells").fontWeight(.semibold).foregroundStyle(accent)
                Text("Rows to retain")
                DataFlowNumberField(context:context,key:"count",title:"Rows to retain",domain:.integer(minimum:1,maximum:100000),fallback:.integer(20))
            }.padding(10).background(accent.opacity(0.06)).overlay(Rectangle().stroke(accent.opacity(0.3)))
            VStack(alignment:.leading,spacing:7) {
                Label("Reproducibility",systemImage:"arrow.triangle.2.circlepath").fontWeight(.semibold).foregroundStyle(accent)
                Text("Sampling seed")
                DataFlowNumberField(context:context,key:"seed",title:"Sampling seed",domain:.integer(minimum:0,maximum:4294967295),fallback:.integer(42))
            }.padding(10).background(accent.opacity(0.06)).overlay(Rectangle().stroke(accent.opacity(0.3)))
        }.font(.caption).tint(accent)
    }
}
