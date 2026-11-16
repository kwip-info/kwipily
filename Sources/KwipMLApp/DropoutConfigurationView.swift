import SwiftUI
import KwipMLCore

/// Typed text stays unpublished; the slider changes the same probability without quantizing it.
struct DropoutInspectorDraft: Equatable {
    var numeric:String="0.5"
    init(probability:Double=0.5) {numeric=String(probability)}
    var probability:Double? {
        guard let value=Double(numeric.trimmingCharacters(in:.whitespacesAndNewlines)),value.isFinite,(0...1).contains(value) else {return nil}
        return value
    }
    mutating func setProbability(_ value:Double) {
        guard value.isFinite else {return}
        numeric=String(format:"%.16g",min(1,max(0,value)))
    }
    var percentageLabel:String? {
        probability.map {String(format:"%.6g%% drop chance",$0*100)}
    }
}

struct DropoutConfigurationView:View {
    let node:GraphNode
    var bound:Bool=false
    var isBusy:Bool
    var onPending:(Bool)->Void={_ in}
    let onApply:(Double,[String:JSONValue])->Bool
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=DropoutInspectorDraft()
    @State private var baseline=DropoutInspectorDraft()
    @State private var expected:[String:JSONValue]=[:]
    @State private var rejection:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.normalization.primary:ComponentVisualFamily.normalization.secondary}
    private var changed:Bool {draft != baseline}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            Text("Tensor · in → out").foregroundStyle(accent).frame(maxWidth:.infinity,alignment:.leading)
                .padding(8).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.25)))
            VStack(alignment:.leading,spacing:4) {
                Text("Drop probability")
                TextField("Drop probability",text:Binding(get:{draft.numeric},set:{draft.numeric=$0;rejection=nil}))
                    .flatField().accessibilityLabel("Drop probability").accessibilityIdentifier("dropout.probability")
                HStack {
                    Text("0")
                    Slider(value:Binding(get:{draft.probability ?? baseline.probability ?? 0.5},set:{draft.setProbability($0);rejection=nil}),in:0...1)
                        .accessibilityLabel("Drop probability slider").accessibilityIdentifier("dropout.slider")
                    Text("1")
                }
                if let label=draft.percentageLabel {Text(label).foregroundStyle(accent).monospacedDigit()}
            }.disabled(isBusy || bound)
            if bound {Image(systemName:"link").foregroundStyle(accent).accessibilityLabel("Controlled by block settings")}
            HStack {
                Button("Apply") {apply()}.buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10))
                    .disabled(isBusy || bound || !changed || draft.probability==nil).accessibilityIdentifier("dropout.apply")
                Button("Revert") {reload()}.buttonStyle(FlatButtonStyle()).disabled(isBusy || !changed).accessibilityIdentifier("dropout.revert")
                Spacer(minLength:0)
                if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Dropout changes")}
            }
            if let rejection {Text(rejection).foregroundStyle(.red).fixedSize(horizontal:false,vertical:true)}
            else if draft.probability==nil {Text("Use a finite value from 0 to 1.").foregroundStyle(.red)}
        }.font(.caption).tint(accent)
            .onAppear {reload()}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node.config) {_,next in
                guard next != expected else {return}
                if !changed {reload()} else {rejection="Saved settings changed. Revert to load them; your draft is preserved."}
            }.onChange(of:changed) {_,value in onPending(value)}.onDisappear {onPending(false)}
    }
    private func reload() {
        expected=node.config
        draft = .init(probability:node.config["p"]?.doubleValue ?? 0.5)
        baseline=draft;rejection=nil;onPending(false)
    }
    private func apply() {
        guard !isBusy,!bound,changed,let value=draft.probability else {return}
        guard onApply(value,expected) else {rejection="Dropout settings were not applied. Revert to load the current source.";return}
        expected["p"] = .number(value);baseline=draft;rejection=nil;onPending(false)
    }
}
