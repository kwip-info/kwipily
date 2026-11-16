import SwiftUI
import KwipMLCore

/// Shared typed rule controls. The caller supplies its existing framed inspector section.
struct InitializationRulesEditorView:View {
    @ObservedObject var context:InitializationRecipeInspectorContext
    var inlinePresentation = false
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Seed").font(.caption.weight(.semibold))
            TextField("Whole-number seed",text:Binding(get:{context.seedText},set:{_=context.setSeedText($0)}))
                .flatField().accessibilityIdentifier("initialization.seed")
            if !inlinePresentation { Text("The same seed and parameter identity reproduce the same starting numbers on the verified runtime.")
                .font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(Array(context.rules.enumerated()),id:\.offset) {index,row in rule(index,row)}
            Button { _=context.addOwner() } label:{Label("Add weights or biases",systemImage:"plus")}
                .buttonStyle(FlatButtonStyle()).disabled(context.owners.isEmpty || context.rules.count>=128 || context.hasUncommittedNumbers)
            if context.owners.isEmpty {Text(inlinePresentation ? "Connect a complete Model definition to this Train to choose weights and biases." : "Connect this recipe to Train and connect a complete Model definition to Train. Its weights and biases will appear here.").font(.caption).foregroundStyle(MLTheme.muted)}
            if let message=context.validationMessage ?? context.rejection {
                Label(message,systemImage:"exclamationmark.circle").font(.caption).foregroundStyle(MLTheme.rose)
            }
            if !inlinePresentation { HStack(spacing:8) {
                Button("Apply") {_=context.apply()}.disabled(!context.canApply)
                Button("Discard") {_=context.discard()}.disabled(!context.hasChanges && !context.hasUncommittedNumbers)
                Spacer(minLength:0)
                if context.hasChanges {Text("Unapplied").font(.caption2).foregroundStyle(MLTheme.muted)}
            }.buttonStyle(FlatButtonStyle()) }
        }.disabled(context.isBusy).id(context.resetGeneration)
    }
    @ViewBuilder private func rule(_ index:Int,_ row:[String:JSONValue])->some View {
        let algorithm=row["algorithm"]?.stringValue ?? "normal"
        let owner:InitializationOwnerChoice? = {
            guard case .object(let selected)=row["owner"] else {return nil}
            return context.owners.first {$0.node==selected["node"]?.stringValue && $0.parameter==selected["parameter"]?.stringValue}
        }()
        VStack(alignment:.leading,spacing:9) {
            HStack {
                Text("Rule \(index+1)").font(.caption.weight(.semibold))
                Spacer()
                Button {_=context.removeRule(at:index)} label:{Image(systemName:"trash")}
                    .buttonStyle(.plain).disabled(context.hasUncommittedNumbers)
                    .accessibilityLabel("Remove weights or biases \(index+1)")
            }
            let layers=context.owners.reduce(into:[InitializationOwnerChoice]()) {values,item in if !values.contains(where:{$0.node == item.node}) {values.append(item)}}
            FlatChoiceControl(title:"Layer",selection:owner?.node ?? "",options:layers.map {.init(id:$0.node,title:$0.name)},placeholder:"Choose a layer") {node in
                let candidates=context.owners.filter {$0.node == node}
                if let selected=candidates.first(where:{$0.parameter == owner?.parameter}) ?? candidates.first {_=context.setOwner(selected.id,at:index)}
            }
            FlatChoiceControl(title:"Parameter",selection:owner?.id ?? "",options:context.owners.filter {$0.node == owner?.node}.map {.init(id:$0.id,title:TrainLayerPresentation.parameterTitle($0.parameter))},placeholder:"Choose weights or biases") {_=context.setOwner($0,at:index)}
            if let owner {Text(owner.shape.map(String.init).joined(separator:" × ")).font(.caption2).foregroundStyle(MLTheme.muted).accessibilityLabel("Parameter dimensions")}
            FlatChoiceControl(title:"Starting method",selection:algorithm,options:[
                .init(id:"normal",title:"Normal · center and spread"),.init(id:"zeros",title:"Zeros"),
                .init(id:"xavier_uniform",title:"Xavier uniform",isEnabled:(owner?.shape.count ?? 0)>=2),
                .init(id:"kaiming_uniform",title:"Kaiming uniform",isEnabled:(owner?.shape.count ?? 0)>=2)]) {_=context.setAlgorithm($0,at:index)}
                .disabled(context.hasUncommittedNumbers)
            if case .object(let settings)=row["settings"] {
                if algorithm=="normal" {
                    number("mean",title:"Center",index:index,settings:settings,minimum:-1_000_000,maximum:1_000_000)
                    number("std",title:"Spread",index:index,settings:settings,minimum:0,maximum:1_000_000)
                } else if algorithm=="xavier_uniform" {
                    number("gain",title:"Gain",index:index,settings:settings,minimum:0,maximum:100)
                } else if algorithm=="kaiming_uniform" {
                    number("a",title:"Slope",index:index,settings:settings,minimum:0,maximum:100)
                    FlatChoiceControl(title:"Fan direction",selection:settings["mode"]?.stringValue ?? "fan_in",options:[.init(id:"fan_in",title:"Incoming connections"),.init(id:"fan_out",title:"Outgoing connections")]) {_=context.setSetting("mode",value:.string($0),at:index)}
                    FlatChoiceControl(title:"Activation behavior",selection:settings["nonlinearity"]?.stringValue ?? "leaky_relu",options:[.init(id:"relu",title:"ReLU"),.init(id:"leaky_relu",title:"Leaky ReLU")]) {_=context.setSetting("nonlinearity",value:.string($0),at:index)}
                }
            }
        }.padding(10).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func number(_ key:String,title:String,index:Int,settings:[String:JSONValue],minimum:Double,maximum:Double)->some View {
        let token="rule.\(index).\(key)",generation=context.resetGeneration
        return VStack(alignment:.leading,spacing:4) {
            Text(title).font(.caption)
            BoundedNumberField(title:title,value:.decimal(settings[key]?.doubleValue ?? 0),domain:.decimal(minimum:minimum,maximum:maximum),origin:context.numericOrigin(token),currentOrigin:{context.resetGeneration==generation ? context.currentNumericOrigin(token):nil},initialText:context.rawNumericDrafts[token],onRawDraftChange:{text in
                guard context.resetGeneration==generation else {return};_=context.setRawNumericDraft(token,text:text,origin:context.numericOrigin(token))
            },onDraftChange:{origin,pending in
                guard context.resetGeneration==generation else {return};context.setNumericPending(token,origin:origin,pending:pending)
            }) {value in
                guard context.resetGeneration==generation,case .decimal(let n)=value else {return false}
                return context.setSetting(key,value:.number(n),at:index)
            }
        }
    }
}
