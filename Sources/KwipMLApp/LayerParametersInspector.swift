import SwiftUI
import Combine
import KwipMLCore

extension LayerParameterChoice {var title:String {TrainLayerPresentation.parameterTitle(id)}}

@MainActor final class LayerParametersInspectorContext:ObservableObject {
    private let busy:()->Bool
    private let current:()->Bool
    private let readSettings:()->[String:JSONValue]
    private let readChoices:()->[LayerParameterChoice]
    private let commit:([String:JSONValue],[String:JSONValue])->Bool
    private let reportPending:(Bool)->Void
    private let identity:String
    let family:ComponentVisualFamily
    let autoApply:Bool
    let legacyOverrides:[String]
    var choices:[LayerParameterChoice] {
return readChoices()
    }
    @Published private(set) var draft:[String:JSONValue]
    @Published private(set) var rejection:String?
    @Published private(set) var generation=0
    private var baseline:[String:JSONValue]
    private var pendingNumbers=Set<String>()
    private var observation:AnyCancellable?
    init(context:WorkflowModelNodeContext) {
        busy={context.isBusy};current={context.isCurrent};identity=context.contextIdentity;autoApply=false;family = .resolve(type:context.primitiveComponentID ?? "")
        let target=context.authority.address.localID
        legacyOverrides=context.authority.owner.document.nodes.compactMap {node in
            guard case .train(let config)=node.payload else {return nil}
            let rules=(try? config["initialization"].map {try InitializationConfiguration.decodeRecipe($0).rules}) ?? nil
            let flags=(try? config["trainability"].map {try TrainabilityConfiguration.decodeRecipe($0).owners}) ?? nil
            let matchesStart=rules?.contains {$0.owner.node==target} ?? false
            let matchesFlag=flags?.contains {if case .object(let row)=$0 {return row["node"] == .string(target)};return false} ?? false
            return matchesStart || matchesFlag ? (node.name.isEmpty ? "Train":node.name):nil
        }
        readSettings={context.authority.node?.parameterSettings ?? [:]}
        readChoices={
            if let member=try? WorkflowInlineContainer.parse(in:context.authority.owner.document,address:context.authority.address),WorkflowMixtureContainer.contains(context.authority.owner.document,root:member.root){return []}
            guard case .modelPrimitive(let type,let version,let config)=context.authority.node?.payload else {return []}
            return (try? LayerParameterInventory.choices(type:type,config:config,version:version,registry:context.authority.owner.modelRegistry)) ?? []
        }
        commit={expected,settings in
            guard context.authority.node?.parameterSettings ?? [:] == expected else {return false}
            return context.authority.apply(.setLayerParameters(address:context.authority.address,settings:settings.isEmpty ? nil:settings))
        }
        reportPending={context.setNumericPending(origin:context.numericOrigin(key:"layerParameters"),key:"layerParameters",pending:$0)}
        baseline=context.authority.node?.parameterSettings ?? [:];draft=baseline
        observation=context.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    init(store:ModelCanvasAdapter,node:GraphNode,onPending:@escaping(Bool)->Void={_ in}) {
        let bridge=LayerScopeEditingBridge(store:store,node:node)
        busy={store.isBusy};current={bridge.isCurrent};identity=UUID().uuidString;autoApply=true;legacyOverrides=[];family = .resolve(type:node.type)
        readSettings={store.graph.nodes.first {$0.id==node.id}?.parameterSettings ?? [:]}
        readChoices={
            guard let latest=store.graph.nodes.first(where:{$0.id==node.id}) else {return []}
            return (try? LayerParameterInventory.choices(type:latest.type,config:latest.config,version:store.graph.schemaVersion,registry:store.primitiveRegistry)) ?? []
        }
        commit={bridge.apply(expected:$0,settings:$1)};reportPending=onPending
        baseline=node.parameterSettings ?? [:];draft=baseline
        observation=store.objectWillChange.sink {[weak self] _ in self?.objectWillChange.send()}
    }
    var hasChanges:Bool {draft != baseline}
    var hasUncommittedNumbers:Bool {!pendingNumbers.isEmpty}
    var isBusy:Bool {!current() || busy()}
    var canApply:Bool {!isBusy && hasChanges && pendingNumbers.isEmpty && validationMessage == nil}
    var validationMessage:String? {
        do {
            try LayerParameterSettings.validate(draft)
            for (parameter,value) in draft {
                guard let choice=choices.first(where:{$0.id==parameter}) else {throw GraphConfigurationError("A configured parameter is no longer present. Restore the layer's geometry or remove its saved override.")}
                if case .object(let row)=value,case .object(let start)=row["initialization"],let algorithm=start["algorithm"]?.stringValue,["xavier_uniform","kaiming_uniform"].contains(algorithm),choice.shape.count<2 {throw GraphConfigurationError("Xavier and Kaiming require at least two parameter dimensions. Use Normal or Zeros for biases.")}
            }
            return nil
        }catch{return error.localizedDescription}
    }
    func row(_ parameter:String)->[String:JSONValue] {if case .object(let row)=draft[parameter] {return row};return [:]}
    func start(_ parameter:String)->[String:JSONValue] {if case .object(let start)=row(parameter)["initialization"] {return start};return [:]}
    private func report() {reportPending(hasChanges || !pendingNumbers.isEmpty)}
    @discardableResult func setMode(_ mode:String,parameter:String)->Bool {
        guard !isBusy,choices.contains(where:{$0.id==parameter}),["inherit","trainable","frozen"].contains(mode) else{return false}
        var value=row(parameter);value["trainability"]=mode=="inherit" ? nil:.string(mode);replace(parameter,value);return true
    }
    @discardableResult func setAlgorithm(_ algorithm:String,parameter:String)->Bool {
        guard !isBusy,pendingNumbers.isEmpty,let choice=choices.first(where:{$0.id==parameter}),algorithm=="default" || (try? InitializationConfiguration.defaults(algorithm)) != nil else{return false}
        if ["xavier_uniform","kaiming_uniform"].contains(algorithm),choice.shape.count<2 {return false}
        var value=row(parameter)
        if algorithm=="default" {value.removeValue(forKey:"initialization")}
        else {value["initialization"] = .object(["algorithm":.string(algorithm),"settings":.object((try? InitializationConfiguration.defaults(algorithm)) ?? [:])])}
        replace(parameter,value);generation+=1;return true
    }
    @discardableResult func setSetting(_ key:String,value:JSONValue,parameter:String)->Bool {
        guard !isBusy,case .object(var settings)=start(parameter)["settings"],settings[key] != nil else {return false}
        settings[key]=value;var method=start(parameter);method["settings"] = .object(settings)
        var row=row(parameter);row["initialization"] = .object(method)
        var candidate=draft;candidate[parameter] = .object(row)
        guard (try? LayerParameterSettings.validate(candidate)) != nil else {return false}
        replace(parameter,row);return true
    }
    private func replace(_ parameter:String,_ row:[String:JSONValue]) {draft[parameter]=row.isEmpty ? nil:.object(row);rejection=nil;report();if autoApply && canApply {_=apply()}}
    func origin(_ key:String)->String {identity+".layerParameters"+".\(generation)."+key}
    func pending(_ key:String,origin:String,value:Bool) {
        guard !isBusy,self.origin(key)==origin else{return}
        if value {pendingNumbers.insert(key)}else{pendingNumbers.remove(key)};report();if autoApply && canApply {_=apply()}
    }
    @discardableResult func apply()->Bool {
        guard canApply,readSettings() == baseline else {return false}
        guard commit(baseline,draft) else {rejection="Settings changed. Reopen this layer before applying.";return false}
        baseline=draft;generation+=1;report();return true
    }
    func revert() {guard !isBusy else{return};draft=baseline;pendingNumbers.removeAll();generation+=1;rejection=nil;report()}
    func release() {reportPending(false)}
}

struct LayerParametersInspector:View {
    @ObservedObject var editor:LayerParametersInspectorContext
    @Environment(\.colorScheme) private var colorScheme
    @State private var selectedGroup=""
    @State private var selectedParameter=""
    private var attentionChoices:Bool {editor.family == .layers && editor.choices.contains {AttentionParameterPresentation.isAttention($0.id)}}
    private var groups:[String] {editor.choices.reduce(into:[]) {result,item in let group=AttentionParameterPresentation.group(item.id);if !result.contains(group) {result.append(group)}}}
    private var currentGroup:String {groups.contains(selectedGroup) ? selectedGroup:groups.first ?? ""}
    private var groupChoices:[LayerParameterChoice] {editor.choices.filter {AttentionParameterPresentation.group($0.id)==currentGroup}}
    private var currentParameter:String {groupChoices.contains {$0.id==selectedParameter} ? selectedParameter:groupChoices.first?.id ?? ""}
    private var visibleChoices:[LayerParameterChoice] {attentionChoices ? groupChoices.filter {$0.id==currentParameter}:editor.choices}
    var body:some View {
        if !editor.choices.isEmpty || !editor.draft.isEmpty {
            InspectorSection(editor.family == .activations ? "Learned slope":editor.family == .embeddings ? "Weights":"Weights and biases",symbol:"square.stack.3d.up",identifier:"layer.parameters",initiallyExpanded:false,summary:editor.draft.isEmpty ? "Defaults":"Configured",accent:colorScheme == .dark ? editor.family.primary:editor.family.secondary) {
                if !editor.legacyOverrides.isEmpty {Label("Legacy Train override",systemImage:"clock.arrow.circlepath").font(.caption2).foregroundStyle(MLTheme.muted).help(editor.legacyOverrides.joined(separator:"\n"))}
                if attentionChoices {
                    if groups.count>1 {
                        Text("Layer").font(.caption)
                        FlatChoiceControl(title:"Layer",selection:currentGroup,options:groups.map {.init(id:$0,title:$0)},accent:colorScheme == .dark ? editor.family.primary:editor.family.secondary) {selectedGroup=$0;selectedParameter=""}.disabled(editor.isBusy || editor.hasUncommittedNumbers)
                    }
                    Text("Parameter").font(.caption)
                    FlatChoiceControl(title:"Parameter",selection:currentParameter,options:groupChoices.map {.init(id:$0.id,title:AttentionParameterPresentation.localTitle($0.id))},accent:colorScheme == .dark ? editor.family.primary:editor.family.secondary) {selectedParameter=$0}.disabled(editor.isBusy || editor.hasUncommittedNumbers)
                }
                ForEach(visibleChoices) {choice in
                    VStack(alignment:.leading,spacing:8) {
                        Text(editor.family == .activations ? "Learned slope":editor.family == .embeddings && choice.id=="weight" ? "Embedding weights":choice.title).font(.caption.weight(.semibold))
                        if !choice.shape.isEmpty {Text(choice.shape.map(String.init).joined(separator:" × ")).font(.caption2).foregroundStyle(MLTheme.muted)}
                        Text("Trainability").font(.caption)
                        FlatChoiceControl(title:"Trainability",selection:editor.row(choice.id)["trainability"]?.stringValue ?? "inherit",options:[.init(id:"inherit",title:"Layer default"),.init(id:"trainable",title:"Trainable"),.init(id:"frozen",title:"Frozen")],accent:colorScheme == .dark ? editor.family.primary:editor.family.secondary) {_=editor.setMode($0,parameter:choice.id)}
                        Text("Starting values").font(.caption)
                        FlatChoiceControl(title:"Starting values",selection:editor.start(choice.id)["algorithm"]?.stringValue ?? "default",options:[.init(id:"default",title:"Layer default"),.init(id:"normal",title:"Normal"),.init(id:"zeros",title:"Zeros"),.init(id:"xavier_uniform",title:"Xavier uniform",isEnabled:choice.shape.count>=2),.init(id:"kaiming_uniform",title:"Kaiming uniform",isEnabled:choice.shape.count>=2)],accent:colorScheme == .dark ? editor.family.primary:editor.family.secondary) {_=editor.setAlgorithm($0,parameter:choice.id)}
                        if case .object(let settings)=editor.start(choice.id)["settings"] {
                            ForEach(settings.keys.sorted(),id:\.self) {key in
                                if let field=field(key) {
                                    RegistryFieldControl(field:field,value:settings[key]!,origin:editor.origin(choice.id+"."+key),currentOrigin:{editor.isBusy ? nil:editor.origin(choice.id+"."+key)},onDraftChange:{origin,pending in editor.pending(choice.id+"."+key,origin:origin,value:pending)}) {value in editor.setSetting(key,value:value,parameter:choice.id)}
                                }
                            }
                        }
                    }.id(editor.generation).padding(10).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                }
                if let message=editor.rejection ?? editor.validationMessage {Label(message,systemImage:"exclamationmark.circle").font(.caption).foregroundStyle(MLTheme.rose)}
                if !editor.autoApply {HStack {
                    Button("Apply") {_=editor.apply()}.disabled(!editor.canApply).accessibilityIdentifier("layer.parameters.apply")
                    Button("Revert") {editor.revert()}.disabled(!editor.hasChanges && !editor.hasUncommittedNumbers).accessibilityIdentifier("layer.parameters.revert")
                }.buttonStyle(FlatButtonStyle())}
            }.tint(colorScheme == .dark ? editor.family.primary:editor.family.secondary).disabled(editor.isBusy).onDisappear {editor.release()}
        }
    }
    private func field(_ key:String)->ComponentField? {
        switch key {
        case "mean":return .init(key:key,title:"Center",kind:.number,defaultValue:0,min:-1_000_000,max:1_000_000)
        case "std":return .init(key:key,title:"Spread",kind:.number,defaultValue:1,min:0,max:1_000_000)
        case "gain":return .init(key:key,title:"Gain",kind:.number,defaultValue:1,min:0,max:100)
        case "a":return .init(key:key,title:"Slope",kind:.number,defaultValue:0,min:0,max:100)
        case "mode":return .init(key:key,title:"Fan direction",kind:.choice,defaultValue:.string("fan_in"),min:nil,max:nil,choices:[.init(value:"fan_in",title:"Incoming connections"),.init(value:"fan_out",title:"Outgoing connections")])
        case "nonlinearity":return .init(key:key,title:"Activation behavior",kind:.choice,defaultValue:.string("leaky_relu"),min:nil,max:nil,choices:[.init(value:"relu",title:"ReLU"),.init(value:"leaky_relu",title:"Leaky ReLU")])
        default:return nil
        }
    }
}

@MainActor private final class LayerScopeEditingBridge {
    let store:ModelCanvasAdapter
    let node:GraphNode
    let selectionGeneration:UInt64
    var token:ModelEditorLease
    init(store:ModelCanvasAdapter,node:GraphNode) {self.store=store;self.node=node;selectionGeneration=store.selectionGeneration;token=store.owner.lease(path:store.path)}
    var isCurrent:Bool {store.selectionGeneration==selectionGeneration && store.selectedNodeIDs == [node.id] && store.owner.lease(path:store.path)==token}
    func apply(expected:[String:JSONValue],settings:[String:JSONValue])->Bool {
        guard isCurrent,store.setParameterSettings(settings,nodeID:node.id,expected:expected,lease:token) else {return false}
        token=store.owner.lease(path:store.path);return true
    }
}
struct ModelScopeLayerParametersInspector:View {
    @StateObject private var editor:LayerParametersInspectorContext
    init(store:ModelCanvasAdapter,node:GraphNode,onPending:@escaping(Bool)->Void={_ in}) {_editor=StateObject(wrappedValue:.init(store:store,node:node,onPending:onPending))}
    var body:some View {LayerParametersInspector(editor:editor)}
}
