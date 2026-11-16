import SwiftUI
import KwipMLCore

struct FilterRowsConfigurationView<Context:DataNodeEditing>: View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft = try! FilterRowsConfigurationDraft(configuration:FilterRowsConfigurationDraft.defaults)
    @State private var savedDraft = try! FilterRowsConfigurationDraft(configuration:FilterRowsConfigurationDraft.defaults)
    @State private var original:[String:JSONValue] = [:]
    @State private var origin = ""
    @State private var error:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var schema:[DataColumnSchema] {context.upstreamSchema}
    private var pending:Bool {draft != savedDraft}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {Label("Input schema",systemImage:"tablecells");Spacer();Text(schema.isEmpty ? "Unresolved":"\(schema.count) columns")}
                .foregroundStyle(accent).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            HStack(spacing:0) {
                ForEach(["simple","expression","pattern"],id:\.self) { mode in
                    Button {draft.mode=mode} label: {Text(mode == "pattern" ? "Text pattern":mode.capitalized).frame(maxWidth:.infinity).padding(.vertical,8).background(draft.mode == mode ? accent.opacity(0.14):Color.clear)}
                        .buttonStyle(.plain).foregroundStyle(draft.mode == mode ? accent:Color.primary)
                        .accessibilityLabel("Filter mode \(mode)").accessibilityValue(draft.mode == mode ? "Selected":"")
                }
            }.overlay(Rectangle().stroke(MLTheme.rule))
            switch draft.mode {
            case "simple": simple
            case "expression": expression
            default: pattern
            }
            choice("Missing values",value:$draft.nullPolicy,options:[("error","Stop run"),("nonmatching","Send to nonmatching")])
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.10)).disabled(!pending)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer()
                if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Filter changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent)
            .onAppear(perform:reload)
            .onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node.config) {_,next in
                if !pending {reload()} else if next != original {error="Saved configuration changed. Revert to load it; your draft is preserved."}
            }
            .onChange(of:draft) {_,_ in publishPending()}
            .onDisappear {context.setNumericPending(origin:origin,key:"filterRows",pending:false)}
    }
    private var simple:some View {
        VStack(alignment:.leading,spacing:9) {
            choice("Match",value:$draft.match,options:[("all","All conditions"),("any","Any condition")])
            ForEach(draft.conditions.indices,id:\.self) {index in condition(index)}
            Button {draft.conditions.append(.init(column:schema.first?.name ?? "feature",comparison:"eq",valueType:literalType(schema.first?.dtype),valueText:literalType(schema.first?.dtype) == "boolean" ? "true":"",caseSensitive:true))} label: {Label("Add condition",systemImage:"plus")}
                .buttonStyle(FlatButtonStyle()).foregroundStyle(accent).disabled(draft.conditions.count >= 32)
        }
    }
    private func condition(_ index:Int)->some View {
        VStack(alignment:.leading,spacing:7) {
            HStack {Text("Condition \(index+1)").fontWeight(.semibold).foregroundStyle(accent);Spacer();Button {draft.conditions.remove(at:index)} label:{Image(systemName:"minus")}.buttonStyle(.plain).accessibilityLabel("Remove Filter condition \(index+1)")}
            column("Column for condition \(index+1)",value:$draft.conditions[index].column) {name in
                if let dtype=schema.first(where:{$0.name == name})?.dtype {draft.conditions[index].valueType=literalType(dtype)}
            }
            choice("Operator for condition \(index+1)",value:$draft.conditions[index].comparison,options:FilterRowsConfigurationDraft.comparisons)
            if draft.conditions[index].hasOperand {
                choice("Value type for condition \(index+1)",value:$draft.conditions[index].valueType,options:[("text","Text"),("number","Number"),("boolean","Boolean")])
                if draft.conditions[index].isList {
                    choice("List format for condition \(index+1)",value:$draft.conditions[index].valueFormat,options:[("lines","One value per line"),("json","JSON array")])
                    Text("Values").foregroundStyle(MLTheme.muted)
                    TextEditor(text:$draft.conditions[index].valueText).font(.system(size:11,design:.monospaced)).frame(height:74).scrollContentBackground(.hidden).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule)).accessibilityLabel("Values for condition \(index+1)")
                } else if draft.conditions[index].valueType == "boolean" {
                    choice("Value for condition \(index+1)",value:$draft.conditions[index].valueText,options:[("true","True"),("false","False")])
                } else {TextField("Value",text:$draft.conditions[index].valueText).flatField().accessibilityLabel("Value for condition \(index+1)")}
                if draft.conditions[index].valueType == "text" {Toggle("Case sensitive",isOn:$draft.conditions[index].caseSensitive).toggleStyle(.checkbox).accessibilityLabel("Case sensitive for condition \(index+1)")}
            }
        }.padding(9).background(MLTheme.background).overlay(Rectangle().stroke(accent.opacity(0.3)))
    }
    private var expression:some View {
        VStack(alignment:.leading,spacing:7) {
            HStack {
                FlatChoiceControl(title:"Insert column",selection:"",options:schema.map{.init(id:$0.name,title:$0.name)},placeholder:"Insert column",compact:true,accent:accent) {draft.expression += " [\($0)]"}
                FlatChoiceControl(title:"Insert operator",selection:"",options:["AND","OR","NOT","==","!=",">",">=","<","<=","IS NULL","IS NOT NULL"].map{.init(id:$0,title:$0)},placeholder:"Operators",compact:true,accent:accent) {draft.expression += " \($0) "}
            }
            TextEditor(text:$draft.expression).font(.system(size:12,design:.monospaced)).frame(height:145).scrollContentBackground(.hidden).padding(5).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule)).accessibilityLabel("Filter expression")
        }
    }
    private var pattern:some View {
        VStack(alignment:.leading,spacing:9) {
            column("Pattern column",value:$draft.column)
            Text("Pattern (regular expression)")
            TextField("Pattern",text:$draft.pattern).flatField().font(.system(size:12,design:.monospaced)).accessibilityLabel("Filter text pattern")
            choice("Match",value:$draft.matchMode,options:[("search","Contains match"),("full","Entire value")])
            flag("Ignore case",id:"ignoreCase");flag("Multiline",id:"multiline");flag("Dot matches newline",id:"dotAll")
        }
    }
    private func flag(_ title:String,id:String)->some View {
        Toggle(title,isOn:Binding(get:{draft.flags.contains(id)},set:{enabled in draft.flags.removeAll{$0==id};if enabled {draft.flags.append(id)}})).toggleStyle(.checkbox)
    }
    private func literalType(_ dtype:String?)->String {dtype == "bool" ? "boolean":["int64","float64"].contains(dtype ?? "") ? "number":"text"}
    private func column(_ title:String,value:Binding<String>,onSelect:((String)->Void)?=nil)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title)
            if schema.isEmpty {TextField("Exact column name",text:value).flatField().accessibilityLabel(title)}
            else {FlatChoiceControl(title:title,selection:value.wrappedValue,options:schema.map{.init(id:$0.name,title:$0.name)},accent:accent) {value.wrappedValue=$0;onSelect?($0)}}
        }
    }
    private func choice(_ title:String,value:Binding<String>,options:[(String,String)])->some View {
        VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value.wrappedValue,options:options.map{.init(id:$0.0,title:$0.1)},accent:accent) {value.wrappedValue=$0}}
    }
    private func publishPending() {context.setNumericPending(origin:origin,key:"filterRows",pending:pending)}
    private func reload() {
        context.setNumericPending(origin:origin,key:"filterRows",pending:false)
        do {draft=try .init(configuration:node.config);savedDraft=draft;original=node.config;origin=context.numericOrigin(key:"filterRows");error=nil;publishPending()}
        catch {self.error=error.localizedDescription}
    }
    private func apply() {
        do {
            let configuration=try draft.validatedConfiguration(schema:schema)
            if context.applyTableConfiguration(configuration,expected:original) {original=configuration;savedDraft=draft;error=nil;publishPending()}
            else {error=context.statusMessage}
        } catch {self.error=error.localizedDescription}
    }
}
