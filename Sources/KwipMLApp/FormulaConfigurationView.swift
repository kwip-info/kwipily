import SwiftUI
import UniformTypeIdentifiers
import KwipMLCore

struct FormulaConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=try! FormulaDraft(configuration:FormulaDraft.defaults)
    @State private var original:[String:JSONValue]=[:]
    @FocusState private var focusedOutput:String?
    @State private var selected:String?
    @State private var origin=""
    @State private var error:String?
    @State private var expressionRows:Set<String>=[]
    @State private var sample:TableViewQuerySession?
    @StateObject private var editorCommands=FormulaEditorCommands()
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var pending:Bool {draft.configuration != original}
    private var index:Int? {draft.rows.firstIndex{$0.id==selected}}
    private var schema:[DataColumnSchema] {context.upstreamSchema}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {Label("Input schema",systemImage:"tablecells");Spacer();Text(schema.isEmpty ? "Unresolved":"\(schema.count) columns")}.padding(9).foregroundStyle(accent).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            grid
            HStack(spacing:7) {
                FlatChoiceControl(title:"Add Formula operation",selection:"",options:operations,placeholder:"Add operation",compact:true,accent:accent) {kind in draft.add(kind:kind,column:availableColumns.first?.name ?? "text");selected=draft.rows.last?.id;if let id=selected,FormulaGuidedTemporal.kinds.contains(kind) {changeKind(id,kind)}}
                Button {if let selected {draft.move(selected,by:-1)}} label:{Image(systemName:"arrow.up")}.disabled(index == nil || index == 0).accessibilityLabel("Move Formula operation up")
                Button {if let selected {draft.move(selected,by:1)}} label:{Image(systemName:"arrow.down")}.disabled(index == nil || index == draft.rows.count-1).accessibilityLabel("Move Formula operation down")
                Button {if let selected {draft.duplicate(selected);self.selected=draft.rows[(index ?? 0)+1].id}} label:{Image(systemName:"doc.on.doc")}.disabled(index == nil || draft.rows.count>=64).accessibilityLabel("Duplicate Formula operation")
                Button {if let selected {draft.remove(selected);self.selected=draft.rows.first?.id}} label:{Image(systemName:"trash")}.disabled(index == nil).accessibilityLabel("Remove Formula operation")
            }.buttonStyle(FlatButtonStyle())
            if let index {
                Text("Selected operation").fontWeight(.semibold).foregroundStyle(accent)
                operationEditor(index)
            }
            HStack {
                Button("Test on sample",action:testSample).buttonStyle(FlatButtonStyle()).disabled(index == nil || sample?.isBusy == true)
                if sample?.isBusy == true {Button("Cancel") {sample?.cancel()}.buttonStyle(FlatButtonStyle())}
            }
            if let sample {
                Text("Draft sample · first 8 captured input rows").foregroundStyle(accent)
                FormulaSampleView(session:sample).frame(minHeight:150,maxHeight:240)
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer();if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Formula changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent)
            .onAppear(perform:reload).onChange(of:focusedOutput) {_,id in if let id {selected=id}}.onChange(of:node.id) {_,_ in reload()}
            .onChange(of:node.config) {_,next in if next == original {return};if !pending {reload()} else {error="Saved configuration changed. Revert to load it; your draft is preserved."}}
            .onChange(of:draft.configuration) {_,_ in sample?.invalidate();sample=nil;error=nil;publish()}
            .onDisappear {sample?.invalidate();context.setNumericPending(origin:origin,key:"formulaGrid",pending:false)}
    }
    private var operations:[FlatChoiceOption] {[.init(id:"expression",title:"Expression"),.init(id:"replace",title:"Replace text"),.init(id:"extract",title:"Extract fields"),.init(id:"match",title:"Match pattern")]+FormulaGuidedTemporal.kinds.map{.init(id:$0,title:FormulaGuidedTemporal(kind:$0).title)}}
    private var grid:some View {
        VStack(spacing:0) {
            HStack(spacing:0) {Text("#").frame(width:24);Text("Output").frame(width:100,alignment:.leading);Text("Operation").frame(width:108,alignment:.leading);Text("Type").frame(width:78,alignment:.leading)}.padding(6).fontWeight(.semibold).background(MLTheme.panel)
            ThemedScrollView {
                LazyVStack(spacing:0) {
                    ForEach(Array(draft.rows.enumerated()),id:\.element.id) {offset,row in
                        HStack(spacing:0) {
                            Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:24).help("Drag to reorder")
                            if row.outputs.count == 1 {
                                outputNameControl(row,output:0,compact:true)
                                    .frame(width:96).accessibilityLabel("Output name for Formula operation \(offset+1)")
                            }
                            else {Button {selected=row.id} label:{Text("\(row.outputs.count) columns").frame(width:96,alignment:.leading)}.buttonStyle(.plain)}
                            FlatChoiceControl(title:"Operation \(offset+1)",selection:row.kind,options:operations,compact:true,accent:accent) {changeKind(row.id,$0)}.frame(width:108)
                            FlatChoiceControl(title:"Type for operation \(offset+1)",selection:row.outputs.first?.dtype ?? "string",options:typeChoices,compact:true,accent:accent) {value in update(row.id) {for i in $0.outputs.indices {$0.outputs[i].dtype=value}}}.frame(width:78).disabled(row.kind != "expression")
                        }.padding(6).background(selected == row.id ? accent.opacity(0.12):Color.clear).contentShape(Rectangle())
                            .onTapGesture {selected=row.id}.accessibilityElement(children:.contain).accessibilityLabel("Formula operation \(offset+1), \(row.title)")
                            .onDrag {NSItemProvider(object:row.id as NSString)}
                            .dropDestination(for:String.self) {ids,_ in guard ids.count==1,draft.rows.contains(where:{$0.id==ids[0]}) else {return false};draft.move(ids[0],before:row.id);selected=ids[0];return true}
                        Rectangle().fill(MLTheme.rule).frame(height:1)
                    }
                }
            }.frame(height:min(232,max(56,CGFloat(draft.rows.count)*42)))
        }.overlay(Rectangle().stroke(MLTheme.rule)).frame(maxWidth:.infinity,alignment:.leading)
    }
    private var typeChoices:[FlatChoiceOption] {[.init(id:"float64",title:"Decimal"),.init(id:"int64",title:"Integer"),.init(id:"string",title:"Text"),.init(id:"bool",title:"Boolean"),.init(id:"date",title:"Date"),.init(id:"datetime",title:"DateTime"),.init(id:"time",title:"Time")]}
    private var availableColumns:[DataColumnSchema] {
        guard let index else {return schema.filter{$0.dtype != "image"}}
        guard index>0 else {return schema.filter{$0.dtype != "image"}}
        var preceding=draft;preceding.rows=Array(draft.rows.prefix(index))
        return ((try? preceding.outputSchema(schema)) ?? schema).filter{$0.dtype != "image"}
    }
    @ViewBuilder private func operationEditor(_ index:Int)->some View {
        let row=draft.rows[index]
        if row.outputs.count == 1 {
            outputTargetEditor(row,output:0)
        }
        if row.guided != nil || row.temporal != nil {
            HStack(spacing:0) {
                Button {expressionRows.remove(row.id)} label:{Text("Guided").frame(maxWidth:.infinity).padding(7).background(!expressionRows.contains(row.id) ? accent.opacity(0.12):Color.clear)}
                Button {expressionRows.insert(row.id)} label:{Text("Expression").frame(maxWidth:.infinity).padding(7).background(expressionRows.contains(row.id) ? accent.opacity(0.12):Color.clear)}
            }.buttonStyle(.plain).overlay(Rectangle().stroke(MLTheme.rule))
        }
        if row.temporal != nil && !expressionRows.contains(row.id) {
            FormulaTemporalEditor(value:temporalBinding(row.id),columns:availableColumns,accent:accent)
        } else if row.guided != nil && !expressionRows.contains(row.id) {guidedEditor(row)} else {expressionEditor(row)}
        choice("Missing result",value:rowBinding(row.id,\.nullPolicy),options:[("preserve","Preserve missing values"),("error","Stop run")])
    }
    private func expressionEditor(_ row:FormulaDraft.Row)->some View {
        VStack(alignment:.leading,spacing:7) {
            HStack {
                FlatChoiceControl(title:"Insert Formula column",selection:"",options:availableColumns.map{.init(id:$0.name,title:$0.name)},placeholder:"Insert column",compact:true,accent:accent) {editorCommands.insert("[\($0)]")}
                FlatChoiceControl(title:"Insert Formula function",selection:"",options:["lower","upper","trim","length","abs","round","is_finite","is_integer","require","coalesce","decimal_parse","decimal_is_finite","decimal_is_integer","decimal_abs","decimal_add","decimal_subtract","decimal_multiply","decimal_compare","regex_replace","regex_extract","regex_match"].map{.init(id:$0,title:$0)},placeholder:"Functions",compact:true,accent:accent) {editorCommands.insert(template($0))}
            }
            FormulaExpressionEditor(text:rowBinding(row.id,\.expression),commands:editorCommands).frame(height:110).overlay(Rectangle().stroke(MLTheme.rule)).accessibilityLabel("Formula expression")
        }
    }
    private func template(_ function:String)->String {
        let column="[\(availableColumns.first?.name ?? "column")]"
        if function.hasPrefix("decimal_") {
            let source="[\(availableColumns.first(where:{$0.dtype == "string"})?.name ?? "decimal_text")]"
            if ["decimal_add","decimal_subtract","decimal_compare"].contains(function) {return "\(function)(\(source), \"0\")"}
            if function == "decimal_multiply" {return "\(function)(\(source), \"1\")"}
            return "\(function)(\(source))"
        }
        switch function {case "regex_replace":return "regex_replace(\(column), \"pattern\", \"replacement\", \"all\", \"\")";case "regex_extract":return "regex_extract(\(column), \"(pattern)\", \"search\", \"null\", \"\", \"1\")";case "regex_match":return "regex_match(\(column), \"pattern\", \"search\", \"\")";case "require":return "require(true, \(column))";case "coalesce":return "coalesce(\(column), \"\")";default:return "\(function)(\(column))"}
    }
    private func guidedEditor(_ row:FormulaDraft.Row)->some View {
        let guided=row.guided!
        return VStack(alignment:.leading,spacing:8) {
            if availableColumns.isEmpty {field("Source column",text:guidedBinding(row.id,\.column))}
            else {choice("Source column",value:guidedBinding(row.id,\.column),options:availableColumns.filter{$0.dtype=="string"}.map{($0.name,$0.name)})}
            field("Pattern",text:guidedBinding(row.id,\.pattern))
            if guided.kind == "replace" {
                field("Replacement",text:guidedBinding(row.id,\.replacement))
                choice("Replace",value:guidedBinding(row.id,\.replaceMode),options:[("all","All matches"),("first","First match")])
            } else {choice("Matching",value:guidedBinding(row.id,\.matchMode),options:[("search","Contains match"),("full","Entire value")])}
            ForEach([("ignoreCase","Ignore case"),("multiline","Multiline"),("dotAll","Dot matches newline")],id:\.0) {flag,title in
                Toggle(title,isOn:Binding(get:{draft.rows.first(where:{$0.id==row.id})?.guided?.flags.contains(flag) == true},set:{value in updateGuided(row.id) {$0.flags.removeAll{$0==flag};if value {$0.flags.append(flag)}}})).toggleStyle(.checkbox)
            }
            if guided.kind == "extract" {
                Text("Capture groups").fontWeight(.semibold).foregroundStyle(accent)
                ForEach(row.outputs.indices,id:\.self) {i in
                    VStack(alignment:.leading,spacing:5) {
                        HStack {Text("Capture \(i+1)");Spacer();Button {removeCapture(row.id,i)} label:{Image(systemName:"minus")}.buttonStyle(.plain).disabled(row.outputs.count<=1).accessibilityLabel("Remove capture \(i+1)")}
                        field("Group number or name",text:Binding(get:{draft.rows.first(where:{$0.id==row.id})?.guided?.groups[safe:i] ?? ""},set:{value in updateGuided(row.id) {if $0.groups.indices.contains(i) {$0.groups[i]=value}}}))
                        outputTargetEditor(row,output:i)
                    }.padding(8).background(accent.opacity(0.05)).overlay(Rectangle().stroke(MLTheme.rule))
                }
                Button("Add capture") {update(row.id) {row in var g=row.guided!;g.groups.append(String(g.groups.count+1));row.guided=g;row.outputs.append(.init(name:"capture_\(g.groups.count)",dtype:"string"))}}.buttonStyle(FlatButtonStyle()).disabled(row.outputs.count>=32)
                choice("No match",value:guidedBinding(row.id,\.noMatchPolicy),options:[("null","Missing values"),("error","Stop run")])
            }
        }
    }
    @ViewBuilder private func outputNameControl(_ row:FormulaDraft.Row,output index:Int,compact:Bool=false)->some View {
        let value=outputBinding(row.id,index,\.name)
        if row.outputs[index].write == "replace" && !schema.isEmpty {
            let columns=draft.replacementColumns(for:row.id,output:index,schema:schema)
            FlatChoiceControl(title:"Existing output column",selection:value.wrappedValue,
                              options:columns.map{.init(id:$0.name,title:$0.name)},
                              placeholder:columns.isEmpty ? "No eligible columns":"Choose column",
                              compact:compact,accent:accent) {value.wrappedValue=$0}
                .disabled(columns.isEmpty)
        } else {
            TextField("Output column",text:value).focused($focusedOutput,equals:row.id).flatField()
                .accessibilityLabel("Output column")
        }
    }
    private func outputTargetEditor(_ row:FormulaDraft.Row,output index:Int)->some View {
        VStack(alignment:.leading,spacing:5) {
            choice("Write to",value:outputBinding(row.id,index,\.write),options:[("new","New column"),("replace","Replace existing column")])
            Text(row.outputs[index].write == "replace" ? "Existing column":"Output column")
            outputNameControl(row,output:index)
        }
    }
    private func removeCapture(_ id:String,_ index:Int) {update(id) {row in guard row.outputs.count>1 else {return};var g=row.guided!;g.groups.remove(at:index);row.guided=g;row.outputs.remove(at:index)}}
    private func update(_ id:String,_ body:(inout FormulaDraft.Row)->Void) {guard let i=draft.rows.firstIndex(where:{$0.id==id}) else {return};body(&draft.rows[i]);selected=id}
    private func updateGuided(_ id:String,_ body:(inout FormulaGuidedRegex)->Void) {update(id) {row in guard var guided=row.guided else {return};body(&guided);row.guided=guided}}
    private func changeKind(_ id:String,_ kind:String) {let column=availableColumns.first(where:{$0.dtype=="string"})?.name ?? "text";update(id) {row in if kind == "expression" {row.expression="0";row.outputs=Array(row.outputs.prefix(1));row.outputs[0].dtype="int64"} else if FormulaGuidedTemporal.kinds.contains(kind) {
                let source=availableColumns.first(where:{kind == "temporal_parse" ? $0.dtype == "string":kind == "temporal_timezone" ? $0.dtype == "datetime":["date","datetime","time"].contains($0.dtype)})?.name ?? column
                var temporal=FormulaGuidedTemporal(kind:kind,column:source);temporal.otherColumn=source
                row.temporal=temporal;row.outputs=Array(row.outputs.prefix(1));row.outputs[0].dtype=temporal.outputType
            } else {row.guided=FormulaGuidedRegex(kind:kind,column:column);row.outputs=Array(row.outputs.prefix(1));row.outputs[0].dtype=kind == "match" ? "bool":"string"}};expressionRows.remove(id)}
    private func rowBinding(_ id:String,_ key:WritableKeyPath<FormulaDraft.Row,String>)->Binding<String> {Binding(get:{draft.rows.first(where:{$0.id==id})?[keyPath:key] ?? ""},set:{value in update(id) {row in
        row[keyPath:key]=value
        if key == \.expression,let temporal=row.temporal {
            row.outputs=Array(row.outputs.prefix(1));row.outputs[0].dtype=temporal.outputType
        } else if key == \.expression,let guided=row.guided {
            let count=guided.kind == "extract" ? guided.groups.count:1
            row.outputs=Array(row.outputs.prefix(count))
            while row.outputs.count<count {row.outputs.append(.init(name:"capture_\(row.outputs.count+1)",dtype:"string"))}
            for i in row.outputs.indices {row.outputs[i].dtype=guided.kind == "match" ? "bool":"string"}
        }
    }})}
    private func temporalBinding(_ id:String)->Binding<FormulaGuidedTemporal> {
        Binding(get:{draft.rows.first(where:{$0.id==id})?.temporal ?? FormulaGuidedTemporal()},set:{value in update(id) {row in
            row.temporal=value
            if !row.outputs.isEmpty {row.outputs[0].dtype=value.outputType}
        }})
    }
    private func outputBinding(_ id:String,_ index:Int,_ key:WritableKeyPath<FormulaDraft.Output,String>)->Binding<String> {Binding(get:{draft.rows.first(where:{$0.id==id})?.outputs[safe:index]?[keyPath:key] ?? ""},set:{value in update(id) {if $0.outputs.indices.contains(index) {$0.outputs[index][keyPath:key]=value}}})}
    private func guidedBinding(_ id:String,_ key:WritableKeyPath<FormulaGuidedRegex,String>)->Binding<String> {Binding(get:{draft.rows.first(where:{$0.id==id})?.guided?[keyPath:key] ?? ""},set:{value in updateGuided(id) {$0[keyPath:key]=value}})}
    private func field(_ title:String,text:Binding<String>)->some View {VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:text).flatField().accessibilityLabel(title)}}
    private func choice(_ title:String,value:Binding<String>,options:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value.wrappedValue,options:options.map{.init(id:$0.0,title:$0.1)},accent:accent) {value.wrappedValue=$0}}}
    private func publish() {context.setNumericPending(origin:origin,key:"formulaGrid",pending:pending)}
    private func reload() {focusedOutput=nil;sample?.invalidate();sample=nil;context.setNumericPending(origin:origin,key:"formulaGrid",pending:false);do {draft=try .init(configuration:node.config);original=node.config;selected=draft.rows.first?.id;origin=context.numericOrigin(key:"formulaGrid");error=nil;publish()} catch {self.error=error.localizedDescription}}
    private func apply() {do {let config=try draft.validated(schema:schema);if context.applyTableConfiguration(config,expected:original) {original=config;error=nil;publish()} else {error=context.statusMessage}} catch {self.error=error.localizedDescription}}
    private func testSample() {do {let config=try draft.validated(schema:schema);guard let session=context.formulaSampleSession(configuration:config) else {throw DataFlowContractError("Run the upstream flow first to capture input rows for a draft test.")};sample?.invalidate();sample=session;_ = session.load();error=nil} catch {self.error=error.localizedDescription}}
}

private extension Array {subscript(safe index:Int)->Element? {indices.contains(index) ? self[index]:nil}}
