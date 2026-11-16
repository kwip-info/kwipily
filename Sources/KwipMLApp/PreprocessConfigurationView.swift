import SwiftUI
import KwipMLCore

struct PreprocessConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft=PreprocessConfiguration()
    @State private var original:[String:JSONValue]=[:]
    @State private var selected:String?
    @State private var origin=""
    @State private var error:String?
    @State private var newColumn=""
    @State private var search=""
    private var schema:[DataColumnSchema] {context.upstreamSchema}
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var pending:Bool {draft.values != ((try? PreprocessConfiguration(values:original).values) ?? original)}
    private var selectedIndex:Int? {draft.columns.firstIndex{$0.column==selected}}
    private let missing=[("keep","Preserve"),("error","Stop run"),("drop","Drop row"),("constant","Fixed value"),("mean","Mean"),("median","Median"),("mostFrequent","Most frequent")]
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            choice("Mode",value:Binding(get:{draft.mode},set:{draft.mode=$0;if $0=="frozen" {draft.columns=[];selected=nil}}),options:[("partitions","Training / Validation / Test"),("table","Table cleanup"),("frozen","Apply saved rules")])
            if draft.mode=="frozen" { frozenRules } else {
            HStack {Label("Input schema",systemImage:"tablecells");Spacer();Text(schema.isEmpty ? "Unresolved":"\(schema.count) columns")}.foregroundStyle(accent).padding(9).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.35)))
            TextField("Search column rules",text:$search).flatField().accessibilityLabel("Search Preprocess column rules")
            grid
            if schema.isEmpty {
                TextField("Exact column name",text:$newColumn).flatField().accessibilityLabel("Preprocess column name")
                Button("Add column") {add(newColumn);newColumn=""}.buttonStyle(FlatButtonStyle()).disabled(newColumn.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
            } else {
                HStack {
                    FlatChoiceControl(title:"Add preprocessing column",selection:"",options:available.map{.init(id:$0.name,title:$0.name)},placeholder:"Add column",accent:accent){add($0)}
                    Button("Add all") {for col in available {add(col.name)}}.buttonStyle(FlatButtonStyle()).disabled(available.isEmpty)
                }
            }
            if let i=selectedIndex {
                HStack {
                    Button {draft.columns.swapAt(i,i-1)} label:{Image(systemName:"arrow.up")}.buttonStyle(FlatButtonStyle()).disabled(i==0).accessibilityLabel("Move Preprocess rule up")
                    Button {draft.columns.swapAt(i,i+1)} label:{Image(systemName:"arrow.down")}.buttonStyle(FlatButtonStyle()).disabled(i==draft.columns.count-1).accessibilityLabel("Move Preprocess rule down")
                }
                selectedControls(i)
            }
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer();if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Preprocess changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).onAppear(perform:reload).onChange(of:node.id){_,_ in reload()}
            .onChange(of:draft) {_,_ in error=nil;publish()}
            .onChange(of:node.config) {_,next in if next==original {return};if !pending {reload()} else {error="Saved configuration changed. Revert to load it; your draft is preserved."}}
            .onDisappear {context.setNumericPending(origin:origin,key:"preprocessGrid",pending:false)}
    }
    private var frozenRules:some View {
        VStack(alignment:.leading,spacing:8) {
            Text(context.inputSourceName(forInput:"model") ?? "Connect Load model → Model").fontWeight(.semibold)
            if let summary=context.connectedLearnedPreparation {
                Text("Saved rules · \(summary["ruleCount"]?.integerValue ?? 0) transforms").foregroundStyle(accent)
                if let digest=summary["contentDigest"]?.stringValue {Text("Chain · "+String(digest.prefix(12))).font(.caption.monospaced()).foregroundStyle(MLTheme.muted)}
                Text("Required feature columns").fontWeight(.semibold)
                if case .array(let schema)=summary["inputSchema"] {
                    ForEach(Array(schema.enumerated()),id:\.offset) {_,column in
                        if case .object(let value)=column {HStack {Text(value["name"]?.stringValue ?? "").lineLimit(2);Spacer();Text(value["dtype"]?.stringValue ?? "").foregroundStyle(MLTheme.muted)}}
                    }
                }
                if case .array(let rules)=summary["rules"] {
                    ThemedScrollView {VStack(alignment:.leading,spacing:6) {
                        ForEach(Array(rules.enumerated()),id:\.offset) {_,value in
                            if case .object(let rule)=value {
                                VStack(alignment:.leading,spacing:3) {Text(rule["column"]?.stringValue ?? "").fontWeight(.semibold);Text("Missing: "+(rule["missing"]?.stringValue ?? "")+" · "+(rule["encoding"]?.stringValue ?? "")+" · "+(rule["scaling"]?.stringValue ?? "")).foregroundStyle(MLTheme.muted);if rule["encoding"] == .string("oneHot") {Text("Unknown: "+(rule["unknownPolicy"] == .string("ignore") ? "All-zero encoding":"Stop run")).foregroundStyle(MLTheme.muted)}}
                                Divider()
                            }
                        }
                    }.padding(8)}.frame(maxHeight:180).overlay(Rectangle().stroke(MLTheme.rule))
                }
                Text("Rules are read-only. Run transforms the connected table without fitting.").foregroundStyle(MLTheme.muted)
            } else {Text("Load a Model + learned preparation package. Numeric-only packages have no saved rules.").foregroundStyle(MLTheme.muted)}
        }
    }
    private var available:[DataColumnSchema] {schema.filter{col in !draft.columns.contains(where:{$0.column==col.name})}}
    private var grid:some View {
        VStack(spacing:0) {
            HStack(spacing:2) {Text("Column").frame(width:84,alignment:.leading);Text("Missing").frame(width:80,alignment:.leading);if draft.mode=="table" {Text("Text cleanup").frame(width:128,alignment:.leading)} else {Text("Encode").frame(width:64,alignment:.leading);Text("Scale").frame(width:64,alignment:.leading)}}.fontWeight(.semibold).padding(6).background(accent.opacity(0.08))
            ThemedScrollView {
                LazyVStack(spacing:0) {
                    ForEach(draft.columns.filter{search.isEmpty || $0.column.localizedCaseInsensitiveContains(search)}) {row in
                        HStack(spacing:2) {
                            Button {selected=row.column} label:{Text(row.column).lineLimit(1).truncationMode(.middle).frame(width:84,alignment:.leading)}.buttonStyle(.plain).help(row.column)
                            FlatChoiceControl(title:"Missing values for \(row.column)",selection:row.missing,options:missingChoices(for:row.column,compact:true),compact:true,accent:accent){set(row.column,\.missing,$0)}.frame(width:80)
                            if draft.mode=="table" {Text(row.hasTextCleanup ? "Configured":"None").frame(width:128,alignment:.leading)} else {
                            FlatChoiceControl(title:"Encoding for \(row.column)",selection:row.encoding,options:encodingChoices(for:row.column,compact:true),compact:true,accent:accent){set(row.column,\.encoding,$0)}.frame(width:64)
                            FlatChoiceControl(title:"Scaling for \(row.column)",selection:row.scaling,options:scalingChoices(for:row.column,compact:true),compact:true,accent:accent){set(row.column,\.scaling,$0)}.frame(width:64)
                            }
                        }.padding(6).background(selected==row.column ? accent.opacity(0.12):Color.clear).contentShape(Rectangle()).onTapGesture{selected=row.column}
                        Rectangle().fill(MLTheme.rule).frame(height:1)
                    }
                }
            }.frame(height:min(224,max(48,CGFloat(draft.columns.count)*42)))
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func missingChoices(for name:String,compact:Bool=false)->[FlatChoiceOption] {
        missing.filter {option in
            if draft.mode=="table" && ["mean","median","mostFrequent"].contains(option.0) {return false}
            if !schema.isEmpty && !["int64","float64"].contains(columnType(name)) && ["mean","median"].contains(option.0) {return false}
            if ["image","date","datetime","time"].contains(columnType(name)) && !["keep","error","drop"].contains(option.0) {return false}
            return true
        }.map{.init(id:$0.0,title:compact && $0.0=="mostFrequent" ? "Mode":$0.1)}
    }
    private func encodingChoices(for name:String,compact:Bool=false)->[FlatChoiceOption] {
        [.init(id:"none",title:"None")] + (draft.mode != "table" && (schema.isEmpty || columnType(name)=="string") ? [.init(id:"oneHot",title:compact ? "1-hot":"One-hot")]:[])
    }
    private func scalingChoices(for name:String,compact:Bool=false)->[FlatChoiceOption] {
        [.init(id:"none",title:"None")] + (draft.mode != "table" && (schema.isEmpty || ["int64","float64"].contains(columnType(name))) ? [.init(id:"standard",title:compact ? "Std":"Standard (Z-score)")]:[])
    }
    private func selectedControls(_ i:Int)->some View {
        let row=draft.columns[i]
        return VStack(alignment:.leading,spacing:8) {
            HStack {Text("Selected column · \(row.column)").fontWeight(.semibold).foregroundStyle(accent);Spacer();Button {draft.columns.remove(at:i);selected=draft.columns.first?.column} label:{Image(systemName:"minus")}.buttonStyle(FlatButtonStyle()).accessibilityLabel("Remove preprocessing rule for \(row.column)")}
            if draft.mode=="table" && (schema.isEmpty || columnType(row.column)=="string") {
                choice("Unicode normalization",value:binding(row.column,\.unicodeNormalization),options:[("none","None"),("NFC","NFC"),("NFKC","NFKC")])
                Toggle("Remove control characters",isOn:boolBinding(row.column,\.removeControls)).toggleStyle(FileInputCheckboxStyle(tint:accent))
                Toggle("Remove punctuation",isOn:boolBinding(row.column,\.removePunctuation)).toggleStyle(FileInputCheckboxStyle(tint:accent))
                Toggle("Remove decimal digits",isOn:boolBinding(row.column,\.removeDigits)).toggleStyle(FileInputCheckboxStyle(tint:accent))
                choice("Whitespace",value:binding(row.column,\.whitespace),options:[("none","None"),("trim","Trim ends"),("collapse","Collapse runs")])
                choice("Case",value:binding(row.column,\.casePolicy),options:[("preserve","Preserve"),("lower","Lowercase"),("upper","Uppercase")])
                choice("Empty text after cleanup",value:binding(row.column,\.emptyAfterCleanup),options:[("keep","Keep empty"),("missing","Convert to missing")])
            }
            choice("Missing values",value:binding(row.column,\.missing),options:missingChoices(for:row.column).map{($0.id,$0.title)})
            if row.missing=="constant" {constant("Replacement value",column:row.column,key:\.value,dtype:columnType(row.column))}
            if ["mean","median","mostFrequent"].contains(row.missing) {
                choice("All missing",value:binding(row.column,\.emptyPolicy),options:[("error","Stop run"),("constant","Fixed value")])
                if row.emptyPolicy=="constant" {constant("All-missing value",column:row.column,key:\.emptyValue,dtype:["mean","median"].contains(row.missing) ? "float64":columnType(row.column))}
                if row.missing=="mostFrequent" {choice("Tie handling",value:binding(row.column,\.tiePolicy),options:[("error","Stop run"),("smallest","Smallest value")])}
            }
            if draft.mode != "table" {
            choice("Encoding",value:binding(row.column,\.encoding),options:encodingChoices(for:row.column).map{($0.id,$0.title)})
            if row.encoding=="oneHot" {
                choice("Unknown categories",value:binding(row.column,\.unknownPolicy),options:[("error","Stop run"),("ignore","All-zero encoding")])
                choice("Vocabulary order",value:binding(row.column,\.vocabularyOrder),options:[("lexicographic","Lexicographic"),("firstSeen","First seen")])
                Stepper("Maximum categories: \(row.maxCategories)",value:Binding(get:{draft.columns.first(where:{$0.column==row.column})?.maxCategories ?? 32},set:{n in update(row.column){$0.maxCategories=n}}),in:1...128).accessibilityLabel("Maximum categories for \(row.column)")
            }
            choice("Scaling",value:binding(row.column,\.scaling),options:scalingChoices(for:row.column).map{($0.id,$0.title)})
            if row.scaling=="standard" {
                Toggle("Center (subtract mean)",isOn:boolBinding(row.column,\.center)).toggleStyle(FileInputCheckboxStyle(tint:accent))
                Toggle("Scale to unit variance",isOn:boolBinding(row.column,\.unitVariance)).toggleStyle(FileInputCheckboxStyle(tint:accent))
            }
            }
        }
    }
    private func columnType(_ name:String)->String {schema.first(where:{$0.name==name})?.dtype ?? "string"}
    private func add(_ name:String) {guard !name.isEmpty,!draft.columns.contains(where:{$0.column==name}),draft.columns.count<128 else {return};draft.columns.append(.init(column:name));selected=name}
    private func update(_ name:String,_ edit:(inout PreprocessConfiguration.Column)->Void) {guard let i=draft.columns.firstIndex(where:{$0.column==name}) else {return};edit(&draft.columns[i]);selected=name}
    private func set(_ name:String,_ key:WritableKeyPath<PreprocessConfiguration.Column,String>,_ value:String) {
        let dtype=columnType(name)
        let zero:JSONValue=dtype=="string" ? .string(""):dtype=="bool" ? .bool(false):.integer(0)
        update(name){row in
            row[keyPath:key]=value
            if key == \.missing && value=="constant" && !PreprocessConfiguration.matches(row.value,dtype:dtype) {row.value=zero}
            if key == \.emptyPolicy && value=="constant" && !PreprocessConfiguration.matches(row.emptyValue,dtype:dtype) {row.emptyValue=zero}
            if key == \.encoding && value=="oneHot" {row.scaling="none"}
            if key == \.scaling && value=="standard" {row.encoding="none"}
        }
    }
    private func binding(_ name:String,_ key:WritableKeyPath<PreprocessConfiguration.Column,String>)->Binding<String> {Binding(get:{draft.columns.first(where:{$0.column==name})?[keyPath:key] ?? ""},set:{set(name,key,$0)})}
    private func boolBinding(_ name:String,_ key:WritableKeyPath<PreprocessConfiguration.Column,Bool>)->Binding<Bool> {Binding(get:{draft.columns.first(where:{$0.column==name})?[keyPath:key] ?? false},set:{v in update(name){$0[keyPath:key]=v}})}
    private func choice(_ title:String,value:Binding<String>,options:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value.wrappedValue,options:options.map{.init(id:$0.0,title:$0.1)},accent:accent){value.wrappedValue=$0}}}
    private func constant(_ title:String,column:String,key:WritableKeyPath<PreprocessConfiguration.Column,JSONValue>,dtype:String)->some View {
        let current=draft.columns.first(where:{$0.column==column})?[keyPath:key] ?? .string("")
        let inferred:String=switch current {case .bool:"bool";case .integer:"int64";case .number:"float64";default:"string"}
        let dtype=schema.isEmpty ? inferred:dtype
        return VStack(alignment:.leading,spacing:4) {
            Text(title)
            if schema.isEmpty {
                FlatChoiceControl(title:"\(title) type",selection:dtype,options:[.init(id:"string",title:"Text"),.init(id:"int64",title:"Integer"),.init(id:"float64",title:"Decimal"),.init(id:"bool",title:"Boolean")],accent:accent){type in
                    let value:JSONValue=type=="string" ? .string(""):type=="bool" ? .bool(false):type=="float64" ? .number(0):.integer(0)
                    update(column){$0[keyPath:key]=value}
                }
            }
            if dtype=="bool" {
                FlatChoiceControl(title:title,selection:draft.columns.first(where:{$0.column==column})?[keyPath:key] == .bool(true) ? "true":"false",options:[.init(id:"true",title:"True"),.init(id:"false",title:"False")],accent:accent){v in update(column){$0[keyPath:key] = .bool(v=="true")}}
            } else {
                TextField(title,text:Binding(get:{let v=draft.columns.first(where:{$0.column==column})?[keyPath:key] ?? .string("");switch v {case .string(let s):return s;case .integer(let n):return String(n);case .number(let n):return String(n);default:return ""}},set:{text in
                    let value:JSONValue=dtype=="int64" ? Int64(text).map(JSONValue.integer) ?? .string(text):dtype=="float64" ? Double(text).filterFinite.map(JSONValue.number) ?? .string(text):.string(text)
                    update(column){$0[keyPath:key]=value}
                })).flatField().accessibilityLabel(title)
            }
        }
    }
    private func publish() {context.setNumericPending(origin:origin,key:"preprocessGrid",pending:pending)}
    private func reload() {context.setNumericPending(origin:origin,key:"preprocessGrid",pending:false);do {draft=try .init(values:node.config);original=node.config;selected=draft.columns.first?.column;origin=context.numericOrigin(key:"preprocessGrid");error=nil;publish()} catch {self.error=error.localizedDescription}}
    private func apply() {do {try draft.validate(schema:schema);if context.applyTableConfiguration(draft.values,expected:original) {original=draft.values;error=nil;publish()} else {error=context.statusMessage}} catch {self.error=error.localizedDescription}}
}
private extension Optional where Wrapped==Double {var filterFinite:Double? {flatMap{$0.isFinite ? $0:nil}}}
