import SwiftUI
import KwipMLCore

struct TableOutputConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    let node:DataFlowNode
    @Environment(\.colorScheme) private var colorScheme
    @State private var original:[String:JSONValue]=[:]
    @State private var draft:[String:JSONValue]=[:]
    @State private var search=""
    @State private var selected:String?
    @State private var dragOrigin:Int?
    @State private var origin=""
    @State private var error:String?
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.data.primary:ComponentVisualFamily.data.secondary}
    private var databaseDestination:Bool {DatabaseOutputSettings.isDatabase(draft)}
    private var databaseBinding:Binding<[String:JSONValue]> {Binding(get:{if case .object(let c)=draft["database"] {return DatabaseOutputSettings.defaults.merging(c){_,v in v}};return DatabaseOutputSettings.defaults},set:{draft["database"] = .object($0)})}
    private var databaseFilename:Binding<String> {Binding(get:{draft["databaseFilename"]?.stringValue ?? "analytics.sqlite"},set:{draft["databaseFilename"] = .string($0)})}
    private var changed:Bool {draft != original}
    private var format:String {TableOutputConfiguration.string(draft["format"],fallback:"csv")}
    private var options:[String:JSONValue] {
        var values=TableOutputConfiguration.defaults(format)
        if case .object(let given)=draft["options"] {values.merge(given,uniquingKeysWith:{$1})}
        return values
    }
    private var columns:[String] {if case .array(let values)=draft["columns"] {return values.compactMap {if case .string(let name)=$0 {return name};return nil}};return []}
    private var names:[String] {context.upstreamSchema.map(\.name)}
    private var effectiveColumns:[String] {TableOutputColumnSelection.effective(columns,available:names)}
    private var visibleColumns:[String] {effectiveColumns.filter {search.isEmpty || $0.localizedCaseInsensitiveContains(search)}}
    private var selectedIndex:Int? {effectiveColumns.firstIndex {$0==selected}}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            choice("Destination",selection:APITableSettings.isAPI(draft) ? "api":databaseDestination ? "database":"file",choices:[("file","File"),("database","Database"),("api","API")]) {draft["destination"] = .string($0);if $0=="api",draft["api"]==nil {draft["api"] = .object(APITableSettings.defaults.merging(["direction":.string("output"),"method":.string("POST")]){_,v in v})};if $0=="database",draft["database"]==nil {draft["database"] = .object(DatabaseOutputSettings.defaults)}}
            if APITableSettings.isAPI(draft),let store=context.databaseOutputs {
                APITableConfigurationView(context:context,store:store,options:Binding(get:{if case .object(let c)=draft["api"] {return APITableSettings.defaults.merging(c){_,v in v}};return APITableSettings.defaults.merging(["direction":.string("output"),"method":.string("POST")]){_,v in v}},set:{draft["api"] = .object($0)}),output:true)
            } else if databaseDestination {
                if let store=context.databaseOutputs {DatabaseOutputConfigurationView(context:context,store:store,node:node,database:databaseBinding,filename:databaseFilename,accent:accent)}
                else {Text("Database destinations require a project canvas.").foregroundStyle(MLTheme.muted)}
            } else {
            VStack(alignment:.leading,spacing:5) {
                Text("Destination on this Mac").fontWeight(.semibold).foregroundStyle(accent)
                Text(context.outputDestinationSummary).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                Button("Choose folder…") {if !context.chooseOutputDestination() {error=context.statusMessage} else {error=nil}}.buttonStyle(FlatButtonStyle()).disabled(changed)
            }
            labeled("Filename or new bundle name") {TextField("output.csv",text:text("filename")).flatField().accessibilityLabel("Output filename")}
            choice("Existing destination",selection:draft["overwrite"] == .bool(true) ? "replace":"stop",choices:[("stop","Stop if existing"),("replace","Replace file atomically")]) {draft["overwrite"] = .bool($0 == "replace")}
            choice("Format",selection:format,choices:TableOutputConfiguration.formats) {next in
                let old=format;draft["format"] = .string(next);draft["options"] = .object([:])
                if ["csv","tsv"].contains(next),next != old {draft["separator"] = .string(next == "tsv" ? "\t":",")}
            }
            columnGrid
            choice("Image columns",selection:TableOutputConfiguration.string(draft["imagePolicy"],fallback:"stop"),choices:[("stop","Stop on image columns"),("copyAssets","Copy assets · new bundle")]) {draft["imagePolicy"] = .string($0);if $0 == "copyAssets" {draft["overwrite"] = .bool(false)}}
            if ["csv","tsv"].contains(format) {delimiter}
            ForEach(TableOutputConfiguration.fields(format).filter {TableOutputConfiguration.visible($0,format:format,options:options)}) {field in optionField(field)}
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!changed)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!changed)
                Spacer();if changed {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied Table Output changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).onAppear(perform:reload)
        .onChange(of:node.id) {_,_ in reload()}
        .onChange(of:node.config) {_,next in if next != original {if !changed {reload()} else {error="Saved settings changed. Revert to load them; your draft is preserved."}}}
        .onChange(of:draft) {_,_ in publish()}
        .onDisappear {context.setNumericPending(origin:origin,key:"tableOutputDraft",pending:false)}
    }
    private var delimiter:some View {
        VStack(alignment:.leading,spacing:5) {
            let value=TableOutputConfiguration.string(draft["separator"],fallback:",")
            let presets=[(",","Comma"),("\t","Tab"),(";","Semicolon"),("|","Pipe")]
            choice("Delimiter",selection:presets.contains {$0.0==value} ? value:"custom",choices:presets+[("custom","Custom character")]) {draft["separator"] = .string($0 == "custom" ? "^":$0)}
            if !presets.contains(where:{$0.0==value}) {TextField("Single delimiter character",text:text("separator")).flatField().accessibilityLabel("Custom delimiter")}
        }
    }
    private var columnGrid:some View {
        VStack(alignment:.leading,spacing:5) {
            HStack {Text("Included columns").fontWeight(.semibold).foregroundStyle(accent);Spacer();Text(columns.isEmpty ? "All":"\(columns.count) selected").foregroundStyle(MLTheme.muted)}
            TextField("Find column",text:$search).flatField().accessibilityLabel("Find Table Output column")
            VStack(spacing:0) {
                HStack(spacing:6) {Color.clear.frame(width:15);Text("Column").frame(maxWidth:.infinity,alignment:.leading);Text("Type").frame(width:65,alignment:.leading);Color.clear.frame(width:20)}.font(.system(size:10,weight:.semibold)).padding(7).background(MLTheme.panel)
                ThemedScrollView(.vertical) {
                    VStack(spacing:0) {
                        ForEach(visibleColumns,id:\.self) {name in columnRow(name)}
                        if visibleColumns.isEmpty {Text(effectiveColumns.isEmpty ? "Connect or inspect the source to choose columns.":"No matching columns.").foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,alignment:.leading).padding(9)}
                    }.frame(maxWidth:.infinity)
                }.frame(height:min(240,CGFloat(max(1,visibleColumns.count))*32))
                    .accessibilityIdentifier("table-output.included-columns")
                HStack {
                    Button("Select all") {draft["columns"] = .array(names.map(JSONValue.string))}.disabled(names.isEmpty).help("Include the current source columns explicitly in source order.")
                    Button("Use all") {draft["columns"] = .array([])}.disabled(columns.isEmpty).help("Include every current and future source column in source order.")
                    Spacer()
                    Button {move(-1)} label:{Image(systemName:"arrow.up")}.disabled(selectedIndex == nil || selectedIndex == 0).accessibilityLabel("Move selected output column earlier")
                    Button {move(1)} label:{Image(systemName:"arrow.down")}.disabled(selectedIndex == nil || selectedIndex == effectiveColumns.count-1).accessibilityLabel("Move selected output column later")
                }.buttonStyle(FlatButtonStyle()).padding(6).background(MLTheme.panel)
            }.overlay(Rectangle().stroke(MLTheme.rule))
            let available=names.filter {!effectiveColumns.contains($0)}
            if !available.isEmpty {FlatChoiceControl(title:"Add included column",selection:"",options:available.map {FlatChoiceOption(id:$0,title:$0)},placeholder:"Add column…") {name in var next=columns;next.append(name);draft["columns"] = .array(next.map(JSONValue.string));selected=name}}
        }
    }
    private func columnRow(_ name:String)->some View {
        HStack(spacing:6) {
            Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:15,height:32).contentShape(Rectangle()).gesture(DragGesture(minimumDistance:4,coordinateSpace:.global).onChanged {value in
                guard context.isCurrent,!context.isBusy,let current=effectiveColumns.firstIndex(of:name) else {return}
                if dragOrigin == nil {dragOrigin=current};selected=name
                let target=min(effectiveColumns.count-1,max(0,(dragOrigin ?? current)+Int((value.translation.height/32).rounded())))
                if let next=TableOutputColumnSelection.moving(name,to:target,configured:columns,available:names) {draft["columns"] = .array(next.map(JSONValue.string))}
            }.onEnded {_ in dragOrigin=nil}).accessibilityLabel("Drag \(name) to reorder")
            Button {selected=name} label:{
                HStack(spacing:6) {
                    Text(name).frame(maxWidth:.infinity,alignment:.leading).lineLimit(1).help(name)
                    Text(context.upstreamSchema.first {$0.name==name}?.dtype ?? "unresolved").foregroundStyle(MLTheme.muted).frame(width:65,alignment:.leading)
                }.contentShape(Rectangle())
            }.buttonStyle(.plain).frame(maxWidth:.infinity).accessibilityLabel("Select output column \(name)")
            Button {
                if let next=TableOutputColumnSelection.removable(name,configured:columns,available:names) {draft["columns"] = .array(next.map(JSONValue.string));if selected==name {selected=nil}}
            } label:{Image(systemName:"minus").frame(width:20,height:28).contentShape(Rectangle())}.buttonStyle(.plain).disabled(effectiveColumns.count<=1).accessibilityLabel("Remove output column \(name)")
        }.padding(.horizontal,7).frame(height:32).background(selected==name ? accent.opacity(0.1):MLTheme.background).overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
    }
    @ViewBuilder private func optionField(_ field:TableOutputConfiguration.Field)->some View {
        switch field.kind {
        case .boolean:
            Toggle(field.title,isOn:Binding(get:{options[field.id] == .bool(true)},set:{setOption(field.id,.bool($0));if format == "xlsx",field.id == "header",!$0 {setOption("autofilter",.bool(false))}})).toggleStyle(FileInputCheckboxStyle(tint:accent))
        case .choices(let choices):
            choice(field.title,selection:TableOutputConfiguration.string(options[field.id]),choices:choices) {next in
                setOption(field.id,.string(next))
                if field.id == "quoting",next == "none" {setOption("escaping",.string("escape"))}
                if field.id == "shape",next == "rowsSchema" {setOption("nullPolicy",.string("null"))}
                if field.id == "compression",!["gzip","brotli","zstd"].contains(next) {setOption("compressionLevel",.null)}
            }
        case .text:labeled(field.title) {TextField(field.title,text:optionText(field.id)).flatField().accessibilityLabel(field.title)}
        case .integer,.optionalInteger:labeled(field.title) {TextField(field.title,text:optionNumber(field.id,optional:field.kind.isOptional)).flatField().accessibilityLabel(field.title)}
        }
    }
    private func move(_ offset:Int) {guard let selected,let index=selectedIndex,let next=TableOutputColumnSelection.moving(selected,to:index+offset,configured:columns,available:names) else {return};draft["columns"] = .array(next.map(JSONValue.string))}
    private func labeled<Content:View>(_ title:String,@ViewBuilder content:()->Content)->some View {VStack(alignment:.leading,spacing:5) {Text(title).fontWeight(.semibold);content()}}
    private func choice(_ title:String,selection:String,choices:[(String,String)],change:@escaping(String)->Void)->some View {labeled(title) {FlatChoiceControl(title:title,selection:selection,options:choices.map {FlatChoiceOption(id:$0.0,title:$0.1)},onSelect:change)}}
    private func text(_ key:String)->Binding<String> {Binding(get:{TableOutputConfiguration.string(draft[key])},set:{draft[key] = .string($0)})}
    private func optionText(_ key:String)->Binding<String> {Binding(get:{TableOutputConfiguration.string(options[key])},set:{setOption(key,.string($0))})}
    private func optionNumber(_ key:String,optional:Bool)->Binding<String> {Binding(get:{if case .integer(let value)=options[key] {return String(value)};return TableOutputConfiguration.string(options[key])},set:{text in if optional,text.isEmpty {setOption(key,.null)} else if let value=Int64(text) {setOption(key,.integer(value))} else {setOption(key,.string(text))}})}
    private func setOption(_ key:String,_ value:JSONValue) {var next=options;next[key]=value;draft["options"] = .object(next)}
    private func publish() {context.setNumericPending(origin:origin,key:"tableOutputDraft",pending:changed)}
    private func reload() {context.setNumericPending(origin:origin,key:"tableOutputDraft",pending:false);original=node.config;draft=node.config;origin=context.numericOrigin(key:"tableOutputDraft");search="";selected=nil;error=nil;publish()}
    private func apply() {
        guard context.isCurrent,!context.isBusy,let definition=context.operatorDefinition else {error="This editing context is no longer current.";return}
        do {
            let checked=try TableOutputConfiguration.validate(draft,definition:definition,schema:context.upstreamSchema)
            if context.applyTableConfiguration(checked,expected:original) {original=checked;draft=checked;error=nil;publish()} else {error=context.statusMessage}
        } catch {self.error=error.localizedDescription}
    }
}
private extension TableOutputConfiguration.Field.Kind {var isOptional:Bool {if case .optionalInteger=self {return true};return false}}
