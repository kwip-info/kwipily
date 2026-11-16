import SwiftUI
import KwipMLCore

struct DatabaseOutputConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    @ObservedObject var store:DatabaseOutputStore
    let node:DataFlowNode
    @Binding var database:[String:JSONValue]
    @Binding var filename:String
    let accent:Color
    @State private var search=""
    @State private var selected:Int?
    @State private var error:String?
    private var address:WorkflowNodeAddress {.init(domain:.data,localID:node.id)}
    private var engine:String {database["backend"]?.stringValue ?? "sqlite"}
    private var mode:String {database["mode"]?.stringValue ?? "append"}
    private var rows:[[String:JSONValue]] {guard case .array(let r)=database["mapping"] else{return []};return r.compactMap {if case .object(let v)=$0 {return v};return nil}}
    private var metadata:[String:JSONValue]? {store.cachedMetadata(database,filename:filename,address:address)}
    private var known:[[String:JSONValue]] {objects(metadata?["columns"])}
    private var keys:[[String:JSONValue]] {objects(metadata?["keys"])}
    private var tableNames:[String] {strings(metadata?["tables"])}
    private var currentProfile:DatabaseOutputStore.Profile? {store.profile(for:address)}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Connection",symbol:"externaldrive.connected.to.line.below",identifier:"table-output.database.connection",initiallyExpanded:true) {
                choice("Engine",value:engine,items:[("sqlite","SQLite"),("postgresql","PostgreSQL"),("mysql","MySQL")]) {value in
                    database["backend"] = .string(value);database["port"] = .integer(value=="mysql" ? 3306:5432);database["schema"] = .string(value=="postgresql" ? "public":value=="mysql" ? database["database"]?.stringValue ?? "":"main");save(rows.map {r in var next=r;let dtype=context.upstreamSchema.first {$0.name==r["source"]?.stringValue}?.dtype ?? "string";next["type"] = .string(DatabaseOutputSettings.suggestedType(dtype,backend:value));return next});database["keys"] = .array([]);selected=nil
                }
                if engine != "sqlite" {
                    DatabaseConnectionProfilePicker(store:store,address:address,config:database,onSelect:{p in for (k,v) in p.endpoint {database[k]=v}})
                    field("Host",key:"host");number("Port",key:"port");field("Database",key:"database")
                    labeled("Credential") {HStack {Image(systemName:"key");Text(currentProfile.map {$0.name+" · private login"} ?? "Choose or save a connection").foregroundStyle(MLTheme.muted);Spacer()}}
                    choice("TLS",value:database["tlsMode"]?.stringValue ?? "verifyFull",items:[("verifyFull","Verify certificate and hostname"),("disabledLoopback","Unencrypted · literal loopback only")]) {database["tlsMode"] = .string($0)}
                } else {
                    labeled("Database file") {
                        HStack {Image(systemName:"doc");Text(filename.isEmpty ? "Choose an existing SQLite database":filename).lineLimit(1);Spacer();Button("Choose…") {do {if let name=try store.chooseDatabase(address:address){filename=name}}catch{self.error=error.localizedDescription}}.buttonStyle(FlatButtonStyle())}
                        Text("Folder permission includes SQLite transaction journals.").font(.caption2).foregroundStyle(MLTheme.muted)
                    }
                }
                HStack {Button("Test connection") {store.inspect(config:database,filename:filename,address:address)}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy);if store.isBusy {Text("Use Cancel Run to stop metadata inspection.").foregroundStyle(MLTheme.muted)}}
                Text(store.status).foregroundStyle(metadata==nil ? MLTheme.muted:accent).fixedSize(horizontal:false,vertical:true)
                if engine=="postgresql" {field("Schema",key:"schema")}
                if mode=="create" {field("Table name",key:"table")}
                else {
                    choice("Table",value:database["table"]?.stringValue ?? "",items:tableNames.isEmpty ? [(database["table"]?.stringValue ?? "","Test connection to inspect tables")]:tableNames.map{($0,$0)}) {database["table"] = .string($0);database["keys"] = .array([])}
                    if metadata?["tablesTruncated"] == .bool(true) {field("Exact table name",key:"table")}
                }
            }
            choice("Write mode",value:mode,items:[("create","Create table"),("append","Append rows")]+(metadata?["upsertSupported"] == .bool(false) ? []:[("upsert","Upsert")])) {database["mode"] = .string($0);database["keys"] = .array([])}
            if mode=="create" {Text("Stop if this table already exists.").foregroundStyle(MLTheme.muted)}
            if mode=="upsert" {
                let selectedKey=keys.first { $0["columns"] == database["keys"] }?["name"]?.stringValue ?? ""
                choice("Conflict key",value:selectedKey,items:keys.isEmpty ? [("","Inspect this table’s unique keys")]:keys.map {($0["name"]?.stringValue ?? "",($0["name"]?.stringValue ?? "")+" · "+strings($0["columns"]).joined(separator:", "))}) {key in database["keys"]=keys.first {$0["name"] == .string(key)}?["columns"] ?? .array([])}
                Text("Uses every column of an existing unique key. Incoming keys must be nonmissing and unique.").font(.caption2).foregroundStyle(MLTheme.muted)
            }
            mappingGrid
            if mode=="create" {
                choice("Primary key",value:strings(database["keys"]).first ?? "",items:[("","None")]+rows.compactMap {r in r["target"]?.stringValue.map{($0,$0)}}) {name in database["keys"] = .array(name.isEmpty ? []:[.string(name)]);if let i=rows.firstIndex(where:{$0["target"] == .string(name)}) {setRow(i,"nullable",.bool(false))}}
            }
            InspectorSection("Write options",symbol:"slider.horizontal.3",identifier:"table-output.database.options",initiallyExpanded:false) {
                number("Rows per batch",key:"batchRows");number("Operation timeout (seconds)",key:"timeoutSeconds")
                Text("Rows are committed in one transaction per destination. MySQL table creation is a separate schema change; a failed row write may leave an empty new table.").font(.caption2).foregroundStyle(MLTheme.muted)
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose)}
        }.font(.caption).tint(accent).onAppear {seedIfEmpty()}.onChange(of:context.upstreamSchema) {_,_ in seedIfEmpty()}
    }
    private var mappingGrid:some View {
        VStack(alignment:.leading,spacing:6) {
            HStack {Text("Column mapping").fontWeight(.semibold).foregroundStyle(accent);Spacer();Text("\(rows.count) included").foregroundStyle(MLTheme.muted)}
            TextField("Find source or destination column",text:$search).flatField().accessibilityLabel("Find database mapping column")
            ThemedScrollView([.horizontal,.vertical]) {
                VStack(spacing:0) {
                    HStack(spacing:0) {Text(" ").frame(width:25);header("Source column",155);header("Destination column",170);header("Database type",155);header(mode=="create" ? "Null":"Key",42);Text("").frame(width:25)}.padding(.vertical,7).background(MLTheme.panel)
                    ForEach(Array(rows.enumerated()),id:\.offset) {i,row in
                        if search.isEmpty || (row["source"]?.stringValue ?? "").localizedCaseInsensitiveContains(search) || (row["target"]?.stringValue ?? "").localizedCaseInsensitiveContains(search) {
                            mappingRow(i,row)
                        }
                    }
                    ForEach(context.upstreamSchema.filter {col in !rows.contains(where:{$0["source"] == .string(col.name)}) && (search.isEmpty || col.name.localizedCaseInsensitiveContains(search))},id:\.name) {col in
                        HStack {
                            Toggle("Include \(col.name)",isOn:Binding(get:{false},set:{include in if include {includeColumn(col)}})).toggleStyle(FileInputCheckboxStyle(tint:accent)).frame(width:145,alignment:.leading)
                            Text(col.dtype).foregroundStyle(MLTheme.muted);Spacer()
                        }.frame(height:42).padding(.horizontal,25).background(MLTheme.background).overlay(alignment:.bottom){Rectangle().fill(MLTheme.rule).frame(height:1)}
                    }
                    if rows.isEmpty {Text("Refresh the connected Table Input preview, or run its upstream flow, to choose known source columns.").foregroundStyle(MLTheme.muted).padding(12)}
                }.frame(minWidth:722)
            }.frame(maxHeight:260).overlay(Rectangle().stroke(MLTheme.rule))
            HStack {Button("Include all") {includeAll()};Button("Auto-map by name") {autoMap()};Spacer();Button {move(-1)} label:{Image(systemName:"arrow.up")}.disabled(selected==nil || selected==0);Button {move(1)} label:{Image(systemName:"arrow.down")}.disabled(selected==nil || selected==rows.count-1)}.buttonStyle(FlatButtonStyle())
            Text(mode=="create" ? "Source identity remains visible when naming new columns. Keys cannot be null.":"Existing database types are read-only. Unmapped columns use server defaults.").font(.caption2).foregroundStyle(MLTheme.muted)
        }
    }
    private func mappingRow(_ i:Int,_ row:[String:JSONValue])->some View {
        HStack(spacing:0) {
            Image(systemName:"line.3.horizontal").foregroundStyle(MLTheme.muted).frame(width:25,height:42).contentShape(Rectangle()).gesture(DragGesture(minimumDistance:4).onEnded {v in moveRow(i,max(0,min(rows.count-1,i+Int((v.translation.height/42).rounded()))))})
            VStack(alignment:.leading,spacing:2) {Text(row["source"]?.stringValue ?? "").lineLimit(1);Text(context.upstreamSchema.first {$0.name==row["source"]?.stringValue}?.dtype ?? "unresolved").font(.caption2).foregroundStyle(MLTheme.muted)}.frame(width:145,alignment:.leading).padding(.horizontal,5)
            Group {
                if mode=="create" {TextField("Column name",text:Binding(get:{row["target"]?.stringValue ?? ""},set:{setRow(i,"target",.string($0))})).flatField().accessibilityLabel("Database destination for \(row["source"]?.stringValue ?? "column")")}
                else {FlatChoiceControl(title:"Destination for \(row["source"]?.stringValue ?? "column")",selection:row["target"]?.stringValue ?? "",options:known.map {.init(id:$0["name"]?.stringValue ?? "",title:$0["name"]?.stringValue ?? "")}) {setRow(i,"target",.string($0))}}
            }.frame(width:160).padding(.horizontal,5)
            Group {
                if mode=="create" {FlatChoiceControl(title:"Database type for \(row["source"]?.stringValue ?? "column")",selection:row["type"]?.stringValue ?? "TEXT",options:(DatabaseOutputSettings.types[engine] ?? []).map {.init(id:$0,title:$0)}) {setRow(i,"type",.string($0))}}
                else {Text(known.first {$0["name"]==row["target"]}?["type"]?.stringValue ?? "Inspect destination").font(.caption2).foregroundStyle(MLTheme.muted).lineLimit(1)}
            }.frame(width:145,alignment:.leading).padding(.horizontal,5)
            Group {
                if mode=="create" {Toggle("",isOn:Binding(get:{row["nullable"] == .bool(true)},set:{setRow(i,"nullable",.bool($0))})).labelsHidden().toggleStyle(FileInputCheckboxStyle(tint:accent)).disabled(strings(database["keys"]).contains(row["target"]?.stringValue ?? "")).accessibilityLabel("Nullable \(row["target"]?.stringValue ?? "column")")}
                else {Image(systemName:strings(database["keys"]).contains(row["target"]?.stringValue ?? "") ? "key":"minus").foregroundStyle(accent)}
            }.frame(width:42)
            Toggle("Include \(row["source"]?.stringValue ?? "column")",isOn:Binding(get:{true},set:{included in if !included {var all=rows;all.remove(at:i);save(all);selected=nil}})).labelsHidden().toggleStyle(FileInputCheckboxStyle(tint:accent)).frame(width:25)
        }.frame(height:42).background(selected==i ? accent.opacity(0.12):MLTheme.background).contentShape(Rectangle()).onTapGesture {selected=i}.overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
    }
    private func header(_ title:String,_ width:CGFloat)->some View {Text(title).font(.caption2.weight(.semibold)).frame(width:width,alignment:.leading)}
    private func seedIfEmpty(){if rows.isEmpty && !context.upstreamSchema.isEmpty {includeAll()}}
    private func includeAll(){var all=rows;for col in context.upstreamSchema where !all.contains(where:{$0["source"] == .string(col.name)}) {all.append(["source":.string(col.name),"target":.string(col.name),"type":.string(DatabaseOutputSettings.suggestedType(col.dtype,backend:engine)),"nullable":.bool(col.nullable ?? true)])};save(all)}
    private func includeColumn(_ col:DataColumnSchema){var all=rows;all.append(["source":.string(col.name),"target":.string(col.name),"type":.string(DatabaseOutputSettings.suggestedType(col.dtype,backend:engine)),"nullable":.bool(col.nullable ?? true)]);save(all)}
    private func autoMap(){save(rows.map {row in var r=row;if known.contains(where:{$0["name"]==r["source"]}) {r["target"]=r["source"]};return r})}
    private func setRow(_ i:Int,_ k:String,_ v:JSONValue){var all=rows;guard all.indices.contains(i) else{return};all[i][k]=v;save(all)}
    private func save(_ all:[[String:JSONValue]]){database["mapping"] = .array(all.map(JSONValue.object))}
    private func move(_ delta:Int){guard let i=selected else{return};moveRow(i,i+delta)}
    private func moveRow(_ from:Int,_ to:Int){var all=rows;guard all.indices.contains(from),all.indices.contains(to) else{return};let row=all.remove(at:from);all.insert(row,at:to);save(all);selected=to}
    private func labeled<C:View>(_ label:String,@ViewBuilder content:()->C)->some View {VStack(alignment:.leading,spacing:5){Text(label).fontWeight(.semibold);content()}}
    private func field(_ label:String,key:String)->some View {labeled(label){TextField(label,text:Binding(get:{database[key]?.stringValue ?? ""},set:{database[key] = .string($0)})).flatField().accessibilityLabel("Database \(label)")}}
    private func number(_ label:String,key:String)->some View {labeled(label){TextField(label,text:Binding(get:{database[key]?.integerValue.map(String.init) ?? ""},set:{database[key] = Int64($0).map(JSONValue.integer) ?? .string($0)})).flatField().accessibilityLabel("Database \(label)")}}
    private func choice(_ label:String,value:String,items:[(String,String)],change:@escaping(String)->Void)->some View {labeled(label){FlatChoiceControl(title:label,selection:value,options:items.map{.init(id:$0.0,title:$0.1)},onSelect:change)}}
    private func objects(_ v:JSONValue?)->[[String:JSONValue]] {guard case .array(let a)=v else{return []};return a.compactMap {if case .object(let r)=$0 {return r};return nil}}
    private func strings(_ v:JSONValue?)->[String] {guard case .array(let a)=v else{return []};return a.compactMap(\.stringValue)}
}
