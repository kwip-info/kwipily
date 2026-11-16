import SwiftUI
import KwipMLCore

/// All controls edit a draft. No I/O, parsing or hashing in layout/gesture callbacks.
struct APITableConfigurationView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    @ObservedObject var store:DatabaseOutputStore
    @Binding var options:[String:JSONValue]
    let output:Bool
    var inspect:(()->Void)?=nil
    @State private var selected:Int?
    @State private var search=""
    @State private var profileSheet=false
    @State private var updating:String?
    @State private var error:String?
    @State private var payloadPreview:String?
    private var address:WorkflowNodeAddress {.init(domain:.data,localID:context.node?.id ?? "")}
    private var rows:[[String:JSONValue]] {if case .array(let values)=options["mapping"] {return values.compactMap {if case .object(let r)=$0 {return r};return nil}};return []}
    private var origin:String? {try? APITableSettings.origin(options)}
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            InspectorSection("Request",symbol:"network",identifier:"table-api.request",initiallyExpanded:true) {
                choice("Method","method",output ? [("POST","POST"),("PUT","PUT"),("PATCH","PATCH"),("DELETE","DELETE")]:[("GET","GET")])
                text("URL","url",placeholder:"https://api.example.com/orders")
                choice("Authentication","auth",[("none","None"),("bearer","Bearer token"),("basic","Basic authentication"),("apiKey","API key header")])
                if options["auth"] == .string("apiKey") {text("Credential header","credentialHeader")}
                profilePicker
                Toggle("Allow HTTP on literal loopback · local development",isOn:Binding(get:{options["allowLoopback"] == .bool(true)},set:{options["allowLoopback"] = .bool($0)})).toggleStyle(FileInputCheckboxStyle(tint:ComponentVisualFamily.data.primary))
                InspectorSection("Query parameters",symbol:"list.bullet",identifier:"table-api.query",initiallyExpanded:false) {APIRequestFields(value:object("query"))}
                InspectorSection("Headers",symbol:"list.bullet",identifier:"table-api.headers",initiallyExpanded:false) {APIRequestFields(value:object("headers"))}
            }
            if !output {pagination}
            if output {
                InspectorSection("Payload",symbol:"tablecells",identifier:"table-api.payload",initiallyExpanded:true) {
                    choice("Delivery","delivery",[("row","One request per row"),("batch","Batch rows")])
                    if options["delivery"] == .string("batch") {number("Rows per request","batchRows");text("Array path · empty sends a root array","arrayPath")}
                    mapping
                    constantsEditor
                    Button("Preview payload structure") {makePreview()}.buttonStyle(FlatButtonStyle())
                    if let payloadPreview {Text("Structure only · column placeholders").foregroundStyle(MLTheme.muted);ScrollView([.horizontal,.vertical]) {Text(payloadPreview).font(.system(.caption,design:.monospaced)).textSelection(.enabled).padding(7)}.frame(height:130).overlay(Rectangle().stroke(MLTheme.rule))}
                    Text("Null cells become JSON null. JSON fields use /parent/field paths.").foregroundStyle(MLTheme.muted)
                }
            } else {
                InspectorSection("Decode & columns",symbol:"tablecells",identifier:"table-api.columns",initiallyExpanded:true) {
                    text("Records path · JSON Pointer","recordsPath",placeholder:"/data/orders")
                    mapping
                    choice("Missing fields","missing",[("error","Stop with error"),("null","JSON null")])
                    Text("Only mapped fields are included. Other fields are ignored; type mismatches stop the capture.").foregroundStyle(MLTheme.muted)
                }
            }
            InspectorSection(output ? "Delivery policy":"Limits & rate handling",symbol:"slider.horizontal.3",identifier:"table-api.limits",initiallyExpanded:false) {
                number("Maximum response bytes · entire operation","maximumBytes");number("Total timeout · seconds","timeoutSeconds");number("Request timeout · seconds","requestTimeout");number("Requests per second","requestsPerSecond")
                if output {
                    number("Maximum requests","maximumRequests");text("Idempotency header · optional","idempotencyHeader")
                    Text("No automatic write retries. A stable operation/request key is sent only in the configured header. Concurrency is one request. Accepted HTTP 200–299 does not confirm downstream completion.").foregroundStyle(MLTheme.muted)
                    text("Response JSON path · optional","responsePath")
                } else {number("Maximum rows","maximumRows");number("Read retries · 0–3","retries");Text("Retries apply to 429, 502, 503 and 504. Retry-After is honored within the total timeout.").foregroundStyle(MLTheme.muted)}
            }
            if !output,let inspect {Button("Fetch snapshot & inspect response…",action:inspect).buttonStyle(FlatButtonStyle());Text("Sends the saved request and captures all configured pages. Run Flow fetches a new snapshot. History and recorded output use captured data.").foregroundStyle(MLTheme.muted)}
            if output {Text("Run Flow sends requests. Apply saves settings. Uncertain or partial delivery requires review before a new write.").foregroundStyle(MLTheme.muted)}
            if let error {Text(error).foregroundStyle(MLTheme.rose)}
        }.font(.caption).onChange(of:options) {_,_ in payloadPreview=nil}.sheet(isPresented:$profileSheet) {APIConnectionEditor(store:store,address:address,config:options,output:output,updating:updating)}
    }
    private var profilePicker:some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Private connection profile")
            FlatChoiceControl(title:"API connection profile",selection:store.profile(for:address)?.id ?? "",options:[.init(id:"",title:"Choose a connection")]+store.profiles.filter {$0.endpoint["kind"] == .string("api") && $0.endpoint["origin"]?.stringValue==origin}.map {.init(id:$0.id,title:$0.name)}) {id in do{try store.bind(id,address:address)}catch{self.error=error.localizedDescription}}
            HStack {Button("New…") {updating=nil;profileSheet=true};Button("Manage…") {updating=store.profile(for:address)?.id;profileSheet=true}.disabled(store.profile(for:address)?.endpoint["kind"] != .string("api"))}.buttonStyle(FlatButtonStyle())
            Text("Origin and credential stay bound to this Mac. Changing the origin/authentication requires a matching connection.").foregroundStyle(MLTheme.muted)
        }
    }
    private var pagination:some View {
        InspectorSection("Pagination",symbol:"arrow.right",identifier:"table-api.pagination",initiallyExpanded:false) {
            choice("Pagination","pagination",[("none","None"),("cursor","Cursor"),("page","Page number"),("offset","Offset"),("nextLink","Next link · same origin")])
            if options["pagination"] == .string("cursor") {text("Cursor response path","cursorPath");text("Cursor query parameter","cursorParameter");Text("Stops when the cursor is absent or empty.").foregroundStyle(MLTheme.muted)}
            if options["pagination"] == .string("nextLink") {text("Next-link response path","nextPath")}
            if ["page","offset"].contains(options["pagination"]?.stringValue ?? "") {text("Page / offset parameter","pageParameter");number("Starting value","pageStart");number("Page size · source must honor this size","pageSize");Text("Stops after a short or empty page. Configure the endpoint’s page-size query parameter in Request.").foregroundStyle(MLTheme.muted)}
            number("Maximum pages","maximumPages")
            Text("Reaching a limit before the source ends fails the capture; it never publishes a truncated complete snapshot.").foregroundStyle(MLTheme.muted)
        }
    }
    private var mapping:some View {
        ScrollViewReader { proxy in
        VStack(alignment:.leading,spacing:7) {
            HStack(spacing:6) {
                TextField("Find column or field",text:$search).flatField()
                Button {proxy.scrollTo("api-mapping-start",anchor:.leading)} label:{Image(systemName:"chevron.left").frame(width:22,height:24)}.buttonStyle(FlatButtonStyle()).help("Show source fields").accessibilityLabel("Show API source fields")
                Button {proxy.scrollTo("api-mapping-end",anchor:.trailing)} label:{Image(systemName:"chevron.right").frame(width:22,height:24)}.buttonStyle(FlatButtonStyle()).help("Show types and removal controls").accessibilityLabel("Show API type fields")
            }
            ScrollView([.horizontal,.vertical]) {
                VStack(spacing:0) {
                    HStack {Color.clear.frame(width:18,height:1).id("api-mapping-start");Text(output ? "Source column":"Source JSON path").frame(width:165,alignment:.leading);Text(output ? "JSON field":"Column name").frame(width:155,alignment:.leading);Text("Type").frame(width:100,alignment:.leading);Color.clear.frame(width:20,height:1).id("api-mapping-end")}.padding(7).background(MLTheme.panel)
                    ForEach(Array(rows.enumerated()),id:\.offset) {i,r in
                        if search.isEmpty || (r["source"]?.stringValue ?? "").localizedCaseInsensitiveContains(search) || (r["target"]?.stringValue ?? "").localizedCaseInsensitiveContains(search) {
                            HStack(spacing:6) {
                                Button {selected=i} label:{Image(systemName:selected==i ? "checkmark.circle.fill":"line.3.horizontal").frame(width:18,height:28).foregroundStyle(selected==i ? ComponentVisualFamily.data.primary:MLTheme.muted)}.buttonStyle(.plain).accessibilityLabel("Select API mapping "+(r["source"]?.stringValue ?? "column"))
                                if output {FlatChoiceControl(title:"API source column",selection:r["source"]?.stringValue ?? "",options:sourceChoices(r)) {edit(i,"source",.string($0))}.frame(width:165)}
                                else {TextField("/field",text:rowText(i,"source")).flatField().frame(width:165)}
                                TextField(output ? "/field":"column",text:rowText(i,"target")).flatField().frame(width:155)
                                FlatChoiceControl(title:"API mapped type",selection:r["type"]?.stringValue ?? "string",options:APITableSettings.types.map {.init(id:$0,title:$0)}) {edit(i,"type",.string($0))}.frame(width:100)
                                Button {var next=rows;next.remove(at:i);save(next)} label:{Image(systemName:"minus.circle")}.buttonStyle(.plain).accessibilityLabel("Remove API mapping")
                            }.padding(7).background(selected==i ? ComponentVisualFamily.data.primary.opacity(0.15):MLTheme.background).contentShape(Rectangle()).onTapGesture {selected=i}
                            Divider()
                        }
                    }
                }.fixedSize(horizontal:true,vertical:false)
            }.frame(maxHeight:300).overlay(Rectangle().stroke(MLTheme.rule))
            HStack {
                Button("Add column") {var next=rows;let col=context.upstreamSchema.first {col in !next.contains {row in row["source"] == .string(col.name)}};let source=output ? (col?.name ?? context.upstreamSchema.first?.name ?? ""):"/field\(next.count+1)";next.append(["source":.string(source),"target":.string(output ? "/"+source:"column\(next.count+1)"),"type":.string("string")]);save(next)}
                if output {Button("Auto-map by name") {save(context.upstreamSchema.map {col in ["source":.string(col.name),"target":.string("/"+col.name.replacingOccurrences(of:"~",with:"~0").replacingOccurrences(of:"/",with:"~1")),"type":.string(APITableSettings.types.contains(col.dtype) ? col.dtype:"string")]})}}
                Spacer();Button {move(-1)} label:{Image(systemName:"arrow.up")}.disabled(selected==nil || selected==0);Button {move(1)} label:{Image(systemName:"arrow.down")}.disabled(selected==nil || selected==rows.count-1)
            }.buttonStyle(FlatButtonStyle())
        }
        }
    }
    private func sourceChoices(_ row:[String:JSONValue])->[FlatChoiceOption] {
        var values=context.upstreamSchema.map {FlatChoiceOption(id:$0.name,title:$0.name)}
        if let saved=row["source"]?.stringValue,!saved.isEmpty,!values.contains(where:{$0.id==saved}) {values.insert(.init(id:saved,title:saved+" · awaiting source schema"),at:0)}
        return values
    }
    private var constants:[[String:JSONValue]] {if case .array(let a)=options["constants"] {return a.compactMap {if case .object(let r)=$0{return r};return nil}};return []}
    private var constantsEditor:some View {
        VStack(alignment:.leading,spacing:6) {
            ForEach(Array(constants.enumerated()),id:\.offset) {i,r in
                VStack(alignment:.leading,spacing:4) {
                    Text("Constant JSON field")
                    TextField("/source",text:constantText(i,"target")).flatField()
                    Text("JSON value · quote text values")
                    HStack {TextField("\"canvas\"",text:constantText(i,"json")).flatField();Button {var c=constants;c.remove(at:i);options["constants"] = .array(c.map(JSONValue.object))} label:{Image(systemName:"minus.circle")}.buttonStyle(.plain)}
                }
            }
            Button("Add constant") {var c=constants;c.append(["target":.string("/constant\(c.count+1)"),"json":.string("null")]);options["constants"] = .array(c.map(JSONValue.object))}.buttonStyle(FlatButtonStyle()).disabled(constants.count>=32)
        }
    }
    private func constantText(_ i:Int,_ key:String)->Binding<String> {Binding(get:{constants.indices.contains(i) ? constants[i][key]?.stringValue ?? "":""},set:{v in var c=constants;guard c.indices.contains(i) else{return};c[i][key] = .string(v);options["constants"] = .array(c.map(JSONValue.object))})}
    private func makePreview() {
        func set(_ path:String,_ value:JSONValue,in object:inout [String:JSONValue]) {
            let keys=path.split(separator:"/",omittingEmptySubsequences:false).dropFirst().map {String($0).replacingOccurrences(of:"~1",with:"/").replacingOccurrences(of:"~0",with:"~")}
            func insert(_ parts:ArraySlice<String>,_ o:inout [String:JSONValue]) {guard let k=parts.first else{return};if parts.count==1 {o[k]=value}else {var child:[String:JSONValue]=[:];if case .object(let v)=o[k] {child=v};insert(parts.dropFirst(),&child);o[k] = .object(child)}}
            insert(keys[...],&object)
        }
        var item:[String:JSONValue]=[:]
        for r in rows {set(r["target"]?.stringValue ?? "",.string("<"+(r["source"]?.stringValue ?? "column")+":"+(r["type"]?.stringValue ?? "string")+">"),in:&item)}
        for r in constants {if let raw=r["json"]?.stringValue,let value=try? JSONDecoder().decode(JSONValue.self,from:Data(raw.utf8)) {set(r["target"]?.stringValue ?? "",value,in:&item)}}
        var value=JSONValue.object(item)
        if options["delivery"] == .string("batch") {value = .array([value]);if let path=options["arrayPath"]?.stringValue,!path.isEmpty {var envelope:[String:JSONValue]=[:];set(path,value,in:&envelope);value = .object(envelope)}}
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys,.withoutEscapingSlashes];payloadPreview=(try? encoder.encode(value)).flatMap {String(data:$0,encoding:.utf8)}
    }
    private func move(_ offset:Int){guard let i=selected,rows.indices.contains(i+offset) else{return};var next=rows;next.swapAt(i,i+offset);save(next);selected=i+offset}
    private func save(_ r:[[String:JSONValue]]){options["mapping"] = .array(r.map(JSONValue.object))}
    private func edit(_ i:Int,_ key:String,_ v:JSONValue){var next=rows;guard next.indices.contains(i) else{return};next[i][key]=v;save(next)}
    private func rowText(_ i:Int,_ key:String)->Binding<String> {Binding(get:{rows.indices.contains(i) ? rows[i][key]?.stringValue ?? "":""},set:{edit(i,key,.string($0))})}
    private func object(_ key:String)->Binding<[String:JSONValue]> {Binding(get:{if case .object(let o)=options[key] {return o};return [:]},set:{options[key] = .object($0)})}
    private func text(_ title:String,_ key:String,placeholder:String="")->some View {VStack(alignment:.leading,spacing:4) {Text(title);TextField(placeholder,text:Binding(get:{options[key]?.stringValue ?? ""},set:{options[key] = .string($0)})).flatField()}}
    private func number(_ title:String,_ key:String)->some View {VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:Binding(get:{options[key]?.integerValue.map(String.init) ?? options[key]?.stringValue ?? ""},set:{options[key] = Int64($0).map(JSONValue.integer) ?? .string($0)})).flatField()}}
    private func choice(_ title:String,_ key:String,_ values:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:options[key]?.stringValue ?? "",options:values.map {.init(id:$0.0,title:$0.1)}) {options[key] = .string($0)}}}
}

private struct APIConnectionEditor:View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var store:DatabaseOutputStore
    let address:WorkflowNodeAddress
    let config:[String:JSONValue]
    let output:Bool
    let updating:String?
    @State private var name=""
    @State private var username=""
    @State private var password=""
    @State private var writes=false
    @State private var error:String?
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            Text("Private API connection").font(.headline)
            Text((try? APITableSettings.origin(config)) ?? "Enter a valid request URL first").font(.caption).textSelection(.enabled)
            TextField("Connection name",text:$name).flatField()
            if config["auth"] == .string("basic") {TextField("Username",text:$username).flatField()}
            if config["auth"] != .string("none") {SecureField("Token / password / API key",text:$password).flatField()}
            Toggle("Allow Table Output requests to this origin",isOn:$writes).toggleStyle(FileInputCheckboxStyle(tint:ComponentVisualFamily.data.primary))
            Text("Credentials are stored in Keychain, outside projects and History. Existing secrets are never prefilled. Only this origin and authentication mode may use the profile.").font(.caption).foregroundStyle(MLTheme.muted)
            if let error {Text(error).foregroundStyle(MLTheme.rose)}
            HStack {Button("Cancel") {dismiss()};Spacer();Button("Save connection") {do {try store.saveProfile(name:name,config:config,username:username,password:password,allowWrites:writes,address:address,updating:updating);password="";dismiss()}catch{self.error=error.localizedDescription}}}.buttonStyle(FlatButtonStyle())
        }.padding(20).frame(width:430).background(MLTheme.background).onAppear {if let p=store.profiles.first(where:{$0.id==updating}) {name=p.name;writes=p.allowWrites}else{writes=output}}
    }
}

private struct APIRequestFields:View {
    @Binding var value:[String:JSONValue]
    struct Entry:Identifiable {let id=UUID();var name:String;var value:String}
    @State private var rows:[Entry]=[]
    var body:some View {
        VStack(alignment:.leading,spacing:6) {
            ForEach($rows) {$row in HStack {TextField("Name",text:$row.name).flatField();TextField("Value",text:$row.value).flatField();Button {rows.removeAll {$0.id==row.id}} label:{Image(systemName:"minus.circle")}.buttonStyle(.plain)}}
            Button("Add field") {rows.append(.init(name:"",value:""))}.buttonStyle(FlatButtonStyle())
            if Set(rows.filter {!$0.name.isEmpty}.map(\.name)).count != rows.filter({!$0.name.isEmpty}).count {Text("Use distinct field names.").foregroundStyle(MLTheme.rose)}
        }.onAppear {rows=value.keys.sorted().map {.init(name:$0,value:value[$0]?.stringValue ?? "")}}
        .onChange(of:value) {_,next in guard !next.keys.contains(where:{$0.contains("\0")}) else{return};var current:[String:JSONValue]=[:];for r in rows where !r.name.isEmpty {current[r.name] = .string(r.value)};if current != next || rows.filter({!$0.name.isEmpty}).count != next.count {rows=next.keys.sorted().map {.init(name:$0,value:next[$0]?.stringValue ?? "")}}}
        .onChange(of:rows.map {$0.name+"\0"+$0.value}) {_,_ in var next:[String:JSONValue]=[:];for r in rows {if r.name.isEmpty {if !r.value.isEmpty {next["\0unnamed"] = .string(r.value)}}else {if next[r.name] != nil {next["\0duplicate"] = .string(r.name)};next[r.name] = .string(r.value)}};value=next}
    }
}
