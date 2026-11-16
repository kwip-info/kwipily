import SwiftUI
import KwipMLCore

struct DatabaseConnectionProfilePicker:View {
    @ObservedObject var store:DatabaseOutputStore
    let address:WorkflowNodeAddress
    let config:[String:JSONValue]
    var allowComponentLogin=false
    var onSelect:(DatabaseOutputStore.Profile)->Void
    @State private var editing=false
    @State private var updating:String?
    @State private var error:String?
    var body:some View {
        VStack(alignment:.leading,spacing:5) {
            Text("Connection profile").fontWeight(.semibold)
            FlatChoiceControl(title:"Connection profile",selection:store.profile(for:address).flatMap {$0.endpoint["backend"]==config["backend"] ? $0.id:nil} ?? "",options:[.init(id:"",title:allowComponentLogin ? "Component-specific login":"Choose a private connection")]+store.profiles.filter {$0.endpoint["backend"]==config["backend"]}.map {.init(id:$0.id,title:$0.name)}) {id in
                do {if id.isEmpty,allowComponentLogin {try store.unbind(address);return};guard let p=store.profiles.first(where:{$0.id==id}) else{return};try store.bind(id,address:address);onSelect(p)}catch{self.error=error.localizedDescription}
            }
            HStack {Button("New…") {updating=nil;editing=true};Button("Manage…") {updating=store.profile(for:address)?.id;editing=true}.disabled(store.profile(for:address)==nil)}.buttonStyle(FlatButtonStyle())
            if let error {Text(error).foregroundStyle(MLTheme.rose)}
        }.onAppear {try? store.reloadProfiles()}.sheet(isPresented:$editing) {DatabaseConnectionEditor(store:store,address:address,config:config,updating:updating,onSaved:onSelect)}
    }
}
private struct DatabaseConnectionEditor:View {
    @Environment(\.dismiss) var dismiss
    @ObservedObject var store:DatabaseOutputStore
    let address:WorkflowNodeAddress
    let config:[String:JSONValue]
    let updating:String?
    let onSaved:(DatabaseOutputStore.Profile)->Void
    @State private var name=""
    @State private var username=""
    @State private var password=""
    @State private var allowWrites=false
    @State private var error:String?
    var body:some View {
        VStack(alignment:.leading,spacing:14) {
            Text(updating==nil ? "New database connection":"Update database connection").font(.headline)
            Text("This Mac · "+(config["host"]?.stringValue ?? "")+" / "+(config["database"]?.stringValue ?? "")).font(.caption).foregroundStyle(MLTheme.muted)
            TextField("Connection name",text:$name).flatField()
            TextField("Username",text:$username).flatField().textContentType(.username)
            SecureField("Password",text:$password).flatField()
            Text("Credentials remain in Keychain. Existing credentials are never prefilled; enter the complete replacement login when updating.").font(.caption).foregroundStyle(MLTheme.muted)
            Toggle("Allow Table Output writes to this database",isOn:$allowWrites).toggleStyle(FileInputCheckboxStyle(tint:ComponentVisualFamily.data.primary))
            if let error {Text(error).font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {Button("Cancel") {dismiss()};Spacer();Button("Save connection") {do {try store.saveProfile(name:name,config:config,username:username,password:password,allowWrites:allowWrites,address:address,updating:updating);password="";if let p=store.profile(for:address){onSaved(p)};dismiss()}catch{self.error=error.localizedDescription}}}.buttonStyle(FlatButtonStyle())
        }.padding(20).frame(width:420).background(MLTheme.background).onAppear {if let p=store.profiles.first(where:{$0.id==updating}) {name=p.name;allowWrites=p.allowWrites}}
    }
}
