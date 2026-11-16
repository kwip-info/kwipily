import SwiftUI
import KwipMLCore
struct WorkspaceArrangementView:View {
    @ObservedObject var store:WorkspaceStore
    @State private var name=""
    @State private var section="Build"
    @State private var renamingID:String?
    @State private var renameDraft=""
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            SectionLabel(title:"Named arrangements")
            Text("Arrangements are reusable across projects. Save the current project’s visible panels, destination, canvas presentation and viewport. Supported model, preparation, reusable and lesson panel widths are included. Applying an arrangement changes the active workspace.").font(.caption).foregroundStyle(MLTheme.muted)
            FlatChoiceControl(title:"Open section",selection:section,options:["Build","Data","Run","Code","Experiments","Settings"].map {FlatChoiceOption(id:$0,title:$0)}) {section=$0}
            HStack {
                TextField("Arrangement name",text:$name).flatField().accessibilityLabel("New arrangement name")
                Button("Save current") {store.saveWorkspaceArrangement(name:name,section:section);if store.arrangementError==nil{name=""}}.buttonStyle(FlatButtonStyle()).disabled(name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty)
            }
            ForEach(store.workspaceArrangements) {arrangement in
                VStack(alignment:.leading,spacing:6) {
                    HStack {
                        Text(arrangement.name).lineLimit(1);Spacer()
                        Button("Apply"){store.applyWorkspaceArrangement(arrangement)}.accessibilityLabel("Apply arrangement \(arrangement.name)")
                        Button("Rename"){renamingID=arrangement.id;renameDraft=arrangement.name}.accessibilityLabel("Rename arrangement \(arrangement.name)")
                        Button("Delete"){store.deleteWorkspaceArrangement(arrangement)}.accessibilityLabel("Delete arrangement \(arrangement.name)")
                    }
                    if renamingID==arrangement.id {
                        HStack {
                            TextField("Arrangement name",text:$renameDraft).flatField().accessibilityLabel("Rename arrangement")
                            Button("Save name"){store.renameWorkspaceArrangement(arrangement,name:renameDraft);if store.arrangementError==nil{renamingID=nil}}
                            Button("Cancel"){renamingID=nil}
                        }
                    }
                }.buttonStyle(FlatButtonStyle())
            }
            if let error=store.arrangementError {Text(error).font(.caption).foregroundStyle(.orange)}
        }.frame(maxWidth:650,alignment:.leading)
    }
}
