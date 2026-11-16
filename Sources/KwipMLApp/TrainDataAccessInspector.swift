import SwiftUI
import KwipMLCore

struct TrainDataAccessInspector:View {
    @ObservedObject var context:WorkflowControlNodeContext
    @Environment(\.colorScheme) private var colorScheme
    @State private var draft = try! TrainDataAccessDraft(configuration:[:])
    @State private var saved = try! TrainDataAccessDraft(configuration:[:])
    @State private var original:[String:JSONValue]=[:]
    @State private var origin=""
    @State private var error:String?
    private var pending:Bool {draft != saved}
    private var accent:Color {colorScheme == .dark ? ComponentVisualFamily.training.primary:ComponentVisualFamily.training.secondary}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            header("Data access",symbol:"tablecells")
            Group {
                Toggle("Custom data access",isOn:$draft.accessEnabled).toggleStyle(.checkbox)
                if draft.accessEnabled {
                    choice("Observation access",value:$draft.kind,options:[("indexed","Indexed rows"),("boundedIterable","Finite training prefix"),("tensor","Tensor tuple"),("subset","Selected training positions")])
                    if draft.kind == "subset" {
                        Text("Original training positions · zero based").foregroundStyle(MLTheme.muted)
                        TextEditor(text:$draft.indices).font(.system(size:11,design:.monospaced)).frame(height:72).scrollContentBackground(.hidden).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule)).accessibilityLabel("Selected training positions")
                    }
                    if draft.kind == "boundedIterable" {number("Maximum training rows",value:$draft.maximumRows)}
                } else {Label("Prepared training data · default access",systemImage:"checkmark.circle").foregroundStyle(MLTheme.muted)}
            }
            header("Batches",symbol:"square.stack.3d.up")
            Group {
                Toggle("Custom batches",isOn:$draft.batchingEnabled).toggleStyle(.checkbox)
                if draft.batchingEnabled {
                    number("Batch size",value:$draft.batchSize)
                    Toggle("Shuffle training rows",isOn:$draft.shuffle).toggleStyle(.checkbox)
                    number("Ordering seed",value:$draft.seed)
                    Toggle("Drop incomplete final batch",isOn:$draft.dropLast).toggleStyle(.checkbox)
                    choice("Collation",value:$draft.collation,options:[("auto","Automatic"),("fixed_stack","Fixed shapes"),("padded_sequence","Padded sequences"),("whole_bags","Whole bags")])
                    DisclosureGroup("Advanced") {
                        choice("Reader processes",value:$draft.workers,options:[("0","0 · Main worker"),("1","1"),("2","2")])
                        if draft.workers != "0" {number("Prefetch batches per reader",value:$draft.prefetch);number("Reader timeout (seconds)",value:$draft.timeout)}
                    }.tint(accent)
                } else {Label("Batch size from Train run settings",systemImage:"arrow.turn.down.right").foregroundStyle(MLTheme.muted)}
            }
            HStack {
                Button("Apply",action:apply).buttonStyle(FlatButtonStyle()).foregroundStyle(accent).background(accent.opacity(0.1)).disabled(!pending)
                Button("Revert",action:reload).buttonStyle(FlatButtonStyle()).disabled(!pending)
                Spacer()
                if pending {Image(systemName:"pencil.circle").foregroundStyle(accent).accessibilityLabel("Unapplied data access changes")}
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
        }.font(.caption).tint(accent).onAppear(perform:reload)
            .onChange(of:context.configuration) {_,next in
                if !pending {reload()}
                else if context.isCurrent, next["datasetAccess"] == original["datasetAccess"], next["batching"] == original["batching"] {
                    original=next;origin=context.numericOrigin(key:"trainDataAccess");publishPending()
                } else if next != original {error="Saved Train settings changed. Revert to load them; your draft is preserved."}
            }
            .onChange(of:draft) {_,_ in publishPending()}
            .onDisappear {context.setNumericPending(origin:origin,key:"trainDataAccess",pending:false)}
    }
    private func header(_ title:String,symbol:String)->some View {Label(title,systemImage:symbol).fontWeight(.semibold).foregroundStyle(accent).padding(8).frame(maxWidth:.infinity,alignment:.leading).background(accent.opacity(0.08)).overlay(Rectangle().stroke(accent.opacity(0.25)))}
    private func number(_ title:String,value:Binding<String>)->some View {VStack(alignment:.leading,spacing:4) {Text(title);TextField(title,text:value).flatField().accessibilityLabel(title)}}
    private func choice(_ title:String,value:Binding<String>,options:[(String,String)])->some View {VStack(alignment:.leading,spacing:4) {Text(title);FlatChoiceControl(title:title,selection:value.wrappedValue,options:options.map{.init(id:$0.0,title:$0.1)},accent:accent){value.wrappedValue=$0}}}
    private func publishPending() {context.setNumericPending(origin:origin,key:"trainDataAccess",pending:pending)}
    private func reload() {
        context.setNumericPending(origin:origin,key:"trainDataAccess",pending:false)
        do {original=context.configuration;draft=try .init(configuration:original);saved=draft;origin=context.numericOrigin(key:"trainDataAccess");error=nil;publishPending()}
        catch {self.error=error.localizedDescription}
    }
    private func apply() {
        do {let next=try draft.configuration(preserving:original)
            if context.applyTrainDataAccess(next,expected:original) {original=next;saved=draft;error=nil;publishPending()}
            else {error=context.statusMessage}
        } catch {self.error=error.localizedDescription}
    }
}
