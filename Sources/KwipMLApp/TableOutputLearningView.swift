import SwiftUI
import AppKit
import KwipMLCore

struct TableOutputLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store:WorkspaceStore
    @State private var copyStatus:String?
    var body:some View {
        VStack(alignment:.leading,spacing:16) {
            if !inLessonRail { LessonTextPanel(title:"Table Output · editable practice",text:DataPracticeLessonCopy.packets["table-output"]!.introduction) }
            if !inLessonRail { LessonTextPanel(title:"Words to know",text:"**Destination grant:** permission to one local folder, kept outside the graph. **Atomic write:** prepare a complete artifact before publishing its filename. **Receipt:** actual row count, bytes, format, digest and saved time. **Schema:** ordered column names, types and temporal units. **Bundle:** a table plus copied image assets and a verifiable manifest.") }
            if !inLessonRail { LessonTextPanel(title:"Predict before running",text:"The supplied table has twelve synthetic orders. Predict the written row count, including records beyond the first visible page. Which format preserves typed columns? What should happen when you run twice with Stop if existing?") }
            if !inLessonRail { LessonTextPanel(title:"Try it on the canvas",text:DataPracticeLessonCopy.packets["table-output"]!.sections.map {"**"+$0.title+"**\n"+$0.text}.joined(separator:"\n\n")) }
            Button("Open Table Output practice") {_ = store.loadComponentWorkflowExample(TableOutputPracticeExample.exampleID,device:"cpu")}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy).accessibilityIdentifier("learning.table-output.open-example")
            Button("Copy disposable SQLite practice database…") {copyDatabase()}.buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
            if let copyStatus {Text(copyStatus).font(.caption).foregroundStyle(MLTheme.muted)}
            TableOutputInformationView()
            if !inLessonRail { LessonTextPanel(title:"Check your understanding",text:DataPracticeLessonCopy.packets["table-output"]!.reflection) }
        }
    }
    private func copyDatabase() {
        let panel=NSOpenPanel();panel.title="Copy SQLite practice database";panel.message="Choose a folder for a new disposable kwip-table-output-practice.sqlite file. Existing files are never replaced.";panel.canChooseDirectories=true;panel.canChooseFiles=false;panel.canCreateDirectories=true;panel.allowsMultipleSelection=false
        guard panel.runModal() == .OK,let folder=panel.url else {return}
        let access=folder.startAccessingSecurityScopedResource();defer {if access {folder.stopAccessingSecurityScopedResource()}}
        do {let bytes=try Data(contentsOf:TableOutputPracticeExample.databaseResourceURL());try DiagnosticFileIO.writeImmutable(bytes,root:folder,parts:["kwip-table-output-practice.sqlite"],limit:64_000);copyStatus="Practice database copied. Choose this file in Table Output; Run Flow performs the database write."}
        catch {copyStatus="Could not create a new practice copy. Choose another folder; existing files were preserved."}
    }
}
