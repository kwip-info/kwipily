import SwiftUI
import AppKit

struct WorkflowEvaluationExportView:View {
    @ObservedObject var export:WorkflowEvaluationExportStore
    let onClose:()->Void
    @State private var folderName="KwipML evaluation"
    @State private var choosing=false
    @State private var destinationError:String?
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {
                Text("Export evaluation").font(.headline)
                Spacer()
                Button("Close",action:onClose).buttonStyle(FlatButtonStyle())
            }
            Text("Portable model, frozen preparation and saved evaluation").font(.caption).foregroundStyle(MLTheme.muted)
            Toggle("Include \(export.partitionTitle.lowercased()) rows and predictions",isOn:Binding(get:{export.includesTestRows},set:{export.setIncludesTestRows($0)}))
                .toggleStyle(.checkbox).font(.caption).disabled(choosing)
                .accessibilityIdentifier("workflow.evaluation.export.includeRows")
            Text("Default: no source rows, row identifiers, predictions or optimizer state. Learned statistics and categorical vocabularies are included; they can reveal information about the source data.")
                .font(.caption).foregroundStyle(MLTheme.muted)
            if export.includesTestRows {
                Text("This option includes every column at the selected \(export.partitionTitle.lowercased()) endpoint, including columns the model does not use. Review those columns below before saving.")
                    .font(.caption).foregroundStyle(MLTheme.rose)
            }
            Divider().overlay(MLTheme.rule)
            Text(export.statusMessage).font(.caption).textSelection(.enabled)
            if let package=export.package {
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:12) {
                        Text("\(package.fileCount) files · \(ByteCountFormatter.string(fromByteCount:Int64(package.totalBytes),countStyle:.file)) · \(package.fittedStateCount) frozen transforms").font(.caption.weight(.medium))
                        Text("Input: \(package.dtype) · shape [N, \(package.sampleShape.map(String.init).joined(separator:", "))]").font(.caption)
                        Text("Model feature order: "+package.featureColumns.joined(separator:" → ")).font(.caption).textSelection(.enabled)
                        Text("Target: "+package.targetColumn).font(.caption).textSelection(.enabled)
                        Text(package.includesTestRows ? "Included \(export.partitionTitle.lowercased()) table · \(package.testCount) observations":"Required external input table").font(.caption.weight(.semibold))
                        ForEach(package.columns,id:\.name) {column in
                            HStack {
                                Text(column.name).textSelection(.enabled)
                                Spacer()
                                Text(column.dtype.rawValue+(column.nullable ? " · nullable":"")).foregroundStyle(MLTheme.muted)
                            }.font(.system(size:11,design:.monospaced))
                        }
                        Divider().overlay(MLTheme.rule)
                        Text("Package files").font(.caption.weight(.semibold))
                        ForEach(package.files.sorted(by:{$0.path<$1.path}),id:\.path) {file in
                            HStack {Text(file.path);Spacer();Text(ByteCountFormatter.string(fromByteCount:Int64(file.bytes),countStyle:.file)).foregroundStyle(MLTheme.muted)}
                                .font(.system(size:11,design:.monospaced))
                        }
                        Text("manifest.json").font(.system(size:11,design:.monospaced))
                        Text(package.includesTestRows ? "python portable_evaluation.py --replay --batch-size 256 --device cpu":"python portable_evaluation.py --input observations.json --batch-size 256 --device cpu")
                            .font(.system(size:11,design:.monospaced)).textSelection(.enabled)
                        Text("Install requirements.txt first. External observations are reported as external input. Local inspection history is not transferred; this package does not establish an untouched holdout. Use executable model packages only from sources you trust.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }.padding(.trailing,8)
                }
            } else {Spacer(minLength:10)}
            HStack {
                Text("Folder").font(.caption).foregroundStyle(MLTheme.muted)
                TextField("Package folder",text:$folderName).textFieldStyle(.plain).padding(7).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    .disabled(export.isBusy || choosing)
            }
            if let destinationError {Text(destinationError).font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {
                if export.isBusy {
                    ProgressView().controlSize(.small)
                    Button("Cancel export") {export.cancel()}.buttonStyle(FlatButtonStyle())
                } else {
                    Button(export.package == nil ? "Prepare package":"Prepare again") {_=export.prepare()}.buttonStyle(FlatButtonStyle()).disabled(choosing)
                }
                Spacer()
                Button("Choose destination…",action:chooseDestination).buttonStyle(FlatButtonStyle(prominent:true))
                    .disabled(export.isBusy || export.package == nil || choosing || !validFolderName)
            }
        }.padding(20).frame(minWidth:660,idealWidth:760,minHeight:560,idealHeight:680)
            .background(MLTheme.background).foregroundStyle(Color.primary)
    }
    private var validFolderName:Bool {(try? WorkflowExportPublication.destination(in:URL(fileURLWithPath:"/"),folderName:folderName)) != nil}
    private func chooseDestination() {
        guard !choosing,!export.isBusy,let package=export.package else {return}
        let reviewedManifest=package.manifestBytes
        choosing=true;destinationError=nil
        let panel = WorkflowExportParentChooser.panel(title: "Choose evaluation export destination", message: "Choose a parent folder. KwipML creates a new package folder without replacing existing files.", prompt: "Save here")
        panel.begin {response in
            choosing=false
            guard response == .OK,let parent=panel.url else {return}
            guard export.package?.manifestBytes==reviewedManifest,!export.isBusy else {destinationError="The review changed. Prepare and review the package again.";return}
            do {
                let destination=try WorkflowExportPublication.destination(in:parent,folderName:folderName)
                if !export.save(to:destination,access:parent) {destinationError="The evaluation or workspace changed. Prepare the export again."}
            } catch {destinationError=error.localizedDescription}
        }
    }
}
