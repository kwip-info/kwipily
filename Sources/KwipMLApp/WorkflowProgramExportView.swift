import SwiftUI
import AppKit
import KwipMLCore

struct WorkflowProgramExportView:View {
    @ObservedObject var export:WorkflowProgramExportStore
    let onClose:()->Void
    @State private var folderName="KwipML inference program"
    @State private var choosing=false
    @State private var destinationError:String?
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            HStack {ComponentIcon(type:"program_export",size:28);Text("Export inference program").font(.headline);Spacer();Button("Close",action:onClose).buttonStyle(FlatButtonStyle())}
            Text("Save the evaluated computation and learned weights as a native PyTorch program.").font(.caption).foregroundStyle(MLTheme.muted)
            ThemedScrollView {
                VStack(alignment:.leading,spacing:14) {
                    framed("Configuration",symbol:"slider.horizontal.3") {
                        VStack(alignment:.leading,spacing:10) {
                            HStack(alignment:.top,spacing:12) {
                                choice("Batch shape",key:"batch_mode",options:[.init(id:"static",title:"Fixed batch"),.init(id:"bounded",title:"Bounded batch")])
                                choice("Capture device",key:"device",options:[.init(id:"cpu",title:"CPU"),.init(id:"mps",title:"Apple GPU (MPS)")])
                            }
                            HStack(alignment:.top,spacing:12) {
                                integer("Example batch",key:"example_batch_size")
                                integer("Minimum batch",key:"minimum_batch_size")
                                integer("Maximum batch",key:"maximum_batch_size")
                            }
                            Toggle("Strict graph capture",isOn:boolean("strict")).toggleStyle(.checkbox).font(.caption)
                            Toggle("Prefer deferred runtime assertions",isOn:boolean("prefer_deferred_runtime_asserts")).toggleStyle(.checkbox).font(.caption)
                            Text(ProgramExportLearningTask.captureControlExplanation).font(.caption).foregroundStyle(MLTheme.muted)
                            Text("Fixed batch: all three sizes match. Bounded batch: choose different minimum/maximum sizes and an example of at least two. Sample dimensions remain fixed. Device selection never falls back silently.").font(.caption).foregroundStyle(MLTheme.muted)
                            HStack {Button("Discard",action:export.discardDraft).buttonStyle(FlatButtonStyle());Button("Apply configuration") {_=export.applyDraft()}.buttonStyle(FlatButtonStyle()).disabled(!export.hasDraftChanges)}
                        }.disabled(export.isBusy || choosing)
                    }
                    framed("Information",symbol:"info.circle") {
                        Text("Inference only; training and optimizer state are not transferred. The package contains learned weights, buffers, a synthetic trace example and graph/debug metadata. Use the exact recorded Torch version and device. Load executable packages only from trusted sources. Bag inputs use the separate trained packet export.").font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    if let package=export.package {
                        framed("Reviewed program",symbol:"checkmark.shield") {
                            VStack(alignment:.leading,spacing:8) {
                                Text("Torch \(package.torchVersion) · \(package.configuration.device.uppercased()) · \(package.fileCount) files · \(ByteCountFormatter.string(fromByteCount:Int64(package.totalBytes),countStyle:.file))").font(.caption.weight(.medium))
                                Text("Input · \(package.inputDtype) [batch, \(package.inputSampleShape.map(String.init).joined(separator:", "))]").font(.system(size:11,design:.monospaced))
                                Text("Output · float32 [batch, \(package.outputSampleShape.map(String.init).joined(separator:", "))]").font(.system(size:11,design:.monospaced))
                                Text("Batch · \(package.configuration.minimumBatchSize)…\(package.configuration.maximumBatchSize) · example \(package.configuration.exampleBatchSize)").font(.caption)
                                Divider().overlay(MLTheme.rule)
                                ForEach(package.files.sorted(by:{$0.path < $1.path}),id:\.path) {file in
                                    HStack {Text(file.path);Spacer();Text(ByteCountFormatter.string(fromByteCount:Int64(file.bytes),countStyle:.file)).foregroundStyle(MLTheme.muted)}.font(.system(size:11,design:.monospaced))
                                }
                                Text("manifest.json").font(.system(size:11,design:.monospaced))
                                Text("from inference import load_model\noutput = load_model().predict(prepared_tensor)").font(.system(size:11,design:.monospaced)).textSelection(.enabled)
                            }
                        }
                    }
                }.padding(.trailing,8)
            }
            Text(export.statusMessage).font(.caption).textSelection(.enabled)
            HStack {Text("Folder").font(.caption);TextField("Package folder",text:$folderName).flatField().disabled(export.isBusy || choosing)}
            if let destinationError {Text(destinationError).font(.caption).foregroundStyle(MLTheme.rose)}
            HStack {
                if export.isBusy {ProgressView().controlSize(.small);Button("Cancel export",action:export.cancel).buttonStyle(FlatButtonStyle())}
                else {Button(export.package == nil ? "Prepare program":"Prepare again") {_=export.prepare()}.buttonStyle(FlatButtonStyle()).disabled(export.hasDraftChanges || choosing)}
                Spacer()
                Button("Choose destination…",action:chooseDestination).buttonStyle(FlatButtonStyle(prominent:true)).disabled(export.isBusy || export.package == nil || export.hasDraftChanges || choosing || !validFolderName)
            }
        }.padding(20).frame(minWidth:660,idealWidth:760,maxHeight:.infinity).background(MLTheme.background)
    }
    private func framed<Content:View>(_ title:String,symbol:String,@ViewBuilder content:()->Content)->some View {
        VStack(alignment:.leading,spacing:0) {
            Label(title,systemImage:symbol).font(.caption.weight(.semibold)).padding(9).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel)
            Divider().overlay(MLTheme.rule)
            content().padding(10).frame(maxWidth:.infinity,alignment:.leading)
        }.overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func choice(_ title:String,key:String,options:[FlatChoiceOption])->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).font(.caption)
            FlatChoiceControl(title:title,selection:export.draft[key]?.stringValue ?? "",options:options) {export.draft[key] = .string($0)}
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
    private func integer(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:4) {
            Text(title).font(.caption)
            TextField(title,text:Binding(get:{export.draft[key]?.integerValue.map {String($0)} ?? export.draft[key]?.stringValue ?? ""},set:{value in
                if let number=Int64(value),String(number) == value {export.draft[key] = .integer(number)}
                else {export.draft[key] = .string(value)}
            })).flatField()
        }.frame(maxWidth:.infinity,alignment:.leading)
    }
    private func boolean(_ key:String)->Binding<Bool> {Binding(get:{export.draft[key]?.boolValue ?? false},set:{export.draft[key] = .bool($0)})}
    private var validFolderName:Bool {(try? WorkflowExportPublication.destination(in:URL(fileURLWithPath:"/"),folderName:folderName)) != nil}
    private func chooseDestination() {
        guard !choosing,!export.isBusy,let package=export.package else {return}
        let reviewed=package.manifestBytes
        choosing=true;destinationError=nil
        let panel = WorkflowExportParentChooser.panel(title: "Choose program export destination", message: "Choose a parent folder. KwipML creates a new program package without replacing existing files.", prompt: "Save here")
        panel.begin {response in
            choosing=false
            guard response == .OK,let parent=panel.url else {return}
            guard export.package?.manifestBytes == reviewed,!export.isBusy,!export.hasDraftChanges else {destinationError="The review changed. Prepare and review the program again.";return}
            do {
                let destination=try WorkflowExportPublication.destination(in:parent,folderName:folderName)
                if !export.save(to:destination,access:parent) {destinationError="The saved model or workspace changed. Prepare again."}
            } catch {destinationError=error.localizedDescription}
        }
    }
}
