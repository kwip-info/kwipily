import SwiftUI
import KwipMLCore

/// Saved-run exercises use the same controls as Train/History, without fake forward ports.
struct CheckpointLearningWorkbenchView<Results:View>:View {
    @ObservedObject var lesson:TensorLearningStore
    let showInspector:Bool
    let task:CheckpointLearningTask
    let results:Results
    var showsInstructions=true
    @State private var resultsExpanded=true
    var body:some View {
        HSplitView {
            if showInspector && showsInstructions {
            ThemedScrollView {
                VStack(alignment:.leading,spacing:14) {
                    LessonTextPanel(title:"What you are learning",text:task.explanation)
                    InspectorSection("Your task",symbol:"checklist",identifier:"lesson.checkpoint.task") {
                        LessonRichText(text:task.controlInstructions)
                        Text("Choose Check construction after applying your controls. The checker supplies a small model, varied rows and a saved run.")
                            .font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    InspectorInformationSection {
                        LessonRichText(text:"**Weights** are the settings a model learns. A **checkpoint** stores those settings and the memory needed to continue training.\n\n**Fine-tune weights** starts a new job using the saved settings. **Exact resume** continues the previous job, including its random choices and adjustment memory.\n\nAn **epoch** is a complete pass through the training rows. A **batch** is the group of rows handled together. During exact resume, Target total epochs includes passes already completed.")
                    }
                }.padding(14)
            }.frame(minWidth:250,idealWidth:310,maxWidth:420).background(MLTheme.panel)
            }
            VSplitView {
                ThemedScrollView {
                    VStack(alignment:.leading,spacing:14) {
                        HStack(spacing:12) {
                            if let image=ComponentArtworkCatalog.checkpointLifecycleImage {
                                Image(nsImage:image).resizable().scaledToFit().frame(width:48,height:48).accessibilityHidden(true)
                            }
                            VStack(alignment:.leading,spacing:4) {
                                Text("Reuse a saved run").font(.headline)
                                Text("Train controls · supplied lesson checkpoint").font(.caption).foregroundStyle(MLTheme.muted)
                            }
                        }
                        InspectorConfigurationSection(isEnabled: !lesson.isBusy) {
                            VStack(alignment:.leading,spacing:12) {
                                choice("Training behavior",key:"trainingMode",options:[.init(id:"fresh",title:"Fresh training"),.init(id:"fineTune",title:"Fine-tune weights"),.init(id:"exactResume",title:"Exact resume")])
                                choice("Parent checkpoint",key:"parent",options:[.init(id:"none",title:"No saved parent"),.init(id:"savedLessonRun",title:"Saved lesson run")])
                                number(lesson.checkpointField("trainingMode") == "exactResume" ? "Target total epochs":"Epochs",key:"epochs")
                                number("Batch size",key:"batchSize")
                                HStack(spacing:10) {
                                    Button("Apply controls") {_ = lesson.applyCheckpointFields()}.buttonStyle(FlatButtonStyle())
                                    Button("Discard changes") {_ = lesson.discardCheckpointFields()}.buttonStyle(FlatButtonStyle())
                                }.disabled(lesson.draft?.pendingFields.isEmpty != false)
                                Text("Changes stay in this lesson. Apply or discard them before checking; closing a section preserves your draft.")
                                    .font(.caption).foregroundStyle(MLTheme.muted)
                            }
                        }
                        InspectorSection("Supplied training context",symbol:"tablecells",identifier:"lesson.checkpoint.context") {
                            Grid(alignment:.leading,horizontalSpacing:24,verticalSpacing:10) {
                                GridRow {Text("Training rows");Text("8 · two input values and two desired values per row")}
                                GridRow {Text("Validation rows");Text("4 · kept separate to check the result")}
                                GridRow {Text("Saved parent");Text("Two completed passes; checker-owned checkpoint")}
                                GridRow {Text("Device");Text("CPU · exact continuation compares every saved state")}
                            }.font(.caption)
                            Text("Each check uses four varied datasets. Saved settings and measured results appear below after execution; no code or file paths are required.")
                                .font(.caption).foregroundStyle(MLTheme.muted)
                        }
                    }.padding(14)
                }.frame(minHeight:220)
                VStack(spacing:0) {
                    HStack {
                        Button {resultsExpanded.toggle()} label:{Label("Results",systemImage:resultsExpanded ? "chevron.down":"chevron.up")}.buttonStyle(.plain)
                        Spacer()
                        Text("Measured saved-run behavior").font(.caption).foregroundStyle(MLTheme.muted)
                    }.padding(.horizontal,12).frame(height:34)
                    if resultsExpanded {Divider();ThemedScrollView {results.frame(maxWidth:.infinity,alignment:.leading).padding(12)}}
                }.background(MLTheme.panel).frame(minHeight:resultsExpanded ? 150:34,idealHeight:resultsExpanded ? 260:34,maxHeight:resultsExpanded ? .infinity:34)
                    .accessibilityIdentifier("checkpoint.lesson.results.bottom")
            }.frame(minWidth:420)
        }.background(MLTheme.background)
    }
    private func choice(_ title:String,key:String,options:[FlatChoiceOption])->some View {
        VStack(alignment:.leading,spacing:5) {
            Text(title).font(.caption)
            FlatChoiceControl(title:title,selection:lesson.checkpointField(key),options:options) {value in _ = lesson.setCheckpointField(key,text:value)}
        }
    }
    private func number(_ title:String,key:String)->some View {
        VStack(alignment:.leading,spacing:5) {
            Text(title+" · 1–8").font(.caption)
            TextField(title,text:Binding(get:{lesson.checkpointField(key)},set:{_ = lesson.setCheckpointField(key,text:$0)}))
                .flatField().accessibilityLabel(title)
        }
    }
}
