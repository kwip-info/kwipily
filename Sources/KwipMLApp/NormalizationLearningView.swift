import SwiftUI
import KwipMLCore

/// The draft editor shares preparation controls, while the lesson owns sources and output meaning.
struct NormalizationLearningView: View {
    @ObservedObject var lesson: NormalizationLearningStore
    @ObservedObject var workspace: WorkspaceStore

    var body: some View {
        ThemedScrollView {
            VStack(alignment:.leading,spacing:16) {
                LessonTextPanel(title:"Training-only preparation moved to Preprocess",text:"Your saved normalization draft and assessment progress are retained as historical evidence. Current workflows connect Partition’s Training, Validation and Test outputs to one Preprocess. It learns replacements, categories and scaling from training only, then applies the same state to the separate evaluation populations. This replacement practice is unscored.")
                PreprocessLearningView(store:workspace)
            }.padding(16)
        }
    }
}

/// Reading navigation only; original mathematical tasks retain assessment ownership.
enum NormalizationLearningContent {
    static let lessonID="normalization-overview"
    static let defaultComponentID="layer_norm"
    static let placementInstructions="Place Normalization, then choose Configuration → Method and the operation's supported dimensions. Choose which values share a summary before adjusting scale and bias."
    static let assessmentDisclosure="The original mathematical lessons and assessments remain separate. Reading this overview does not award completion. Model normalization is distinct from Preprocess's training-set preparation."
    struct Entry:Identifiable {
        let componentID:String
        let title:String
        let lessonID:String
        let summary:String
        var id:String {componentID}
    }
    static let entries:[Entry]=[
        .init(componentID:"rms_norm",title:"RMS",lessonID:"rms_norm",summary:"Scale trailing dimensions by root mean square without subtracting their mean; optionally learn scale."),
        .init(componentID:"layer_norm",title:"Layer",lessonID:"layernorm",summary:"Normalize ordered trailing dimensions independently within each observation."),
        .init(componentID:"batch_norm1d",title:"Batch · 1D",lessonID:"channel-normalization",summary:"Compare each channel across the current group, with optional remembered summaries."),
        .init(componentID:"batch_norm2d",title:"Batch · 2D",lessonID:"channel-normalization",summary:"Compare channels across a group of maps and their spatial positions."),
        .init(componentID:"instance_norm1d",title:"Instance · 1D",lessonID:"channel-normalization",summary:"Normalize each observation's channel pattern separately."),
        .init(componentID:"instance_norm2d",title:"Instance · 2D",lessonID:"channel-normalization",summary:"Normalize each observation's channel map separately."),
        .init(componentID:"group_norm",title:"Group",lessonID:"channel-normalization",summary:"Normalize teams of channels within each observation.")
    ]
}

struct NormalizationOverviewLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail { Text("Normalization").font(.title3.weight(.semibold)) }
            if !inLessonRail { Text(NormalizationLearningContent.placementInstructions).font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(NormalizationLearningContent.entries) {entry in
                HStack(alignment:.top,spacing:12) {
                    VStack(alignment:.leading,spacing:3) {
                        Text(entry.title).font(.system(size:13,weight:.semibold))
                        Text(entry.summary).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    Button("Read lesson") {ApplicationNavigationWindows.shared.showLearn(entryID:entry.lessonID)}
                        .buttonStyle(.link)
                        .accessibilityLabel("Read \(entry.title) normalization lesson")
                        .accessibilityIdentifier("normalization.lesson."+entry.componentID)
                }
            }
            if !inLessonRail { Text(NormalizationLearningContent.assessmentDisclosure).font(.caption).foregroundStyle(MLTheme.muted) }
        }
    }
}
