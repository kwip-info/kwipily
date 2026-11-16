import SwiftUI
import KwipMLCore

/// Discovery and reading only; scored function tasks retain their original identities.
enum ActivationLearningContent {
    static let lessonID="activation"
    static let defaultFunction=ActivationConfiguration.Function.relu
    static let placementInstructions="Place Activation, then select the block and choose Configuration → Function. A restricted lesson offers only the functions permitted by that lesson."
    static let assessmentDisclosure="Each function keeps its original lessons and assessment. Reading this overview does not award completion."
    static let dropoutDisclosure="Dropout remains a separate tool: its behavior changes between training and evaluation."
    struct Entry:Identifiable {
        let function:ActivationConfiguration.Function
        let lessonID:String
        let summary:String
        var id:String {function.rawValue}
    }
    static let entries:[Entry]=[
        .init(function:.relu,lessonID:"relu-lesson",summary:"Keep positives and replace negatives with zero."),
        .init(function:.leakyReLU,lessonID:"leaky_relu",summary:"Keep a chosen fraction of negative values."),
        .init(function:.prelu,lessonID:"prelu",summary:"Learn a shared or per-channel negative multiplier."),
        .init(function:.elu,lessonID:"elu",summary:"Use a smooth negative branch that approaches a limit."),
        .init(function:.gelu,lessonID:"gelu",summary:"Compare an exact Gaussian gate with its tanh approximation."),
        .init(function:.silu,lessonID:"silu",summary:"Multiply each value by its sigmoid response."),
        .init(function:.sigmoid,lessonID:"sigmoid-lesson",summary:"Transform each value into the range zero to one."),
        .init(function:.tanh,lessonID:"tanh-lesson",summary:"Preserve signs within the range minus one to one."),
        .init(function:.softmax,lessonID:"softmax",summary:"Compare a chosen group of scores as shares that sum to one."),
        .init(function:.logSoftmax,lessonID:"log_softmax",summary:"Store relative score shares on a logarithmic scale.")
    ]
}

struct ActivationLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail { Text("Activation").font(.title3.weight(.semibold)) }
            if !inLessonRail { Text(ActivationLearningContent.placementInstructions).font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(ActivationLearningContent.entries) {entry in
                HStack(alignment:.top,spacing:12) {
                    VStack(alignment:.leading,spacing:3) {
                        Text(entry.function.title).font(.system(size:13,weight:.semibold))
                        Text(entry.summary).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    Button("Read lesson") {ApplicationNavigationWindows.shared.showLearn(entryID:entry.lessonID)}
                        .buttonStyle(.link)
                        .accessibilityLabel("Read \(entry.function.title) lesson")
                        .accessibilityIdentifier("activation.lesson."+entry.function.rawValue)
                }
            }
            if !inLessonRail { Text(ActivationLearningContent.assessmentDisclosure).font(.caption).foregroundStyle(MLTheme.muted) }
            HStack {
                Text(ActivationLearningContent.dropoutDisclosure).font(.caption).foregroundStyle(MLTheme.muted)
                Button("Read Dropout lesson") {ApplicationNavigationWindows.shared.showLearn(entryID:"dropout")}.buttonStyle(.link)
            }
        }
    }
}
