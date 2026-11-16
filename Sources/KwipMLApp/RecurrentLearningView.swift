import SwiftUI
import KwipMLCore

struct RecurrentOverviewLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail { Text("Recurrent").font(.title3.weight(.semibold)) }
            if !inLessonRail { Text("Place Recurrent from Layers, then select RNN, LSTM or GRU in Method. These readers process a whole ordered sequence; Recurrent Cell is the separate single-step tool.").font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(RecurrentSequenceConfiguration.Method.allCases,id:\.rawValue) {method in
                HStack(alignment:.top,spacing:12) {
                    VStack(alignment:.leading,spacing:3) {
                        Text(method.title).font(.system(size:13,weight:.semibold))
                        Text(summary(method)).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    Button("Read example") {ApplicationNavigationWindows.shared.showLearn(entryID:method.rawValue)}.buttonStyle(.link)
                        .accessibilityLabel("Read \(method.title) example")
                }
            }
            Button("Sequences and explicit memory lab") {ApplicationNavigationWindows.shared.showLearn(entryID:"recurrent-memory")}.buttonStyle(.link)
            if !inLessonRail { Text("Original assessed exercises retain their progress. Reading this guide does not award completion. Hidden/cell state is explicit tensor data; it is separate from learned checkpoint weights.").font(.caption).foregroundStyle(MLTheme.muted) }
        }
    }
    private func summary(_ method:RecurrentSequenceConfiguration.Method)->String {
        switch method {case .rnn:return "Update a hidden summary with Tanh or ReLU.";case .lstm:return "Keep separate hidden and cell memories, with optional hidden projection.";case .gru:return "Use gates to update one hidden summary."}
    }
}

struct RecurrentCellOverviewLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail { Text("Recurrent Cell").font(.title3.weight(.semibold)) }
            if !inLessonRail { Text("Place Recurrent Cell from Layers, then select RNN Cell, LSTM Cell or GRU Cell in Method. Each call reads one feature vector per observation; use Recurrent for an entire ordered sequence.").font(.caption).foregroundStyle(MLTheme.muted) }
            ForEach(RecurrentCellConfiguration.Method.allCases,id:\.rawValue) {method in
                HStack(alignment:.top,spacing:12) {
                    VStack(alignment:.leading,spacing:3) {
                        Text(method.title+" Cell").font(.system(size:13,weight:.semibold))
                        Text(summary(method)).font(.caption).foregroundStyle(MLTheme.muted)
                    }
                    Spacer()
                    Button("Read example") {ApplicationNavigationWindows.shared.showLearn(entryID:method.rawValue+"_cell")}.buttonStyle(.link)
                        .accessibilityLabel("Read \(method.title) Cell example")
                }
            }
            Button("Sequences and explicit memory lab") {ApplicationNavigationWindows.shared.showLearn(entryID:"recurrent-memory")}.buttonStyle(.link)
            if !inLessonRail { Text("Original exercises retain their progress. Reading does not award completion. State travels on tensor connections; the cell never secretly remembers an earlier call. Output and hidden are the same step result; LSTM also exposes distinct cell memory.").font(.caption).foregroundStyle(MLTheme.muted) }
        }
    }
    private func summary(_ method:RecurrentCellConfiguration.Method)->String {
        switch method {case .rnn:return "Update hidden state with Tanh or ReLU.";case .lstm:return "Update separate hidden and cell memories.";case .gru:return "Use gates to update one hidden state."}
    }
}
