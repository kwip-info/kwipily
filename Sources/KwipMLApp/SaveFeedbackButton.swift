import SwiftUI

/// Success is shown only after the owner reports a completed save.
struct SaveFeedbackButton:View {
    let action:()->Bool
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var outcome:Bool?
    @State private var sequence=0
    var body:some View {
        Button {
            let saved=action()
            withAnimation(reduceMotion ? nil:.easeInOut(duration:0.18)) {outcome=saved;sequence+=1}
        } label: {
            HStack(spacing:5) {
                if let outcome {Image(systemName:outcome ? "checkmark":"exclamationmark.circle")}
                Text(outcome.map {$0 ? "Saved":"Not saved"} ?? "Save")
            }.frame(minWidth:62)
                .foregroundStyle(outcome == true ? MLTheme.teal:outcome == false ? Color.orange:Color.primary)
        }.accessibilityLabel(outcome.map {$0 ? "Saved successfully":"Save needs attention"} ?? "Save")
            .task(id:sequence) {
                guard sequence>0 else {return}
                do {try await Task.sleep(for:.seconds(2));withAnimation(reduceMotion ? nil:.easeInOut(duration:0.18)) {outcome=nil}} catch {}
            }
    }
}
