import SwiftUI

/// Shared beginner presentation; the captured fact is shown verbatim. Detailed
/// thresholds and archived advice remain in the caller's optional evidence frame.
struct DiagnosticExplanationView: View {
    let guide: DiagnosticAudienceGuide
    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            field("Observed", guide.observedFact)
            field("What it means", guide.meaning)
            field("Try on canvas", guide.action)
            field("Other explanations", guide.alternatives)
            field("Limits", guide.limits)
            Button("Learn how to review this finding") {
                ApplicationNavigationWindows.shared.showLearn(search: DiagnosticAudienceGuide.lessonSearchTerm)
            }.buttonStyle(FlatButtonStyle()).padding(.top, 12)
        }
    }
    private func field(_ title: String, _ value: String) -> some View {
        HStack(alignment: .top, spacing: 16) {
            Text(title).font(.system(size: 11, weight: .semibold)).foregroundStyle(MLTheme.muted)
                .frame(width: 120, alignment: .leading)
            LessonRichText(text: value).font(.system(size: 12)).frame(maxWidth: .infinity, alignment: .leading)
        }.padding(.vertical, 9).overlay(alignment: .bottom) { Rectangle().fill(MLTheme.rule).frame(height: 1) }
    }
}
