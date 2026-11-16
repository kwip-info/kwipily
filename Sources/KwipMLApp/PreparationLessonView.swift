import SwiftUI

/// Small shared reading surface for bundled conceptual lessons; execution remains in the canvas.
struct PreparationLessonView: View {
    let text: String
    let close: () -> Void
    private var paragraphs: [String] { text.components(separatedBy: "\n\n").filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } }
    var body: some View {
        VStack(spacing: 0) {
            HStack {
                SectionLabel(title: "Preparation concepts")
                Spacer()
                Button(action: close) { Image(systemName: "xmark") }.accessibilityLabel("Close preparation lesson")
            }.padding(18)
            Divider()
            ThemedScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    ForEach(Array(paragraphs.enumerated()), id: \.offset) { _, paragraph in
                        if paragraph.hasPrefix("# ") {
                            Text(String(paragraph.dropFirst(2))).font(.system(size: 24, weight: .semibold))
                        } else if paragraph.hasPrefix("## ") {
                            Text(String(paragraph.dropFirst(3))).font(.system(size: 17, weight: .semibold)).foregroundStyle(MLTheme.teal)
                        } else {
                            Text((try? AttributedString(markdown: paragraph, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(paragraph))
                                .font(.system(size: 13)).lineSpacing(5)
                        }
                    }
                }.textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(24)
            }
        }.frame(width: 620, height: 600).background(MLTheme.panel).foregroundStyle(.primary)
    }
}
