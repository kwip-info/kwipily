import SwiftUI
import KwipMLCore

/// Public editable practice only; there is no assessment pass or completion award.
struct DataEngineeringPracticeView: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    let lessonID: String
    private var lesson: DataEngineeringPractice.Lesson? { try? DataEngineeringPractice.lesson(lessonID) }
    var body: some View {
        if let lesson {
            VStack(alignment: .leading, spacing: 13) {
                Text(lesson.title).font(.system(size: 18, weight: .semibold))
                if !inLessonRail {
                LessonRichText(text: lesson.objective)
                LessonTextPanel(title: "Words to know", text: lesson.terms)
                LessonTextPanel(title: "Predict before running", text: lesson.prediction)
                LessonTextPanel(title: "Try it on the canvas", text: lesson.steps.enumerated().map { "**Step \($0.offset + 1)**\n\($0.element)" }.joined(separator: "\n\n"))
                LessonTextPanel(title: "Compare the actual output", text: lesson.inspect)
                }
                LessonTextPanel(title: "What this tool supports", text: lesson.limits)
                Button("Open editable practice") { _ = store.loadDataEngineeringPracticeExample(lesson.id) }
                    .buttonStyle(FlatButtonStyle()).disabled(store.isBusy)
                    .accessibilityIdentifier("learning.data-engineering." + lesson.id + ".open")
                Text(lesson.disclosure).font(.caption).foregroundStyle(MLTheme.muted)
                Text("Content v\(lesson.contentVersion) · Authored \(lesson.authoredAt) · Guided practice · Results require execution")
                    .font(.caption2).foregroundStyle(MLTheme.muted)
            }.accessibilityIdentifier("learning.data-engineering." + lesson.id)
        } else {
            Text("This practice material is unavailable. Reinstall the complete application to restore its bundled resources.")
                .font(.caption).foregroundStyle(MLTheme.rose)
        }
    }
}

struct DataEngineeringSharedGuidanceView: View {
    var body: some View {
        if let catalog = try? DataEngineeringPractice.catalog() {
            ForEach(catalog.sharedGuidance) { topic in
                VStack(alignment: .leading, spacing: 12) {
                    Text(topic.title).font(.system(size: 16, weight: .semibold))
                    LessonRichText(text: topic.objective)
                    LessonTextPanel(title: "Words to know", text: topic.terms)
                    LessonTextPanel(title: "Try it on the canvas", text: topic.steps.enumerated().map { "**Step \($0.offset + 1)**\n\($0.element)" }.joined(separator: "\n\n"))
                    LessonTextPanel(title: "Read the evidence", text: topic.inspect)
                    Text("Guided reading and practice · No completion award").font(.caption2).foregroundStyle(MLTheme.muted)
                }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
                    .background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                    .accessibilityIdentifier("learning.data-engineering.guidance." + topic.id)
            }
        }
    }
}
