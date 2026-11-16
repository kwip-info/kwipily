import SwiftUI
import KwipMLCore

/// Guided placement remains an explicit edit, independent of reading navigation.
struct WorkflowCourseCoachView:View {
    @Environment(\.lessonTypography) private var typography
    @ObservedObject var owner:WorkflowAuthoringStore
    @ObservedObject var host:WorkflowWorkspaceSession
    var body:some View {
        if let progress=owner.document.guidedBuild,
           let plan=try? WorkflowGuidedBuildPlan.parallelClassifier(projectID:owner.projectID,version:progress.version) {
            VStack(alignment:.leading,spacing:10) {
                if progress.courseID != plan.id || progress.version != plan.version || !(0...plan.steps.count).contains(progress.completedSteps) {
                    Text("This course version is unsupported. Your canvas remains editable; placement is unavailable.")
                } else {
                    Text("\(progress.completedSteps) of \(plan.steps.count) components placed").font(.system(size:typography.size(11))).foregroundStyle(MLTheme.muted)
                    if progress.completedSteps < plan.steps.count {
                        Button("Place next: \(ClassifierLessonGuide.title(for:plan.steps[progress.completedSteps].address.localID,fallback:plan.steps[progress.completedSteps].title,version:progress.version))") {_ = host.advanceGuidedCourse(expected:progress)}
                            .disabled(host.canvas.isBusy || owner.hasPendingInspectorEdits)
                            .accessibilityIdentifier("lesson.guided.place-next")
                    } else {
                        Button("Run this workflow") {_ = host.run()}.disabled(host.canvas.isBusy || owner.hasPendingInspectorEdits)
                        Text("Placement complete · not an assessment").font(.system(size:typography.size(10))).foregroundStyle(MLTheme.muted)
                    }
                }
            }.frame(maxWidth:.infinity,alignment:.leading)
        }
    }
}
