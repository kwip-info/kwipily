import SwiftUI
import KwipMLCore

/// The family guide links to original assessed lessons; no new progress identity.
struct LinearOverviewLearningView:View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    var body:some View {
        VStack(alignment:.leading,spacing:12) {
            if !inLessonRail { Text("Linear").font(.title3.weight(.semibold)) }
            if !inLessonRail { Text("Place Linear from Layers, then choose Input sizing in Configuration. Both modes learn a final-dimension affine projection.").font(.caption).foregroundStyle(MLTheme.muted) }
            row("Set explicitly",detail:"Choose both input and output widths. Explore weights, bias and affine composition.",lesson:"affine")
            row("Infer from input",detail:"Resolve the connected final width before training; choose output width and bias.",lesson:"lazy-linear-lesson")
            if !inLessonRail { Text("Original lessons and assessments retain their progress. Reading this guide does not award completion. Width or sizing changes require compatible fresh state; saved weights never resize automatically.").font(.caption).foregroundStyle(MLTheme.muted) }
        }
    }
    private func row(_ title:String,detail:String,lesson:String)->some View {
        HStack(alignment:.top,spacing:12) {
            VStack(alignment:.leading,spacing:3) {Text(title).font(.system(size:13,weight:.semibold));Text(detail).font(.caption).foregroundStyle(MLTheme.muted)}
            Spacer()
            Button("Read lesson") {ApplicationNavigationWindows.shared.showLearn(entryID:lesson)}.buttonStyle(.link).accessibilityLabel("Read \(title) Linear lesson")
        }
    }
}
