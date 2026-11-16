import SwiftUI
import KwipMLCore

/// Only a verified formative projection enters this view; private assessment has none.
struct CheckpointMeasuredResultsView:View {
    let projection:CheckpointFormativeProjection
    @State private var fixtureID=""
    private var fixture:CheckpointFormativeFixture? {projection.fixtures.first {$0.id == fixtureID} ?? projection.fixtures.first}
    private func title(_ key:String)->String {
        switch key {
        case "savedParameters":return "Saved weights and bias · one row per result"
        case "updatedParameters":return "Weights and bias after new training"
        case "resumedParameters":return "Weights and bias after continuation"
        case "restoredPredictions":return "Reopened model predictions · validation rows"
        default:return CheckpointLessonReference.label(key)
        }
    }
    var body:some View {
        if let fixture {
            InspectorSection("Measured checkpoint results",symbol:"tablecells",identifier:"lesson.checkpoint.results") {
                FlatChoiceControl(title:"Dataset",selection:fixture.id,options:projection.fixtures.enumerated().map {.init(id:$0.element.id,title:"Dataset \($0.offset+1)")}) {fixtureID=$0}
                Text("Differences compare independently obtained results. Zero means they match. Pass counts also verify that continuation starts where the saved run stopped.")
                    .font(.caption).foregroundStyle(MLTheme.muted)
                LazyVGrid(columns:[GridItem(.adaptive(minimum:260),alignment:.leading)],alignment:.leading,spacing:12) {
                    ForEach(fixture.expected.keys.sorted(),id:\.self) {key in
                        VStack(alignment:.leading,spacing:10) {
                            if let measured=fixture.measured[key] {LearningTensorTable(tensor:measured,title:title(key))}
                            if let expected=fixture.expected[key] {LearningTensorTable(tensor:expected,title:"Expected comparison")}
                        }.padding(10).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
                    }
                }
            }
            InspectorSection("Saved settings and actual results",symbol:"square.stack.3d.up",identifier:"lesson.checkpoint.values") {
                LazyVGrid(columns:[GridItem(.adaptive(minimum:260),alignment:.leading)],alignment:.leading,spacing:12) {
                    ForEach(fixture.measured.keys.filter {fixture.expected[$0] == nil}.sorted(),id:\.self) {key in
                        if let values=fixture.measured[key] {
                            LearningTensorTable(tensor:values,title:title(key)).padding(10).background(MLTheme.background).overlay(Rectangle().stroke(MLTheme.rule))
                        }
                    }
                }
            }
        }
    }
}
