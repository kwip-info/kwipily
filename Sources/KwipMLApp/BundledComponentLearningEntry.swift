import SwiftUI
import KwipMLCore

/// Bundle-authored explanations share presentation and example launch behavior.
struct BundledComponentLearningEntry: View {
    @Environment(\.lessonRailPresented) private var inLessonRail
    @ObservedObject var store: WorkspaceStore
    let componentID: String
    let learning: any BundledComponentLearningContent
    var body: some View {
        HStack(alignment: .top, spacing: 24) {
            if !inLessonRail || !BundledExampleLessonCopy.isReviewed(componentID) {
            VStack(alignment: .leading, spacing: 8) {
                Text(learning.title).font(.title3.weight(.semibold))
                Text(learning.subtitle).font(.system(size: 13)).foregroundStyle(MLTheme.muted)
                if RecurrentCellConfiguration.componentIDs.contains(componentID) {
                    Text("Place Recurrent Cell from Layers and choose "+componentID.replacingOccurrences(of:"_cell",with:"").uppercased()+" Cell in Method. Configure one step, its feature widths and starting state; Weights and biases owns named parameter policies.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                if RecurrentSequenceConfiguration.componentIDs.contains(componentID) {
                    Text("Place Recurrent from Layers and choose "+componentID.uppercased()+" in Method. Configuration contains widths, layers, starting state and applicable method controls; Weights and biases owns named parameter policies.").font(.caption).foregroundStyle(MLTheme.muted)
                }
                InspectorSection(learning.disclosureTitle, symbol: "info.circle", identifier: "learn.component.\(componentID)", initiallyExpanded: false) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(learning.formula).font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                        ForEach(Array(learning.paragraphs.enumerated()), id: \.offset) { _, paragraph in
                            Text(paragraph)
                        }
                        Text("Content v\(learning.contentVersion) · Updated \(learning.updatedAt)")
                            .font(.caption2)
                    }.font(.caption).foregroundStyle(MLTheme.muted).padding(.top, 6)
                }
            }
            }
            Spacer()
            Button(learning.buttonTitle) { store.loadComponentWorkflowExample(componentID) }.disabled(store.isBusy)
        }
    }
}
