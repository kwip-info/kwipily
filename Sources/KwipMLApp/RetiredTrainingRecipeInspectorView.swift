import SwiftUI
import KwipMLCore

/// Recovery evidence only; these settings have no independent authoring authority.
struct RetiredTrainingRecipeInspectorView: View {
    let node: WorkflowNode
    private var configuration: String {
        guard let data=try? JSONEncoder().encode(node),let text=String(data:data,encoding:.utf8) else {return "Original configuration is retained in the project."}
        return text
    }
    var body: some View {
        VStack(alignment:.leading,spacing:12) {
            Label("Project repair required",systemImage:"exclamationmark.triangle").foregroundStyle(MLTheme.rose)
            Text("These training settings now belong to Train or the relevant layer. An exact automatic upgrade was unavailable, so the original tool and its connections have been preserved.")
            Text("Review the retained settings, apply the intended values to Train or the layer, then remove this retired tool. Running and Code export remain unavailable until repair is complete.").font(.caption).foregroundStyle(MLTheme.muted)
            DisclosureGroup("Original configuration") {
                Text(configuration).font(.system(.caption,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)
            }
        }
    }
}
