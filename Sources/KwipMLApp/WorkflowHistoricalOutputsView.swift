import SwiftUI
import KwipMLCore

/// Uses the archive's verified complete table capability, never a current output
/// or mutable source. Older archives retain explicitly sampled preview behavior.
struct WorkflowHistoricalOutputsView:View {
    let verified:VerifiedWorkflowOutcome
    @ObservedObject var history:WorkflowHistoryStore
    @ObservedObject var workspace:WorkspaceStore
    @State private var selected="predictions"
    @State private var table:TableViewQuerySession?
    private struct TableSelection:Hashable {let reference:WorkflowHistoryReference;let port:String}
    private var content:WorkflowVerifiedModelContent? {verified.modelContent}
    private var tensors:[String:JSONValue] {if case .object(let values)=content?.activationPreviews {return values};return [:]}
    private var choices:[FlatChoiceOption] {
        var values:[FlatChoiceOption]=[]
        if content?.predictionTable != nil || content?.predictionPreview != nil || content?.objectivePredictionPreview != nil || content?.bagPredictionPreview != nil {values.append(.init(id:"predictions",title:"Predictions"))}
        values += (content?.routingTables.keys.sorted() ?? []).map{.init(id:$0,title:$0=="routing" ? "Routing summary":"Routing sample")}
        values += tensors.keys.sorted().map {key in
            let node=verified.plan.model.graph.nodes.first {$0.id==key}
            let authored=node?.name ?? ""
            let name=authored.isEmpty ? node?.type.capitalized ?? "Tensor":authored
            return .init(id:key,title:name)
        };return values
    }
    private var effective:String {choices.contains {$0.id==selected} ? selected:choices.first?.id ?? ""}
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            if choices.isEmpty {Text("No component output captured").font(.caption).foregroundStyle(MLTheme.muted);Spacer()}
            else {
                FlatChoiceControl(title:"Output",selection:effective,options:choices) {selected=$0}
                if content?.routingTables[effective] != nil {
                    if let table {TableViewOutputView(session:table).frame(maxWidth:.infinity,maxHeight:.infinity)}else{Text("Captured routing table unavailable. Reopen the verified record.").font(.caption);Spacer()}
                } else if effective=="predictions" {
                    if let table {TableViewOutputView(session:table).frame(maxWidth:.infinity,maxHeight:.infinity)}
                    else if content?.predictionTable != nil {Text("Captured table unavailable. Check the local runtime and reopen this record.").font(.caption).foregroundStyle(MLTheme.rose);Spacer()}
                    else {ThemedScrollView {
                        if let preview=content?.bagPredictionPreview {BagResultSectionsView(preview:preview,evaluation:nil,provenance:"Archived sample")}
                        else if let preview=content?.objectivePredictionPreview,let evaluation=content?.objectiveEvaluation {ObjectiveResultSectionsView(preview:preview,evaluation:evaluation,provenance:"Archived sample")}
                        else if let preview=content?.predictionPreview {PredictionPreviewTable(preview:preview,provenance:"Archived sample")}
                    }}
                } else if case .object(let tensor)=tensors[effective] {
                    ThemedScrollView {TensorOutputPreviewView(preview:tensor.mapValues(WorkflowCapturedDataOutput.render)).frame(maxWidth:.infinity,alignment:.leading)}
                }
            }
        }.task(id:TableSelection(reference:verified.reference,port:effective)) {
            table?.invalidate();table=nil
            if let evidence=effective=="predictions" ? content?.predictionTable:content?.routingTables[effective] {
                let reference=verified.reference
                let root=workspace.workspaceDirectory.appendingPathComponent("workflow-runs/"+reference.projectID+"/"+reference.directoryID.uuidString)
                table=workspace.makeTableViewQuerySession(artifact:evidence.artifact(directory:root),projectID:reference.projectID,accepts:{history.isActive && history.selectedReference==reference})
            }
        }.onDisappear {table?.invalidate();table=nil}
    }
}
