import SwiftUI
import KwipMLCore

/// All row-valued MoE evidence uses the ordinary bounded query/view surface.
struct MixtureTrainingOutputView:View {
    @ObservedObject var host:WorkflowWorkspaceSession
    let address:WorkflowNodeAddress
    @State private var selected="predictions"
    var body:some View {
        VStack(alignment:.leading,spacing:8){
            FlatChoiceControl(title:"Output",selection:selected,options:[.init(id:"predictions",title:"Predictions"),.init(id:"routing",title:"Routing summary"),.init(id:"routingSample",title:"Routing sample")]){selected=$0}
            if let query=host.predictionTableViewSession(for:.init(address:address,port:selected)){TableViewOutputView(session:query,executionBusy:host.owner.isBusy)}
            else {Text(selected=="predictions" ? "No current prediction table. Run Train to publish verified outputs.":"Not captured. Enable the corresponding routing capture and History diagnostic retention before running.").font(.caption).foregroundStyle(MLTheme.muted);Spacer()}
        }
    }
}
