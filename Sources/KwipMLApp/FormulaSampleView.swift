import SwiftUI
import KwipMLCore

/// Compact draft evidence; no full-population navigation or source preview authority.
struct FormulaSampleView:View {
    @ObservedObject var session:TableViewQuerySession
    @State private var selectedRow:Int?
    var body:some View {
        VStack(alignment:.leading,spacing:6) {
            if let page=session.page {
                let columns=page.columns.enumerated().map {i,c in CapturedDataGrid.Column(id:i,name:c["name"]?.stringValue ?? "",label:c["name"]?.stringValue ?? "",dtype:c["dtype"]?.stringValue ?? "",nullability:c["nullable"] == .bool(true) ? "nullable":"required")}
                let rows=zip(page.positions,page.rows).map {position,values in CapturedDataGrid.Row(index:position,cells:values.map{CapturedDataGrid.cell(WorkflowCapturedDataOutput.render($0))})}
                CapturedNativeTable(columns:columns,rows:rows,selectedRow:$selectedRow,rowHeaderTitle:"Sample",imageResolver:session.imageResolver)
            } else if session.isBusy {ProgressView().frame(maxWidth:.infinity)}
            if !session.statusMessage.isEmpty {Text(session.statusMessage).font(.caption).foregroundStyle(MLTheme.muted).textSelection(.enabled)}
        }
    }
}
