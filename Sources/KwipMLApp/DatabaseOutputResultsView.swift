import SwiftUI
import KwipMLCore

struct DatabaseOutputReceiptView:View {
    let record:DatabaseOutputLedger.Record
    var acknowledge:(()->Void)?=nil
    private var status:String {record.receipt["status"]?.stringValue ?? "unknown"}
    private var unknown:Bool {status=="intent" || status=="unknown"}
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack {Image(systemName:unknown ? "exclamationmark.triangle":status=="committed" ? "checkmark.circle":"xmark.circle");Text(title).fontWeight(.semibold);Spacer();Text(record.createdAt.formatted(date:.abbreviated,time:.standard)).font(.caption2)}.foregroundStyle(unknown ? Color.orange:status=="committed" ? ComponentVisualFamily.data.primary:MLTheme.rose)
            VStack(spacing:0) {
                row("Destination",(record.configuration["schema"]?.stringValue ?? "")+"."+(record.configuration["table"]?.stringValue ?? ""))
                row("Engine",record.configuration["backend"]?.stringValue ?? "")
                row("Mode",record.configuration["mode"]?.stringValue ?? "")
                if let count=record.receipt["inputRows"]?.integerValue {row("Input rows",String(count))}
                row("Operation",record.id)
                if let created=record.receipt["createdTable"],created == .bool(true) {row("Schema","New table created"+(record.configuration["backend"] == .string("mysql") ? " · separate DDL change":""))}
            }.overlay(Rectangle().stroke(MLTheme.rule))
            if let message=record.receipt["message"]?.stringValue {Text(message).foregroundStyle(unknown ? Color.orange:MLTheme.rose).fixedSize(horizontal:false,vertical:true)}
            if unknown {
                Text("Connection lost or no final acknowledgment. Check the destination before retrying. This receipt does not prove whether rows committed.").foregroundStyle(MLTheme.muted).fixedSize(horizontal:false,vertical:true)
                if record.acknowledged {Text("Acknowledged · a future Run Flow is a new write.").foregroundStyle(MLTheme.muted)}
                else if let acknowledge {Button("I checked the destination · allow a new write",action:acknowledge).buttonStyle(FlatButtonStyle())}
            }
            Text("Recorded write outcome. Viewing this receipt never connects to the database or repeats the write.").font(.caption2).foregroundStyle(MLTheme.muted)
        }.font(.caption)
    }
    private var title:String {switch status {case "committed":"Committed";case "rolledBack":"Rolled back · no input rows committed";case "notStarted":"Not started";default:"Outcome unknown"}}
    private func row(_ label:String,_ value:String)->some View {HStack(alignment:.top,spacing:10) {Text(label).foregroundStyle(MLTheme.muted).frame(width:95,alignment:.leading);Text(value).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading)}.padding(8).background(MLTheme.background).overlay(alignment:.bottom){Rectangle().fill(MLTheme.rule).frame(height:1)}}
}
struct DatabaseOutputResultsView<Context:DataNodeEditing>:View {
    @ObservedObject var context:Context
    @ObservedObject var store:DatabaseOutputStore
    var executionBusy:Bool
    @State private var tab="receipt"
    @State private var error:String?
    private var address:WorkflowNodeAddress {.init(domain:.data,localID:context.node?.id ?? "")}
    private var unresolved:[DatabaseOutputLedger.Record] {store.records.filter {!$0.acknowledged && ["intent","unknown"].contains($0.receipt["status"]?.stringValue ?? "")}}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {Button(APITableSettings.isAPI(context.node?.config ?? [:]) ? "Delivery receipt":"Write receipt") {tab="receipt"}.buttonStyle(FlatButtonStyle(prominent:tab=="receipt"));if APITableSettings.isAPI(context.node?.config ?? [:]) {Button("Responses") {tab="responses"}.buttonStyle(FlatButtonStyle(prominent:tab=="responses"))};Button("Input table") {tab="input"}.buttonStyle(FlatButtonStyle(prominent:tab=="input"));Spacer()}
            if tab=="responses" {
                if let record=store.latest(for:address) {APIDeliveryResponsesView(record:record)}else {Text("Run Flow to capture API response metadata.").foregroundStyle(MLTheme.muted)}
            } else if tab=="input" {
                if let session=context.capturedTableViewSession(port:"table") {TableViewOutputView(session:session,executionBusy:executionBusy).id(ObjectIdentifier(session))}
                else {Text("Run Flow to capture the complete prepared input table.").foregroundStyle(MLTheme.muted)}
            } else {
                DatabaseReceiptViewport {
                if let latest=store.latest(for:address) {receipt(latest)}
                else {Text("No write recorded. Apply configuration, then Run Flow to send the prepared table.").foregroundStyle(MLTheme.muted)}
                ForEach(unresolved.filter {$0.id != store.latest(for:address)?.id}) {r in
                    Divider();Text("Previous write needing attention").fontWeight(.semibold);receipt(r)
                }
                }
            }
            if let error {Text(error).foregroundStyle(MLTheme.rose)}
        }.frame(minWidth:0,maxWidth:.infinity,minHeight:0,maxHeight:.infinity,alignment:.topLeading).font(.caption).onAppear {store.reloadRecords()}
    }
    @ViewBuilder private func receipt(_ r:DatabaseOutputLedger.Record)->some View {if r.configuration["kind"] == .string("api") {APIWriteReceiptView(record:r,acknowledge:{ack(r.id)})} else {DatabaseOutputReceiptView(record:r,acknowledge:{ack(r.id)})}}
    private func ack(_ id:String){do{try store.acknowledge(id)}catch{self.error=error.localizedDescription}}
}

/// The dock owns the finite viewport; receipt length must never set its minimum
/// height. Keep tabs outside this surface so compact Results can still switch.
struct DatabaseReceiptViewport<Content:View>:View {
    @ViewBuilder var content:()->Content
    var body:some View {
        GeometryReader {viewport in
            ThemedScrollView {
                VStack(alignment:.leading,spacing:10) {content()}
                    .frame(width:max(0,viewport.size.width),alignment:.topLeading)
            }
        }.frame(minWidth:0,maxWidth:.infinity,minHeight:0,maxHeight:.infinity)
    }
}

struct APIWriteReceiptView:View {
    let record:DatabaseOutputLedger.Record
    var acknowledge:(()->Void)?
    private var unresolved:Bool {["intent","unknown"].contains(record.receipt["status"]?.stringValue ?? "")}
    private var requests:[[String:JSONValue]] {if case .array(let values)=record.receipt["requests"] {return values.compactMap {if case .object(let r)=$0{return r};return nil}};return []}
    var body:some View {
        VStack(alignment:.leading,spacing:9) {
            HStack {Image(systemName:unresolved ? "exclamationmark.triangle":"checkmark.circle");Text(unresolved ? "Delivery needs review":record.receipt["status"] == .string("committed") ? "Requests accepted":"Delivery stopped").fontWeight(.semibold);Spacer();Text(record.createdAt.formatted()).font(.caption2)}.foregroundStyle(unresolved ? Color.orange:ComponentVisualFamily.data.primary)
            Text((record.configuration["method"]?.stringValue ?? "")+" · "+(record.configuration["url"]?.stringValue ?? "")).textSelection(.enabled)
            Text("Operation · "+record.id).foregroundStyle(MLTheme.muted).textSelection(.enabled)
            ScrollView(.horizontal) {
                VStack(spacing:0) {
                    HStack {Text("Request").frame(width:65);Text("First row").frame(width:65);Text("Rows").frame(width:65);Text("HTTP").frame(width:65);Text("Outcome").frame(width:120);Text("Response JSON").frame(width:260)}.padding(7).background(MLTheme.panel)
                    ForEach(Array(requests.enumerated()),id:\.offset) {_,r in HStack {Text(r["request"]?.integerValue.map(String.init) ?? "—").frame(width:65);Text(r["firstRow"]?.integerValue.map(String.init) ?? "—").frame(width:65);Text(r["inputRows"]?.integerValue.map(String.init) ?? "—").frame(width:65);Text(r["httpStatus"]?.integerValue.map(String.init) ?? "—").frame(width:65);Text(r["outcome"]?.stringValue ?? "unknown").frame(width:120);Text(r["responseJSON"]?.stringValue ?? (r["responseOmitted"] == .bool(true) ? "Not retained · response limit":"—")).frame(width:260)}.padding(7);Divider()}
                }.fixedSize(horizontal:true,vertical:false)
            }.overlay(Rectangle().stroke(MLTheme.rule))
            Text("\(record.receipt["acceptedRows"]?.integerValue ?? 0) of \(record.receipt["inputRows"]?.integerValue ?? 0) input rows confirmed accepted. HTTP acceptance does not confirm downstream completion.").foregroundStyle(MLTheme.muted)
            if let message=record.receipt["message"]?.stringValue,record.receipt["status"] != .string("committed") {Text(message).foregroundStyle(unresolved ? Color.orange:MLTheme.muted)}
            if unresolved && !record.acknowledged,let acknowledge {Button("I checked delivery · allow a new write",action:acknowledge).buttonStyle(FlatButtonStyle())}
            if record.acknowledged {Text("Acknowledged · a future run sends new requests.").foregroundStyle(MLTheme.muted)}
            if record.receipt["responseCapture"] == .string("disabledByRunPolicy") {Text("Response bodies were not retained · component-output capture was off.").foregroundStyle(MLTheme.muted)}
            Text("Viewing this recorded receipt never sends a request.").foregroundStyle(MLTheme.muted)
        }.font(.caption).fixedSize(horizontal:false,vertical:true)
    }
}


private struct APIDeliveryResponsesView:View {
    let record:DatabaseOutputLedger.Record
    @State private var grid=CapturedDataGrid(preview:[:])
    var body:some View {
        CapturedDataGridView(grid:grid,tableOnly:true,evidenceLabel:"Captured responses · bounded per-request metadata")
            .task(id:record.id) {
                let values:[[String:JSONValue]]
                if case .array(let requests)=record.receipt["requests"] {values=requests.compactMap {if case .object(let r)=$0{return r};return nil}}else {values=[]}
                let columns:[[String:Any]]=[("request","int64"),("first_input_row","int64"),("input_rows","int64"),("http_status","int64"),("outcome","string"),("response_json","string")].map {["name":$0.0,"dtype":$0.1,"nullable":true]}
                let rows:[[Any]]=values.map {r in [r["request"]?.integerValue as Any? ?? NSNull(),r["firstRow"]?.integerValue as Any? ?? NSNull(),r["inputRows"]?.integerValue as Any? ?? NSNull(),r["httpStatus"]?.integerValue as Any? ?? NSNull(),r["outcome"]?.stringValue ?? "unknown",r["responseJSON"]?.stringValue as Any? ?? NSNull()]}
                grid=CapturedDataGrid(preview:["columns":columns,"rows":rows,"rowCount":rows.count,"partition":"API requests"],rowLimit:256)
            }
    }
}
