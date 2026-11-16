import SwiftUI

/// Receipt metadata only. Selecting a row delegates to its existing archive owner.
struct HistoryRecordsTable:View {
    @Environment(\.appearanceTheme) private var theme
    @ObservedObject var history:WorkflowHistoryStore
    @ObservedObject var evaluations:WorkflowEvaluationHistoryStore
    @ObservedObject var backends:BackendFitHistoryStore
    @ObservedObject var preparations:WorkflowDataPreparationHistory
    var selfPlay:[SelfPlayRecord]=[]
    var games:[GameEpisodeRecord]=[]
    var generations:[WorkflowGenerationRecord]=[]
    var fineTuning:[ModelFineTuningRecord]=[]
    var causalEvaluations:[ModelCausalEvaluationRecord]=[]
    let root:URL
    let refreshKey:Int
    let selectedTarget:HistoryTableCatalog.Target?
    let select:(HistoryTableCatalog.Row)->Void
    @State private var groups=WorkflowExperimentHistoryReader.Listing(entries:[],issues:[],truncated:false)
    @State private var groupMessage=""
    @State private var captureLabels:[String:String]=[:]
    @State private var query=""
    @State private var kind=HistoryTableCatalog.Kind.all
    @State private var status=HistoryTableCatalog.Status.all
    @State private var order=HistoryTableCatalog.Sort.newest
    private var rows:[HistoryTableCatalog.Row] {
        HistoryTableCatalog.rows(training:history.listing.entries,evaluation:evaluations.listing.entries,experiments:groups.entries,backends:backends.entries,preparations:preparations.entries,generations:generations,fineTuning:fineTuning,causalEvaluations:causalEvaluations,games:games,selfPlay:selfPlay)
    }
    private var visible:[HistoryTableCatalog.Row] {HistoryTableCatalog.filtered(rows,kind:kind,status:status,query:query,sort:order)}
    var body:some View {
        VStack(spacing:0) {
            HStack(spacing:12) {
                Text("\(rows.count) records").font(.system(size:11)).foregroundStyle(MLTheme.muted)
                Spacer()
                TextField("Search recorded history",text:$query).flatField().frame(width:250)
                    .accessibilityLabel("Search history records")
            }.padding(.horizontal,16).padding(.bottom,8)
            HStack(spacing:8) {
                ForEach(HistoryTableCatalog.Kind.allCases,id:\.self) { value in
                    Button {kind=value} label: {
                        Text(value.rawValue).font(.system(size:11,weight:kind == value ? .semibold:.regular))
                            .padding(.horizontal,9).frame(height:30)
                            .overlay(alignment:.bottom) {Rectangle().fill(kind == value ? MLTheme.rose:.clear).frame(height:2)}
                    }.buttonStyle(.plain).accessibilityLabel("Show "+value.rawValue+" history")
                        .accessibilityValue(kind == value ? "Selected":"")
                }
                Spacer(minLength:8)
                FlatChoiceControl(title:"History status",selection:status.rawValue,options:HistoryTableCatalog.Status.allCases.map {.init(id:$0.rawValue,title:$0.rawValue)}) {if let next=HistoryTableCatalog.Status(rawValue:$0) {status=next}}
                    .frame(width:150)
                FlatChoiceControl(title:"History time order",selection:order.rawValue,options:HistoryTableCatalog.Sort.allCases.map {.init(id:$0.rawValue,title:$0.rawValue)}) {if let next=HistoryTableCatalog.Sort(rawValue:$0) {order=next}}
                    .frame(width:150)
            }.padding(.horizontal,16).padding(.bottom,8)
            GeometryReader {viewport in
                ScrollView(.horizontal) {
                    VStack(spacing:0) {
                        tableHeader(width:max(1000,viewport.size.width))
                        ThemedScrollView {
                            LazyVStack(spacing:0) {
                                ForEach(visible) {record in tableRow(record,width:max(1000,viewport.size.width))}
                            }
                        }
                    }.frame(width:max(1000,viewport.size.width),height:viewport.size.height)
                }
            }
            .overlay {if visible.isEmpty {
                VStack(spacing:8) {
                    Text(rows.isEmpty ? "No recorded history":"No matching records").font(.system(size:14,weight:.semibold))
                    Text(rows.isEmpty ? "Completed, failed and interrupted records appear here after capture.":"Change the filters or clear your search.").font(.caption).foregroundStyle(MLTheme.muted)
                    if !rows.isEmpty {Button("Reset filters") {query="";kind = .all;status = .all}.buttonStyle(FlatButtonStyle())}
                }.padding(20).background(MLTheme.background)
            }}
            HStack {
                Text("\(visible.count) shown · \(rows.count) total").font(.system(size:10)).foregroundStyle(MLTheme.muted)
                Spacer()
                Text("Select a record to inspect its captured evidence").font(.system(size:10)).foregroundStyle(MLTheme.muted)
            }.padding(.horizontal,16).padding(.vertical,8)
            if !issues.isEmpty {DisclosureGroup("History notices · \(issues.count)") {
                ForEach(Array(issues.enumerated()),id:\.offset) {_,issue in Text(issue).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
            }.font(.caption).padding(.horizontal,16).padding(.bottom,8)}
        }.task(id:rows.map(\.id)) {await loadCaptureLabels()}
        .task(id:"\(history.sessionID)-\(refreshKey)") {
            groups = .init(entries:[],issues:[],truncated:false);groupMessage=""
            let project=history.projectID,root=root
            let worker=Task.detached(priority:.utility) {try WorkflowExperimentHistoryReader.list(root:root,projectID:project)}
            do {
                let value=try await withTaskCancellationHandler(operation:{try await worker.value},onCancel:{worker.cancel()})
                guard !Task.isCancelled,history.isActive else {return};groups=value
            } catch {guard !Task.isCancelled,history.isActive else {return};groupMessage=error.localizedDescription}
        }
    }
    private func loadCaptureLabels()async {
        let records=rows,root=root,projectID=history.projectID
        let worker=Task.detached(priority:.utility) {()->[String:String] in
            let store=RunHistoryPolicyStore(root:root,projectID:projectID)
            var labels:[String:String]=[:]
            for row in records {
                if Task.isCancelled {break}
                let captureID:String
                switch row.target {
                case .generation,.gameSession,.selfPlay:labels[row.id]="Required";continue
                case .experiment:labels[row.id]="Per variant";continue
                case .preparation(let id),.backend(let id),.fineTuning(let id),.causalEvaluation(let id):captureID=id
                case .evaluation(let reference):captureID=reference.evaluationID.uuidString
                case .training:captureID=row.searchIdentity
                }
                do {let policy=try store.decision(id:captureID)
                    labels[row.id]=policy == .all ? "Details":!policy.enabled ? "Off":policy.componentOutputs ? "Outputs":"Metrics"
                } catch {labels[row.id]="Unavailable"}
            };return labels
        }
        let labels=await withTaskCancellationHandler(operation:{await worker.value},onCancel:{worker.cancel()})
        guard !Task.isCancelled,history.isActive else {return};captureLabels=labels
    }
    private func captureLabel(_ row:HistoryTableCatalog.Row)->String {captureLabels[row.id] ?? "—"}
    private var issues:[String] {
        var result=history.listing.issues+evaluations.listing.issues+groups.issues
        if history.listing.truncated || evaluations.listing.truncated || groups.truncated {result.append("Scan limit reached; the listing may omit records.")}
        if !groupMessage.isEmpty {result.append(groupMessage)}
        if !backends.status.isEmpty {result.append(backends.status)}
        if !preparations.status.isEmpty {result.append(preparations.status)}
        return result
    }
    private func tableHeader(width:CGFloat)->some View {
        HStack(spacing:0) {
            tableCell("Record",width:width*0.28,header:true)
            tableCell(order == .newest ? "Recorded ↓":"Recorded ↑",width:width*0.18,header:true)
            tableCell("Type",width:width*0.12,header:true)
            tableCell("Status",width:width*0.12,header:true)
            tableCell("Capture",width:width*0.12,header:true)
            tableCell("Evidence",width:width*0.18,header:true)
        }.frame(height:30).background(MLTheme.panel)
            .overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule).frame(height:1)}
            .accessibilityElement(children:.combine).accessibilityLabel("History columns: Record, Recorded, Type, Status, Capture, Evidence")
    }
    private func tableCell(_ text:String,width:CGFloat,header:Bool=false)->some View {
        Text(text).font(.system(size:header ? 10:11,weight:header ? .semibold:.regular))
            .lineLimit(1).frame(width:width-24,alignment:.leading).padding(.horizontal,12)
    }
    private func tableRow(_ record:HistoryTableCatalog.Row,width:CGFloat)->some View {
        let selected=record.target == selectedTarget
        return Button {if history.isActive {select(record)}} label: {
            HStack(spacing:0) {
                VStack(alignment:.leading,spacing:3) {
                    Text(record.title).font(.system(size:12,weight:.medium)).lineLimit(1)
                    if let message=record.message,!message.isEmpty {Text(message).font(.system(size:10)).foregroundStyle(MLTheme.muted).lineLimit(1)}
                }.frame(width:width*0.28-24,alignment:.leading).padding(.horizontal,12)
                tableCell(record.timeLabel,width:width*0.18).foregroundStyle(MLTheme.muted)
                tableCell(record.typeLabel,width:width*0.12)
                Label(record.statusLabel,systemImage:statusSymbol(record.status)).font(.system(size:11))
                    .foregroundStyle(record.status == .failed || record.status == .unavailable ? MLTheme.rose:Color.primary)
                    .frame(width:width*0.12-24,alignment:.leading).padding(.horizontal,12)
                tableCell(captureLabel(record),width:width*0.12).foregroundStyle(MLTheme.muted)
                tableCell(evidenceLabel(record),width:width*0.18).foregroundStyle(MLTheme.muted)
            }.frame(height:44)
                .background(selected ? theme.accentColor.opacity(0.16):MLTheme.background)
                .overlay(alignment:.leading) {Rectangle().fill(selected ? theme.accentColor:.clear).frame(width:3)}
                .overlay(alignment:.bottom) {Rectangle().fill(MLTheme.rule.opacity(0.65)).frame(height:0.5)}
                .contentShape(Rectangle())
        }.buttonStyle(.plain)
            .help([record.title,record.timeLabel,record.typeLabel,record.statusLabel,captureLabel(record),evidenceLabel(record),record.message ?? ""].joined(separator:" · "))
            .accessibilityElement(children:.ignore)
            .accessibilityAddTraits(.isButton)
            .accessibilityLabel([record.title,record.timeLabel,record.typeLabel,record.statusLabel,captureLabel(record),evidenceLabel(record)].joined(separator:". "))
            .accessibilityValue(selected ? "Selected":"")
    }
    private func evidenceLabel(_ record:HistoryTableCatalog.Row)->String {
        if case .training(let reference)=record.target,history.selectedReference == reference,
           case .verified(let value)=history.selection,value.reference == reference {return "Verified archive"}
        if case .evaluation(let reference)=record.target,evaluations.selectedReference == reference,evaluations.disclosure != nil {return "Verified · inspection recorded"}
        return record.evidenceLabel
    }
    private func statusSymbol(_ status:HistoryTableCatalog.Status)->String {
        switch status {case .completed:return "checkmark.circle.fill";case .failed,.unavailable:return "exclamationmark.circle.fill";case .cancelled:return "slash.circle";default:return "clock"}
    }
}
