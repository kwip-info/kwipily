import SwiftUI
import KwipMLCore

struct WorkflowPerformanceView:View {
    let capture:WorkflowPerformanceCapture
    let graph:GraphDocument
    @State private var selectedRow:Int?
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            Text("Validation forward · \(capture.device.uppercased()) · \(capture.records.count) operations").font(.caption).foregroundStyle(MLTheme.muted)
            WorkflowResultTable(columns:WorkflowResultTableData.columns(["Operation","Elapsed (ms)","Resident before (MiB)","Resident after (MiB)"],numeric:[1,2,3]),rows:capture.records.enumerated().map {index,record in
                WorkflowResultTableData.row(index,[WorkflowResultTableData.operationName(record.nodeID,graph:graph),measurement(record.durationNanoseconds,divisor:1_000_000),measurement(record.residentBytesBefore,divisor:1_048_576),measurement(record.residentBytesAfter,divisor:1_048_576)])
            },selectedRow:$selectedRow)
            if let selectedRow,capture.records.indices.contains(selectedRow) {
                let record=capture.records[selectedRow]
                if let reason=record.timingUnavailableReason ?? record.memoryUnavailableReason {Text(reason).font(.caption).foregroundStyle(MLTheme.muted)}
            }
            DisclosureGroup("Measurement details") {
                Text(capture.device=="cpu" ? "Elapsed time is observed CPU host wall time.":"Elapsed time is host wall time with MPS synchronization around each operation. Synchronization adds overhead; these are not GPU kernel timings.")
                Text("Resident memory covers the worker process, not layer allocations, peak memory, GPU memory or training throughput. Unavailable measurements remain unavailable. Sorting changes this display only.")
            }.font(.caption).foregroundStyle(MLTheme.muted)
        }
    }
    private func measurement(_ value:Int64?,divisor:Double)->String? {value.map {String(format:"%.3f",Double($0)/divisor)}}
}
