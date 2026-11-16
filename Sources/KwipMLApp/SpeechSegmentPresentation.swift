import Foundation
import KwipMLCore

/// Shared by captured previews and paged View/History. Timing travels with the
/// selected row and immutable media reference, never an unrelated current selection.
enum SpeechSegmentPresentation {
    static func interval(row:CapturedDataGrid.Row,columns:[CapturedDataGrid.Column])->ClosedRange<Double>? {
        guard columns.contains(where:{$0.name=="timestamp_coverage"}),columns.contains(where:{$0.name=="source_audio"}),
              let a=columns.firstIndex(where:{$0.name=="start_seconds"}),let b=columns.firstIndex(where:{$0.name=="end_seconds"}),row.cells.indices.contains(a),row.cells.indices.contains(b),
              let start=Double(row.cells[a].text),let end=Double(row.cells[b].text),start.isFinite,end.isFinite,start>=0,end>start else {return nil}
        return start...end
    }
}
