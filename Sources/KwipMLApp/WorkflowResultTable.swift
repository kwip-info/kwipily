import SwiftUI
import KwipMLCore

/// Table presentation over already bounded observations. No source reads or
/// execution/capture mutation. Sorting does not change recorded evidence.
enum WorkflowResultTableData {
    static func columns(_ names:[String],numeric:Set<Int>=[])->[CapturedDataGrid.Column] {
        names.enumerated().map {.init(id:$0.offset,name:$0.element,label:$0.element,dtype:numeric.contains($0.offset) ? "float64":"string",nullability:"recorded")}
    }
    static func row(_ index:Int,_ values:[String?])->CapturedDataGrid.Row {
        .init(index:index,cells:values.map {.init(text:$0 ?? "Unavailable",isAbsent:$0==nil)})
    }
    static func sorted(_ rows:[CapturedDataGrid.Row],columns:[CapturedDataGrid.Column],sort:TableGridSort)->[CapturedDataGrid.Row] {
        rows.sorted {a,b in
            guard let column=sort.column,columns.indices.contains(column),a.cells.indices.contains(column),b.cells.indices.contains(column) else {return sort.descending ? a.index>b.index:a.index<b.index}
            let left=a.cells[column],right=b.cells[column]
            if left.isAbsent != right.isAbsent {return !left.isAbsent}
            let comparison:ComparisonResult
            if columns[column].dtype=="float64",let x=Double(left.text),let y=Double(right.text) {comparison=x<y ? .orderedAscending:x>y ? .orderedDescending:.orderedSame}
            else {comparison=left.text.localizedStandardCompare(right.text)}
            if comparison == .orderedSame {return a.index<b.index}
            return sort.descending ? comparison == .orderedDescending:comparison == .orderedAscending
        }
    }
    static func operationName(_ nodeID:String,graph:GraphDocument)->String {
        guard let node=graph.nodes.first(where:{$0.id==nodeID}) else {return "Unavailable component"}
        return node.name.isEmpty ? ComponentStyle.shortTitle(node.type,fallback:node.type):node.name
    }
}

struct WorkflowResultTable:View {
    let columns:[CapturedDataGrid.Column]
    let rows:[CapturedDataGrid.Row]
    @Binding var selectedRow:Int?
    var maximumHeight:CGFloat=360
    @State private var sort=TableGridSort()
    var body:some View {
        CapturedNativeTable(columns:columns,rows:WorkflowResultTableData.sorted(rows,columns:columns,sort:sort),selectedRow:$selectedRow,rowHeaderTitle:"#",sort:sort,onSort:{sort=sort.toggled($0)})
            .frame(height:min(maximumHeight,max(120,CGFloat(rows.count)*31+40)))
            .frame(maxWidth:.infinity).overlay(Rectangle().stroke(MLTheme.rule))
    }
}
