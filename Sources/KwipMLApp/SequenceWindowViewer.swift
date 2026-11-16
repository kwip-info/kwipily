import SwiftUI
import KwipMLCore

/// A single verified captured pair. No live configuration or source reads.
struct SequenceWindowSelection {
    let window:String
    let inputs:[Int64]
    let targets:[Int64]
    let mask:[Int64]
    let vocabulary:[String]?
    init(row:CapturedDataGrid.Row,columns:[CapturedDataGrid.Column],configuration c:[String:JSONValue],vocabulary:[String]?)throws {
        try SequenceWindowsConfiguration.validate(c)
        func cell(_ name:String)throws->CapturedDataGrid.Cell {
            guard let i=columns.firstIndex(where:{$0.name==name}),row.cells.indices.contains(i) else {throw DataFlowContractError("Missing sequence pair column.")}
            return row.cells[i]
        }
        guard let x=try cell(c["inputColumn"]!.stringValue!).sequence,let y=try cell(c["targetColumn"]!.stringValue!).sequence,x.count==y.count,x.count==c["contextLength"]?.integerValue,x.allSatisfy({(0...65535).contains($0)}),y.allSatisfy({(0...65535).contains($0)}),let valid=Int(try cell(c["lengthColumn"]!.stringValue!).text),(1...x.count).contains(valid) else {throw DataFlowContractError("Invalid sequence pair dimensions or token IDs.")}
        guard valid==1 || Array(x[1..<valid])==Array(y[0..<(valid-1)]) else {throw DataFlowContractError("Captured targets are not shifted by one token.")}
        let padding=Int64(c["paddingID"]!.integerValue!)
        if valid<x.count {guard x[valid...].allSatisfy({$0==padding}),y[valid...].allSatisfy({$0==padding}) else {throw DataFlowContractError("Captured padding differs from its configured token ID.")}}
        let expected=(0..<x.count).map {Int64($0<valid ? 1:0)}
        if c["incomplete"] == .string("pad") {
            guard try cell(c["maskColumn"]!.stringValue!).sequence==expected else {throw DataFlowContractError("Loss mask does not match valid target positions.")}
        } else {guard valid==x.count else {throw DataFlowContractError("Unpadded window has incomplete targets.")}}
        if let vocabulary {guard (x+y).allSatisfy({$0<Int64(vocabulary.count)}) else {throw DataFlowContractError("Token ID is outside the captured vocabulary.")}}
        window=try cell("window").text;inputs=x;targets=y;mask=expected;self.vocabulary=vocabulary
    }
    func token(_ id:Int64)->String {
        guard let vocabulary,id>=0,id<Int64(vocabulary.count) else {return "Unavailable"}
        let value=vocabulary[Int(id)];return String(reflecting:String(value.prefix(120)))+(value.count>120 ? "…":"")
    }
    func positions(matching query:String)->[Int] {
        let q=query.trimmingCharacters(in:.whitespacesAndNewlines)
        return inputs.indices.filter {q.isEmpty || String($0)==q || String(inputs[$0])==q || String(targets[$0])==q || token(inputs[$0]).localizedCaseInsensitiveContains(q) || token(targets[$0]).localizedCaseInsensitiveContains(q)}
    }
}

struct SequenceWindowViewer:View {
    let selection:SequenceWindowSelection
    let back:()->Void
    @State private var offset=0
    @State private var search=""
    @State private var selectedRow:Int?
    @State private var display="both"
    private let pageSize=256
    var body:some View {
        VStack(alignment:.leading,spacing:8) {
            HStack(spacing:10) {
                Button(action:back) {Label("Back to Output",systemImage:"chevron.left")}.buttonStyle(FlatButtonStyle())
                Text("Window \(selection.window) · \(selection.inputs.count) positions").font(.caption.weight(.semibold))
                Spacer()
                FlatChoiceControl(title:"Display",selection:display,options:[.init(id:"both",title:"Tokens + IDs"),.init(id:"tokens",title:"Tokens"),.init(id:"ids",title:"IDs")]) {display=$0}.frame(width:150)
                TextField("Find token, ID or position",text:$search).flatField().frame(width:190)
            }
            let positions=selection.positions(matching:search)
            CapturedNativeTable(columns:columns,rows:positions.dropFirst(offset).prefix(pageSize).map(row),selectedRow:$selectedRow,rowHeaderTitle:"Position",fillsWidth:true,rowIndexOffset:0).frame(minHeight:100,maxHeight:.infinity)
            HStack {
                Button("Previous") {offset=max(0,offset-pageSize)}.disabled(offset==0)
                Button("Next") {offset+=pageSize}.disabled(offset+pageSize>=positions.count)
                Spacer();Text("\(positions.count) matches").font(.caption).foregroundStyle(MLTheme.muted)
            }.buttonStyle(FlatButtonStyle())
        }.onChange(of:search){_,_ in offset=0;selectedRow=nil}
    }
    private var fields:[(String,String)] {var result:[(String,String)]=[];if display != "ids" {result.append(("input","Input token"))};if display != "tokens" {result.append(("input_id","Input ID"))};if display != "ids" {result.append(("target","Target token"))};if display != "tokens" {result.append(("target_id","Target ID"))};result.append(("mask","Loss"));return result}
    private var columns:[CapturedDataGrid.Column] {fields.enumerated().map {.init(id:$0.offset,name:$0.element.0,label:$0.element.1,dtype:"string",nullability:"required")}}
    private func row(_ i:Int)->CapturedDataGrid.Row {
        .init(index:i,cells:fields.map {key,_ in
            let text:String
            switch key {case "input":text=selection.token(selection.inputs[i]);case "target":text=selection.token(selection.targets[i]);case "input_id":text=String(selection.inputs[i]);case "target_id":text=String(selection.targets[i]);default:text=selection.mask[i]==1 ? "Include":"Exclude"}
            return .init(text:text,isAbsent:false)
        })
    }
}
