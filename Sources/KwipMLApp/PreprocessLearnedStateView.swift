import SwiftUI

struct PreprocessLearnedStateView:View {
    let summary:[String:Any]
    var body:some View {
        let rules=summary["rules"] as? [[String:Any]] ?? []
        let names=["Column","Missing","Replacement","Encoding","Categories","Center offset","Scale factor"]
        let columns=names.map{["name":$0,"dtype":"string","nullable":false] as [String:Any]}
        let rows:[[Any]]=rules.map {rule in
            let scaling=rule["scaling"] as? String == "standard"
            let missing=rule["missing"] as? String ?? "keep"
            let vocabulary=rule["vocabulary"] as? [String] ?? []
            return [rule["column"] as? String ?? "",missing=="mostFrequent" ? "Most frequent":missing.capitalized,
                    ["constant","mean","median","mostFrequent"].contains(missing) ? display(rule["fill"]):"—",
                    rule["encoding"] as? String == "oneHot" ? "One-hot":"None",
                    vocabulary.isEmpty ? "—":"\(vocabulary.count): "+vocabulary.prefix(8).joined(separator:", "),
                    scaling ? display(rule["mean"]):"—",scaling ? display(rule["scale"]):"—"]
        }
        VStack(alignment:.leading,spacing:8) {
            Text("Learned preprocessing · \(rules.count) column rules").font(.caption).foregroundStyle(MLTheme.teal)
            CapturedDataGridView(grid:CapturedDataGrid(preview:["columns":columns,"rows":rows,"rowCount":rows.count]),tableOnly:true,evidenceLabel:"Learned rules")
                .frame(minHeight:140,maxHeight:.infinity)
            ResultsSection(title:"Exact learned-state evidence") {CapturedMetadataView(values:summary)}
        }
    }
    private func display(_ value:Any?)->String {
        guard let value,!(value is NSNull) else {return "—"}
        if let text=value as? String {return text}
        return String(describing:value)
    }
}
