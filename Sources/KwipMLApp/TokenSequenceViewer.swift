import SwiftUI

/// Bounded immutable sequence, indexed in pages. No source/vocabulary refitting.
struct TokenSequenceViewer:View {
    let ids:[Int64]
    var vocabulary:[String]? = nil
    @State private var offset=0
    @State private var query=""
    private let pageSize=256
    private var positions:[Int] {let text=query.trimmingCharacters(in:.whitespacesAndNewlines);return ids.indices.filter {text.isEmpty || String($0)==text || String(ids[$0])==text || token(ids[$0])?.contains(text)==true}}
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            HStack {Text("\(ids.count) tokens").font(.headline);Spacer();TextField("Find token, ID or position",text:$query).flatField().frame(maxWidth:220)}
            let found=positions
            let page=Array(found.dropFirst(offset).prefix(pageSize))
            HStack {Text("Position").frame(width:80,alignment:.leading);Text("Token ID").frame(width:90,alignment:.leading);Text("Token");Spacer()}.font(.caption.weight(.semibold)).padding(8).background(MLTheme.panel)
            ThemedScrollView {
                LazyVStack(alignment:.leading,spacing:0) {
                    ForEach(page,id:\.self) {i in HStack {Text(String(i)).frame(width:80,alignment:.leading);Text(String(ids[i])).frame(width:90,alignment:.leading);Text(token(ids[i]).map {String(reflecting:String($0.prefix(120)))+($0.count>120 ? "…":"")} ?? "Unavailable").lineLimit(1);Spacer()}.font(.system(.caption,design:.monospaced)).padding(8).background(i%2==0 ? MLTheme.panel.opacity(0.4):Color.clear)}
                }
            }
            HStack {
                Button("Previous") {offset=max(0,offset-pageSize)}.disabled(offset==0)
                Button("Next") {offset+=pageSize}.disabled(offset+pageSize>=found.count)
                Spacer();Text("\(found.count) matches").font(.caption).foregroundStyle(MLTheme.muted)
            }.buttonStyle(FlatButtonStyle())
        }.padding(14).frame(minWidth:360,minHeight:300).background(MLTheme.background).onChange(of:query){_,_ in offset=0}
    }
    private func token(_ id:Int64)->String? {guard let vocabulary,id>=0,id<Int64(vocabulary.count) else {return nil};return vocabulary[Int(id)]}

}
