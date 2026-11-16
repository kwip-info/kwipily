import SwiftUI

/// Results shares the visible, sharp section framing used by the inspectors.
struct ResultsSection<Content:View>:View {
    let title:String
    var symbol="tablecells"
    var initiallyExpanded=false
    @ViewBuilder let content:()->Content
    var body:some View {
        InspectorSection(title,symbol:symbol,identifier:"results.section."+title,initiallyExpanded:initiallyExpanded,content:content)
    }
}

/// Captured metadata stays in aligned fields. Raw evidence has its own bounded viewport.
struct CapturedMetadataView:View {
    let values:[String:Any]
    private var fields:[(String,String)] {
        var result:[(String,String)]=[]
        func visit(_ value:Any,path:String,depth:Int) {
            guard result.count<80 else {return}
            if depth<6,let dictionary=value as? [String:Any] {
                if dictionary.isEmpty {result.append((path,"Empty object"))}
                for key in dictionary.keys.sorted() {visit(dictionary[key]!,path:path.isEmpty ? key:path+"."+key,depth:depth+1)}
            } else if depth<6,let array=value as? [Any] {
                if array.isEmpty {result.append((path,"Empty list"))}
                for (index,item) in array.prefix(8).enumerated() {visit(item,path:path+"[\(index)]",depth:depth+1)}
                if array.count>8 {result.append((path,"\(array.count-8) additional items · see raw evidence"))}
            } else if value is [String:Any] || value is [Any] {result.append((path,"Nested payload · see raw evidence"))}
            else {
                let text=value is NSNull ? "null":String(describing:value)
                result.append((path,text.count>500 ? String(text.prefix(500))+"… · see raw evidence":text))
            }
        }
        visit(values,path:"",depth:0)
        return result
    }
    var body:some View {
        VStack(spacing:0) {
            ForEach(Array(fields.enumerated()),id:\.offset) {_,field in
                HStack(alignment:.top,spacing:16) {
                    Text(field.0).font(.system(size:10,weight:.medium,design:.monospaced)).foregroundStyle(MLTheme.muted).frame(width:210,alignment:.leading)
                    Text(field.1).font(.system(size:11,design:.monospaced)).frame(maxWidth:.infinity,alignment:.leading).textSelection(.enabled)
                }.padding(.vertical,7).overlay(alignment:.bottom){Rectangle().fill(MLTheme.rule).frame(height:1)}
            }
            Text("Structured view · up to 80 fields, eight items per list. Raw evidence preserves the captured payload.").font(.caption2).foregroundStyle(MLTheme.muted).padding(.top,8)
            ResultsSection(title:"Raw evidence",symbol:"curlybraces") {
                ThemedScrollView([.horizontal,.vertical]) {
                    Text(raw).font(.system(size:10,design:.monospaced)).textSelection(.enabled).padding(8)
                }.frame(height:180).background(MLTheme.background)
            }
        }
    }
    private var raw:String {(try? JSONSerialization.data(withJSONObject:values,options:[.prettyPrinted,.sortedKeys])).flatMap {String(data:$0,encoding:.utf8)} ?? "Captured metadata cannot be rendered as JSON."}
}
