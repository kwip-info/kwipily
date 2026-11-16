import SwiftUI
import KwipMLCore

/// Shared inert editor, never tokenizes, loads weights or executes a question on render.
struct DecisionQuestionEditor:View {
    @Binding var value:[String:JSONValue]
    @State private var selected=0
    private var questions:[JSONValue] {if case .array(let a)=value["questions"]{return a};return []}
    private var current:[String:JSONValue] {if questions.indices.contains(selected),case .object(let q)=questions[selected]{return q};return [:]}
    private var options:[JSONValue] {if case .array(let a)=current["options"]{return a};return []}
    var body:some View {
        VStack(alignment:.leading,spacing:18) {
            TypedDecisionGroup("Question set") {
                entry("Set name",key:"name",root:true)
                entry("Stable set ID",key:"id",root:true)
                Text("Revision \(value["revision"]?.integerValue ?? 1) · edits create a new revision on Apply. Train, Decide and saved models pin the actual content; History keeps its original revision.").font(.caption).foregroundStyle(MLTheme.muted)
            }
            TypedDecisionGroup("Questions") {
                FlatChoiceControl(title:"Question",selection:String(selected),options:questions.enumerated().map {index,entry in
                    let q=object(entry);return .init(id:String(index),title:(q["id"]?.stringValue ?? "Question")+" · "+typeTitle(q["type"]?.stringValue ?? "choice"))
                }) {if let index=Int($0){selected=index}}
                HStack {
                    Button("Add question"){var a=questions;let id="question"+String(UUID().uuidString.prefix(8));a.append(.object(["id":.string(id),"type":.string("choice"),"instructions":.string("Choose the option supported by this record."),"options":.array([.object(["id":.string("first"),"label":.string("First option")]),.object(["id":.string("second"),"label":.string("Second option")])])]));value["questions"] = .array(a);selected=a.count-1}.buttonStyle(FlatButtonStyle()).disabled(questions.count>=16)
                    Button("Remove question"){var a=questions;guard a.indices.contains(selected)else{return};a.remove(at:selected);value["questions"] = .array(a);selected=max(0,min(selected,a.count-1))}.buttonStyle(FlatButtonStyle()).disabled(questions.count<=1)
                }
            }
            if !current.isEmpty {
                TypedDecisionGroup("Question definition") {
                    entry("Stable question ID",key:"id")
                    FlatChoiceControl(title:"Question type",selection:current["type"]?.stringValue ?? "choice",options:[("choice","Choice"),("score","Ordinal score"),("noul","Yes / No")].map{.init(id:$0.0,title:$0.1)}){type in
                        change("type",.string(type))
                        if type=="noul"{change("options",.array([.object(["id":.string("false"),"label":.string("no, the statement does not hold")]),.object(["id":.string("true"),"label":.string("yes, the statement holds")])]))}
                    }
                    Text("Instructions · what should the model decide?").font(.caption).foregroundStyle(MLTheme.muted)
                    TextEditor(text:Binding(get:{current["instructions"]?.stringValue ?? ""},set:{change("instructions",.string($0))})).font(.body).scrollContentBackground(.hidden).padding(8).frame(minHeight:86,maxHeight:120).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))
                }
                TypedDecisionGroup(current["type"] == .string("score") ? "Ordered levels":"Options") {
                    Text(current["type"] == .string("score") ? "Order defines the score's meaning. The expected index is a position on these levels, not a probability or physical unit.":"Every option keeps its stable key and complete probability. Yes/no semantic order is false then true.").font(.caption).foregroundStyle(MLTheme.muted)
                    ForEach(Array(options.indices),id:\.self){index in optionRow(index)}
                    if current["type"] != .string("noul") {Button("Add option"){var a=options;a.append(.object(["id":.string("option"+String(UUID().uuidString.prefix(8))),"label":.string("New option")]));change("options",.array(a))}.buttonStyle(FlatButtonStyle()).disabled(options.count>=32)}
                }
            }
        }.onChange(of:questions.count){_,count in selected=max(0,min(selected,count-1))}
    }
    private func optionRow(_ index:Int)->some View {
        VStack(alignment:.leading,spacing:6) {
            HStack(alignment:.top,spacing:10) {
                VStack(alignment:.leading,spacing:4){Text("Stable key").font(.caption2).foregroundStyle(MLTheme.muted);TextField("Option key",text:optionBinding(index,"id")).flatField().disabled(current["type"] == .string("noul"))}.frame(width:160)
                VStack(alignment:.leading,spacing:4){Text("Label shown to model").font(.caption2).foregroundStyle(MLTheme.muted);TextField("Option label",text:optionBinding(index,"label")).flatField()}
            }
            if current["type"] != .string("noul") {
                HStack{Text("Position \(index)").font(.caption2).foregroundStyle(MLTheme.muted);Spacer();Button("Up"){move(index,-1)}.buttonStyle(FlatButtonStyle()).disabled(index==0);Button("Down"){move(index,1)}.buttonStyle(FlatButtonStyle()).disabled(index==options.count-1);Button("Remove"){var a=options;a.remove(at:index);change("options",.array(a))}.buttonStyle(FlatButtonStyle()).disabled(options.count<=2)}
            }
        }.padding(10).overlay(Rectangle().stroke(MLTheme.rule))
    }
    private func entry(_ title:String,key:String,root:Bool=false)->some View {VStack(alignment:.leading,spacing:5){Text(title).font(.caption).foregroundStyle(MLTheme.muted);TextField(title,text:Binding(get:{(root ? value:current)[key]?.stringValue ?? ""},set:{if root{value[key] = .string($0)}else{change(key,.string($0))}})).flatField()}}
    private func optionBinding(_ index:Int,_ key:String)->Binding<String>{Binding(get:{guard options.indices.contains(index)else{return ""};return object(options[index])[key]?.stringValue ?? ""},set:{raw in var a=options;guard a.indices.contains(index)else{return};var o=object(a[index]);o[key] = .string(raw);a[index] = .object(o);change("options",.array(a))})}
    private func change(_ key:String,_ next:JSONValue){var a=questions;guard a.indices.contains(selected)else{return};var q=current;q[key]=next;a[selected] = .object(q);value["questions"] = .array(a)}
    private func move(_ index:Int,_ offset:Int){var a=options;let next=index+offset;guard a.indices.contains(next)else{return};a.swapAt(index,next);change("options",.array(a))}
    private func object(_ value:JSONValue)->[String:JSONValue]{if case .object(let x)=value{return x};return [:]}
    private func typeTitle(_ type:String)->String{type=="score" ? "Ordinal score":type=="noul" ? "Yes / No":"Choice"}
}

struct TypedDecisionGroup<Content:View>:View {
    let title:String
    @ViewBuilder var content:()->Content
    init(_ title:String,@ViewBuilder content:@escaping()->Content){self.title=title;self.content=content}
    var body:some View{VStack(alignment:.leading,spacing:12){Text(title).font(.headline);content()}.padding(16).frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.panel).overlay(Rectangle().stroke(MLTheme.rule))}
}

enum DecisionQuestionResource {
    static func advanced(_ value:[String:JSONValue],from old:[String:JSONValue])->[String:JSONValue] {
        var next=value
        if next != old,next["id"]==old["id"],next["revision"]==old["revision"]{next["revision"] = .integer(Int64((old["revision"]?.integerValue ?? 1)+1))}
        return next
    }
    static func choices(_ document:WorkflowDocument)->[(String,[String:JSONValue])] {
        document.nodes.compactMap{node in
            let config:[String:JSONValue]
            switch node.payload {case .dataOperator(let id,let c) where id==TypedDecisionConfiguration.operatorID:config=c;case .train(let c):if case .object(let f)=c["fineTuning"]{config=f}else{return nil};default:return nil}
            guard case .object(let q)=config["questionSet"] else{return nil}
            return ((node.name.isEmpty ? node.address.localID:node.name)+" · "+(q["name"]?.stringValue ?? "Questions")+" r"+String(q["revision"]?.integerValue ?? 1),q)
        }
    }
}
