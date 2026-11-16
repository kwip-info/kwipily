import SwiftUI
import KwipMLCore

struct LessonContentBlockView:View {
    let block:LessonContentBlock
    var selectTarget:((String)->Void)? = nil
    var check:(()->Void)? = nil
    @State private var selectedChoices=Set<String>()
    @Environment(\.lessonContentAccent) private var topicAccent
    @Environment(\.colorScheme) private var scheme
    private var calloutTint:Color {scheme == .light ? Color(red:0.55,green:0.39,blue:0.12):Color(red:0.88,green:0.71,blue:0.44)}
    var body:some View {
        LessonBlockSection(title:LessonBlockPresentation.title(block),symbol:LessonBlockPresentation.symbol(block.kind),
            accent:block.kind == .callout ? calloutTint:nil,collapsible:block.kind == .detail,identifier:"lesson.block."+block.id) {
            if block.kind == .visual {LessonVisualView(block:block)}
            else {
                if !block.text.isEmpty {LessonRichText(text:block.text)}
                if block.kind == .question {
                    ForEach(Array((block.choices ?? []).prefix(32).enumerated()),id:\.offset) {_,choice in
                        Toggle(isOn:Binding(get:{selectedChoices.contains(choice.id)},set:{value in
                            if value {selectedChoices.insert(choice.id)} else {selectedChoices.remove(choice.id)}
                        })) {LessonRichText(text:choice.text)}
                            .toggleStyle(.checkbox).tint(topicAccent).padding(8).frame(maxWidth:.infinity,alignment:.leading)
                            .overlay(Rectangle().stroke(MLTheme.rule))
                    }
                    if let answer=block.answer {
                        LessonBlockSection(title:"Compare explanation",symbol:"questionmark.circle",collapsible:true,identifier:"lesson.block."+block.id+".answer") {LessonRichText(text:answer)}
                    }
                }
                if let target=block.targetType,let selectTarget {
                    Button {selectTarget(target)} label:{Label("Select component",systemImage:"scope")}
                }
                if block.kind == .check,let check {Button("Check construction",action:check)}
            }
        }
    }
}
