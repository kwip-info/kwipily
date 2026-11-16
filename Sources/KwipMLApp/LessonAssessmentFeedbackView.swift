import SwiftUI
import KwipMLCore

/// View data from real outcomes; never an authored completion claim or hidden-fixture handout.
struct LessonFeedbackItem:Identifiable {
    let id:String
    let title:String
    let passed:Bool
    let nextStep:String?
    let evidence:String?
    let technicalNextStep:String?
    static func project(_ outcome:LearningCriterionOutcome,title:String,publicNextStep:String?=nil,assessment:Bool)->Self {
        .init(id:outcome.criterionID,title:title,passed:outcome.passed,
            nextStep:outcome.passed ? nil:publicNextStep ?? outcome.nextStep,
            evidence:assessment ? nil:outcome.evidence,technicalNextStep:outcome.passed ? nil:outcome.nextStep)
    }
}

struct LessonCriteriaRowsView:View {
    @Environment(\.lessonTypography) private var typography
    let items:[LessonFeedbackItem]
    var showsTechnicalFeedback=false
    var body:some View {
        VStack(alignment:.leading,spacing:10) {
            ForEach(Array(items.enumerated()),id:\.offset) {index,item in
                VStack(alignment:.leading,spacing:7) {
                    ViewThatFits(in:.horizontal) {
                        HStack(alignment:.top,spacing:10) {title(item);Spacer(minLength:6);status(item)}
                        VStack(alignment:.leading,spacing:5) {title(item);status(item)}
                    }
                    if let detail=item.nextStep,!detail.isEmpty {LessonRichText(text:detail)}
                    if showsTechnicalFeedback {
                        LessonBlockSection(title:"Exact feedback",symbol:"function",collapsible:true,identifier:"lesson.criterion."+item.id+".technical") {
                            Text(item.id).font(.system(size:typography.size(10))).foregroundStyle(MLTheme.muted).textSelection(.enabled)
                            if let evidence=item.evidence {LessonRichText(text:evidence)}
                            if let next=item.technicalNextStep {LessonRichText(text:next)}
                        }
                    }
                }
                if index<items.count-1 {Divider()}
            }
        }
    }
    private func title(_ item:LessonFeedbackItem)->some View {Text(item.title).font(.system(size:typography.size(12),weight:.medium)).fixedSize(horizontal:false,vertical:true)}
    private func status(_ item:LessonFeedbackItem)->some View {
        Label(item.passed ? "Met":"Needs adjustment",systemImage:item.passed ? "checkmark.circle":"exclamationmark.circle")
            .font(.system(size:typography.size(10),weight:.medium)).foregroundStyle(item.passed ? MLTheme.teal:Color.orange).fixedSize()
    }
}
