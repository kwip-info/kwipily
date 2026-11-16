import SwiftUI
import KwipMLCore

private struct LessonContentAccentKey:EnvironmentKey {static let defaultValue=MLTheme.rose}
extension EnvironmentValues {
    var lessonContentAccent:Color {get {self[LessonContentAccentKey.self]} set {self[LessonContentAccentKey.self]=newValue}}
}

/// The inspector's frame/spacing, without making essential instructions collapsible.
struct LessonBlockSection<Content:View>:View {
    @Environment(\.lessonTypography) private var typography
    let title:String
    let symbol:String
    var accent:Color? = nil
    var collapsible=false
    var initiallyExpanded=false
    var identifier:String
    @ViewBuilder let content:()->Content
    @Environment(\.lessonContentAccent) private var topicAccent
    @State private var expanded:Bool?
    private var isExpanded:Bool {!collapsible || (expanded ?? initiallyExpanded)}
    private var tint:Color {accent ?? topicAccent}
    private var header:some View {
        HStack(spacing:8) {
            if collapsible {Image(systemName:isExpanded ? "chevron.down":"chevron.right").font(.system(size:typography.size(9),weight:.semibold)).frame(width:10)}
            Image(systemName:symbol).foregroundStyle(tint)
            Text(title).foregroundStyle(Color.primary).fixedSize(horizontal:false,vertical:true)
            Spacer(minLength:0)
        }.font(.system(size:typography.size(12),weight:.semibold)).padding(.horizontal,12).frame(minHeight:36 * typography.scale).contentShape(Rectangle())
            .background(accent.map {$0.opacity(0.07)} ?? MLTheme.panel)
    }
    var body:some View {
        VStack(spacing:0) {
            if collapsible {
                Button {expanded = !isExpanded} label:{header}.buttonStyle(.plain)
                    .accessibilityLabel(title).accessibilityValue(isExpanded ? "Expanded":"Collapsed").accessibilityIdentifier(identifier+".toggle")
            } else {header.accessibilityAddTraits(.isHeader)}
            VStack(alignment:.leading,spacing:10,content:content)
                .font(.system(size:typography.size(12))).frame(maxWidth:.infinity,alignment:.leading).padding(12)
                .overlay(alignment:.top) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                .frame(height:isExpanded ? nil:0,alignment:.top).clipped()
                .opacity(isExpanded ? 1:0).allowsHitTesting(isExpanded).accessibilityHidden(!isExpanded)
        }.frame(maxWidth:.infinity,alignment:.leading).background(MLTheme.background)
            .overlay(Rectangle().stroke(accent.map {$0.opacity(0.35)} ?? MLTheme.rule))
            .accessibilityIdentifier(identifier)
    }
}

enum LessonBlockPresentation {
    static func title(_ block:LessonContentBlock)->String {
        if !block.title.isEmpty {return block.title}
        switch block.kind {
        case .explanation:return "Understand the idea"
        case .canvasAction:return "Try it on the canvas"
        case .observation:return "What to notice"
        case .visual:return "Worked example"
        case .callout:return "Keep in mind"
        case .detail:return "More detail"
        case .question:return "Predict before you run"
        case .check:return "Check your construction"
        case .feedback:return "Feedback"
        }
    }
    static func symbol(_ kind:LessonContentBlock.Kind)->String {
        switch kind {
        case .explanation:return "book.closed"
        case .canvasAction:return "list.number"
        case .observation:return "eye"
        case .visual:return "tablecells"
        case .callout:return "lightbulb"
        case .detail:return "text.alignleft"
        case .question:return "questionmark.circle"
        case .check:return "checkmark.circle"
        case .feedback:return "text.bubble"
        }
    }
}

struct LessonCheckState:Equatable {
    enum Tone:Equatable {case neutral,good,caution}
    let label:String
    let tone:Tone
    init(grade:LearningGrade?,running:Bool,canSubmit:Bool,unavailable:Bool,current:Bool) {
        if running {label="Checking";tone = .neutral;return}
        if unavailable {label="Unavailable";tone = .caution;return}
        guard let grade else {label=canSubmit ? "Ready":"Not ready";tone = .neutral;return}
        if !current && (grade.status == .passed || grade.status == .failed) {label="Previous result";tone = .neutral;return}
        switch grade.status {
        case .passed:label="Criteria met";tone = .good
        case .failed:label="Needs adjustment";tone = .caution
        case .incomplete:label="Incomplete";tone = .caution
        case .unavailable:label="Unavailable";tone = .caution
        case .cancelled:label="Cancelled";tone = .neutral
        case .executionFailed:label="Could not complete";tone = .caution
        case .budgetExceeded:label="Budget reached";tone = .caution
        }
    }
    var symbol:String {tone == .good ? "checkmark.circle":tone == .caution ? "exclamationmark.circle":"circle.dotted"}
    var tint:Color {tone == .good ? MLTheme.teal:tone == .caution ? .orange:MLTheme.muted}
}
