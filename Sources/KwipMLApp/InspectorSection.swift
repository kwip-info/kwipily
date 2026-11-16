import SwiftUI

/// Reusable inspector slot. Collapsing changes presentation only: controls stay
/// mounted so local drafts and their pending-edit ownership are retained.
struct InspectorSection<Content:View>:View {
    let title:String
    let symbol:String
    let identifier:String
    let summary:String?
    let accent:Color?
    @State private var isExpanded:Bool
    @ViewBuilder let content:()->Content
    init(_ title:String,symbol:String,identifier:String,initiallyExpanded:Bool=true,summary:String?=nil,accent:Color?=nil,@ViewBuilder content:@escaping()->Content) {
        self.title=title;self.symbol=symbol;self.identifier=identifier;self.summary=summary;self.accent=accent
        _isExpanded=State(initialValue:initiallyExpanded);self.content=content
    }
    var body:some View {
        VStack(spacing:0) {
            Button {isExpanded.toggle()} label: {
                HStack(spacing:8) {
                    Image(systemName:isExpanded ? "chevron.down":"chevron.right").font(.system(size:9,weight:.semibold)).frame(width:10)
                    Label(title,systemImage:symbol).font(.system(size:12,weight:.semibold)).foregroundStyle(accent ?? Color.primary).lineLimit(1).layoutPriority(1)
                    Spacer(minLength:0)
                    if let summary,!isExpanded {Text(summary).font(.caption2).foregroundStyle(MLTheme.muted).lineLimit(1).frame(maxWidth:100,alignment:.trailing)}
                }.padding(.horizontal,12).frame(height:36).contentShape(Rectangle())
            }.buttonStyle(.plain).background(accent.map {$0.opacity(0.07)} ?? MLTheme.panel)
                .accessibilityLabel(title).accessibilityValue(isExpanded ? "Expanded":"Collapsed")
                .accessibilityIdentifier(identifier+".toggle")
            VStack(alignment:.leading,spacing:10,content:content)
                .frame(maxWidth:.infinity,alignment:.leading).padding(12)
                .overlay(alignment:.top) {Rectangle().fill(MLTheme.rule).frame(height:1)}
                .frame(height:isExpanded ? nil:0,alignment:.top).clipped()
                .opacity(isExpanded ? 1:0).allowsHitTesting(isExpanded).accessibilityHidden(!isExpanded)
        }.frame(maxWidth:.infinity,alignment:.leading)
            .overlay(Rectangle().stroke(accent.map {$0.opacity(0.25)} ?? MLTheme.rule))
    }
}
struct InspectorConfigurationSection<Content:View>:View {
    let isEnabled:Bool
    @ViewBuilder let content:()->Content
    init(isEnabled:Bool=true,@ViewBuilder content:@escaping()->Content) {
        self.isEnabled=isEnabled;self.content=content
    }
    var body:some View {
        InspectorSection("Configuration",symbol:"slider.horizontal.3",identifier:"inspector.configuration") {
            VStack(alignment:.leading,spacing:10,content:content).disabled(!isEnabled)
        }
    }
}
