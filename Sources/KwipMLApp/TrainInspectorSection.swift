import SwiftUI

/// Shared inspector behavior keeps drafts mounted when a training group collapses.
struct TrainInspectorSection<Content:View>:View {
    let title:String
    let symbol:String
    var summary:String?=nil
    var initiallyExpanded=false
    @ViewBuilder let content:()->Content
    @Environment(\.colorScheme) private var colorScheme
    init(_ title:String,symbol:String,summary:String?=nil,initiallyExpanded:Bool=false,@ViewBuilder content:@escaping()->Content) {
        self.title=title;self.symbol=symbol;self.summary=summary;self.initiallyExpanded=initiallyExpanded;self.content=content
    }
    var body:some View {
        InspectorSection(title,symbol:symbol,identifier:"train.section."+title.lowercased().replacingOccurrences(of:" ",with:"-"),initiallyExpanded:initiallyExpanded,summary:summary,accent:colorScheme == .dark ? ComponentVisualFamily.training.primary:ComponentVisualFamily.training.secondary,content:content)
    }
}
