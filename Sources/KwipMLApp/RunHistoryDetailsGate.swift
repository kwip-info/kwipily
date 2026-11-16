import SwiftUI

struct RunHistoryDetailsGate<Content:View>:View {
    let root:URL;let projectID:String;let runID:String
    @ViewBuilder var content:(RunHistoryPolicy)->Content
    @State private var policy:RunHistoryPolicy?
    @State private var issue:String?
    var body:some View {
        Group {
            if let issue {Text(issue).font(.caption).foregroundStyle(MLTheme.rose).textSelection(.enabled)}
            else if let policy {
                if policy.enabled {content(policy)}
                else {Text("History details disabled for this run").font(.caption).foregroundStyle(MLTheme.muted).frame(maxWidth:.infinity,maxHeight:.infinity,alignment:.topLeading)}
            } else {ProgressView().controlSize(.small)}
        }.task(id:projectID+runID) {
            policy=nil;issue=nil
            do {let value=try RunHistoryPolicyStore(root:root,projectID:projectID).decision(id:runID);guard !Task.isCancelled else {return};policy=value}
            catch {issue="Run History capture policy unavailable: "+error.localizedDescription}
        }
    }
}
