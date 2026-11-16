import SwiftUI
import KwipMLCore

/// Captured column names are authoring choices, not a request to reread source files.
/// Keep a saved missing name visible so opening the editor never silently remaps data.
enum ConfigurationColumnChoices {
    static func values(current:String,columns:[DataColumnSchema],allowed:Set<String>)->[FlatChoiceOption] {
        let names=columns.filter{allowed.contains($0.dtype)}.map(\.name)
        var result=names.map{FlatChoiceOption(id:$0,title:$0)}
        if !current.isEmpty && !names.contains(current) {
            result.insert(.init(id:current,title:current+" · unavailable in captured schema"),at:0)
        }
        return result
    }
}

struct ConfigurationColumnField:View {
    let title:String
    @Binding var value:String
    let columns:[DataColumnSchema]
    var allowed:Set<String>=["string","int64","float32","float64","bool"]
    var body:some View {
        VStack(alignment:.leading,spacing:5) {
            Text(title).font(.caption).foregroundStyle(MLTheme.muted)
            if columns.isEmpty {TextField(title,text:$value).flatField()}
            else {FlatChoiceControl(title:title,selection:value,options:ConfigurationColumnChoices.values(current:value,columns:columns,allowed:allowed)){value=$0}}
        }
    }
}
