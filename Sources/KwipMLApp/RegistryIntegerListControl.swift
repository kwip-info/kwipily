import SwiftUI
import KwipMLCore

/// Shared whole-value draft for bounded text and integer-list field templates.
struct RegistryStringFieldDraft {
    var text:String
    var error:String?
    private var committed:String
    init(value:JSONValue) {
        let display:String
        if case .string(let text)=value {display=text}
        else if case .array(let items)=value {display=items.compactMap(\.integerValue).map(String.init).joined(separator:", ")}
        else {display=""}
        text=display;committed=display
    }
    var pending:Bool {text != committed || error != nil}
    mutating func commit(field:ComponentField,origin:String,currentOrigin:String?,apply:(JSONValue)->Bool)->Bool {
        guard pending else {return true}
        guard origin == currentOrigin else {error="Context changed. This list was not applied.";return false}
        guard text.count<=4096 else {error="This list is too long.";return false}
        if field.kind == .text {
            let candidate=JSONValue.string(text)
            guard field.isValidDefinition,field.accepts(candidate) else {error="Enter \(field.minItems ?? 1)–\(field.maxItems ?? 128) characters.";return false}
            guard apply(candidate) else {error="The component rejected this notation. Check the named directions and try again.";return false}
            self=Self(value:candidate);return true
        }
        let parts=text.split(separator:",",omittingEmptySubsequences:false)
        let numbers=parts.compactMap {Int64($0.trimmingCharacters(in:.whitespacesAndNewlines))}
        let candidate=JSONValue.array(numbers.map(JSONValue.integer))
        guard numbers.count == parts.count,field.kind == .integerList,field.accepts(candidate) else {
            error="Enter \(field.minItems ?? 1)–\(field.maxItems ?? 64) comma-separated integers within the field bounds.";return false
        }
        guard apply(candidate) else {error="The component rejected this list. Check its dimensions and try again.";return false}
        self=Self(value:candidate);return true
    }
}

typealias IntegerListDraft=RegistryStringFieldDraft

struct RegistryStringFieldControl:View {
    let field:ComponentField
    let value:JSONValue
    let origin:String
    let currentOrigin:()->String?
    let onDraftChange:((String,Bool)->Void)?
    let apply:(JSONValue)->Bool
    @State private var edit:RegistryStringFieldDraft
    @State private var lifecycle=NumericDraftLifecycle()
    @FocusState private var focused:Bool
    @Environment(\.isEnabled) private var enabled
    init(field:ComponentField,value:JSONValue,origin:String,currentOrigin:@escaping ()->String?,onDraftChange:((String,Bool)->Void)?,apply:@escaping (JSONValue)->Bool) {
        self.field=field;self.value=value;self.origin=origin;self.currentOrigin=currentOrigin;self.onDraftChange=onDraftChange;self.apply=apply
        _edit=State(initialValue:RegistryStringFieldDraft(value:value))
    }
    var body:some View {
        VStack(alignment:.leading,spacing:4) {
            Text(field.title).font(.caption)
            HStack {
                TextField(field.title,text:$edit.text).flatField().focused($focused)
                    .accessibilityLabel(field.title)
                    .accessibilityHint(field.kind == .text ? "Bounded text. Press Return or Apply to commit the complete value.":"Comma-separated integers. Press Return or Apply to commit the complete list.").onSubmit(commit)
                Button("Apply",action:commit).buttonStyle(.borderless).accessibilityLabel("Apply \(field.title)")
            }
            if let error=edit.error {Text(error).font(.system(size:10)).foregroundStyle(MLTheme.rose).fixedSize(horizontal:false,vertical:true)}
        }
        .onAppear {lifecycle.appear();report()}
        .onDisappear {let state=lifecycle.disappear(origin:origin);onDraftChange?(state.origin,state.pending)}
        .onChange(of:origin) {old,_ in onDraftChange?(old,false);edit=RegistryStringFieldDraft(value:value);report()}
        .onChange(of:value) {_,next in if !focused && !edit.pending {edit=RegistryStringFieldDraft(value:next)}}
        .onChange(of:edit.text) {_,_ in report()}
        .onChange(of:focused) {_,next in if !next {commit()}}
    }
    private func report() {if let state=lifecycle.report(origin:origin,pending:edit.pending) {onDraftChange?(state.origin,state.pending)}}
    private func commit() {
        guard enabled,lifecycle.isMounted else {return}
        _=edit.commit(field:field,origin:origin,currentOrigin:currentOrigin(),apply:apply);report()
    }
}

// Preserve the existing call sites and numeric-list acceptance contract.
typealias RegistryIntegerListControl=RegistryStringFieldControl
