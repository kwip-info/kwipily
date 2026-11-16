import SwiftUI
import KwipMLCore

/// Decimal-list parsing is independent of tensor geometry and integer-list policy.
enum RegistryNumberListParsing {
    static let maximumTextBytes=131072
    static func display(_ value:JSONValue)->String {
        guard case .array(let items)=value else {return ""}
        return items.compactMap(\.doubleValue).map {String($0)}.joined(separator:", ")
    }
    static func parse(_ text:String,field:ComponentField)->JSONValue? {
        guard field.kind == .numberList,field.isValidDefinition,text.utf8.count<=maximumTextBytes else {return nil}
        let trimmed=text.trimmingCharacters(in:.whitespacesAndNewlines)
        if trimmed.isEmpty {let empty=JSONValue.array([]);return field.accepts(empty) ? empty:nil}
        let parts=trimmed.split(separator:",",omittingEmptySubsequences:false)
        guard let limit=field.maxItems,parts.count<=limit else {return nil}
        var values:[JSONValue]=[];values.reserveCapacity(parts.count)
        for part in parts {
            guard let number=Double(part.trimmingCharacters(in:.whitespacesAndNewlines)),number.isFinite else {return nil}
            values.append(.number(number))
        }
        let value=JSONValue.array(values)
        return field.accepts(value) ? value:nil
    }
}

struct RegistryNumberListDraft {
    var text:String
    var error:String?
    private var committed:String
    init(value:JSONValue) {let display=RegistryNumberListParsing.display(value);text=display;committed=display}
    var pending:Bool {text != committed || error != nil}
    mutating func discard() {text=committed;error=nil}
    mutating func commit(field:ComponentField,origin:String,currentOrigin:String?,apply:(JSONValue)->Bool)->Bool {
        guard pending else {return true}
        guard origin == currentOrigin else {error="Context changed. This list was not applied.";return false}
        guard let value=RegistryNumberListParsing.parse(text,field:field) else {
            error="Enter \(field.minItems ?? 1)–\(field.maxItems ?? 4096) comma-separated finite numbers within the field bounds.";return false
        }
        guard apply(value) else {error="The component rejected this list. Check its required size and try again.";return false}
        self=Self(value:value);return true
    }
}

/// Shared registry template for decimal lists; ownership guards match other fields.
struct RegistryNumberListControl:View {
    let field:ComponentField
    let value:JSONValue
    let origin:String
    let currentOrigin:()->String?
    let onDraftChange:((String,Bool)->Void)?
    let apply:(JSONValue)->Bool
    @State private var edit:RegistryNumberListDraft
    @State private var lifecycle=NumericDraftLifecycle()
    @FocusState private var focused:Bool
    @Environment(\.isEnabled) private var enabled
    init(field:ComponentField,value:JSONValue,origin:String,currentOrigin:@escaping ()->String?,onDraftChange:((String,Bool)->Void)?,apply:@escaping (JSONValue)->Bool) {
        self.field=field;self.value=value;self.origin=origin;self.currentOrigin=currentOrigin;self.onDraftChange=onDraftChange;self.apply=apply
        _edit=State(initialValue:RegistryNumberListDraft(value:value))
    }
    var body:some View {
        VStack(alignment:.leading,spacing:4) {
            Text(field.title).font(.caption)
            HStack {
                TextField(field.title,text:$edit.text).flatField().focused($focused)
                    .accessibilityLabel(field.title).accessibilityHint("Comma-separated numbers. Apply commits the complete list; Discard restores the saved value.").onSubmit(commit)
                Button("Apply",action:commit).buttonStyle(.borderless).accessibilityLabel("Apply \(field.title)")
                if edit.pending {Button("Discard",action:discard).buttonStyle(.borderless).accessibilityLabel("Discard \(field.title)")}
            }
            if let error=edit.error {Text(error).font(.system(size:10)).foregroundStyle(MLTheme.rose).fixedSize(horizontal:false,vertical:true)}
        }
        .onAppear {lifecycle.appear();report()}
        .onDisappear {let state=lifecycle.disappear(origin:origin);onDraftChange?(state.origin,state.pending)}
        .onChange(of:origin) {old,_ in onDraftChange?(old,false);edit=RegistryNumberListDraft(value:value);report()}
        .onChange(of:value) {_,next in if !focused && !edit.pending {edit=RegistryNumberListDraft(value:next)}}
        .onChange(of:edit.text) {_,_ in report()}
        .onChange(of:focused) {_,next in if !next {commit()}}
    }
    private func report() {if let state=lifecycle.report(origin:origin,pending:edit.pending) {onDraftChange?(state.origin,state.pending)}}
    private func commit() {
        guard enabled,lifecycle.isMounted else {return}
        _=edit.commit(field:field,origin:origin,currentOrigin:currentOrigin(),apply:apply);report()
    }
    private func discard() {
        guard enabled,lifecycle.isMounted,currentOrigin() == origin else {return}
        edit.discard();report()
    }
}
