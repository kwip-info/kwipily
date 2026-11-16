import SwiftUI
import KwipMLCore

/// One registry-driven editor for primitive, instance and exposed-default fields.
struct RegistryFieldControl: View {
    let field: ComponentField
    let value: JSONValue
    let origin: String
    let currentOrigin: () -> String?
    var onDraftChange: ((String, Bool) -> Void)? = nil
    var componentID: String? = nil
    var configuration: [String: JSONValue] = [:]
    let apply: (JSONValue) -> Bool
    @Environment(\.isEnabled) private var enabled
    @State private var rejection: String?
    @State private var numberListPending=false

    var body: some View {
        VStack(alignment: .leading, spacing: 5) {
            if field.allowsNull {
                Toggle("Use "+field.title,isOn:Binding(get:{value != .null},set:{selected in
                    guard enabled,currentOrigin() == origin else {rejection="Context changed. This value was not applied.";return}
                    guard field.kind != .numberList || !numberListPending else {rejection="Apply or discard this list before changing its optional setting.";return}
                    let candidate:JSONValue=selected ? field.valueWhenEnabled:.null
                    guard field.isValidDefinition,field.accepts(candidate),apply(candidate) else {rejection="This optional setting was not applied.";return}
                    rejection=nil
                })).toggleStyle(.checkbox).font(.caption).accessibilityLabel("Use "+field.title)
            }
            if field.allowsNull && value == .null {
                Text("Not set").font(.caption).foregroundStyle(MLTheme.muted)
            } else {
            switch field.kind {
            case .choice:
                Text(field.title).font(.caption)
                if field.isValidDefinition {
                    FlatChoiceControl(title: field.title, selection: value.stringValue ?? "",
                        options: (field.choices ?? []).map { FlatChoiceOption(id: $0.value, title: $0.title) },
                        placeholder: "Select…") { next in
                            guard enabled, currentOrigin() == origin else { rejection = "Context changed. This value was not applied."; return }
                            let candidate = JSONValue.string(next)
                            guard field.isValidDefinition, field.accepts(candidate), apply(candidate) else { rejection = "This value was not applied. The component changed or rejected the edit."; return }
                            rejection = nil
                        }
                } else { Text("This field has no valid choices.").font(.caption).foregroundStyle(MLTheme.rose) }
            case .numberList:
                RegistryNumberListControl(field:field,value:value,origin:origin,currentOrigin:currentOrigin,onDraftChange:{key,pending in numberListPending=pending;onDraftChange?(key,pending)},apply:apply)
            case .integerList, .text:
                RegistryIntegerListControl(field:field,value:value,origin:origin,currentOrigin:currentOrigin,onDraftChange:onDraftChange,apply:apply)
            case .integer:
                Text(field.title).font(.caption)
                if let minimum = field.min, let maximum = field.max {
                    BoundedNumberField(title: field.title, value: .wholeDisplay(value.doubleValue ?? field.defaultValue.doubleValue ?? 0), domain: .integer(minimum: minimum, maximum: maximum), origin: origin, currentOrigin: currentOrigin, onDraftChange: onDraftChange) { numeric in
                        guard case .integer(let integer) = numeric else { return false }
                        let candidate = JSONValue.integer(integer)
                        return field.accepts(candidate) && apply(candidate)
                    }
                } else { Text("This field has no valid integer bounds.").font(.caption).foregroundStyle(MLTheme.rose) }
            case .number:
                Text(field.title).font(.caption)
                if field.isValidDefinition, let minimum = field.min, let maximum = field.max {
                    BoundedNumberField(title: field.title, value: .decimal(value.doubleValue ?? field.defaultValue.doubleValue ?? 0), domain: .decimal(minimum: minimum, maximum: maximum), origin: origin, currentOrigin: currentOrigin, onDraftChange: onDraftChange) { numeric in
                        guard case .decimal(let decimal) = numeric else { return false }
                        let candidate = JSONValue.number(decimal)
                        return field.accepts(candidate) && apply(candidate)
                    }
                } else { Text("This field has no valid number bounds.").font(.caption).foregroundStyle(MLTheme.rose) }
            case .boolean:
                Toggle(field.title, isOn: Binding(get: { value.boolValue ?? field.defaultValue.boolValue ?? false }, set: { next in
                    guard enabled, currentOrigin() == origin else { rejection = "Context changed. This value was not applied."; return }
                    let candidate = JSONValue.bool(next)
                    guard field.accepts(candidate), apply(candidate) else { rejection = "This value was not applied. The component changed or rejected the edit."; return }
                    rejection = nil
                })).toggleStyle(.checkbox).font(.system(size: 12)).accessibilityLabel(field.title)
            }
            }
            if let rejection { Text(rejection).font(.system(size: 10)).foregroundStyle(MLTheme.rose) }
        }.disabled(inactiveReason != nil)
        .onChange(of: origin) { _, _ in rejection = nil;numberListPending=false }.onChange(of: value) { _, _ in rejection = nil }
        if let inactiveReason { Text(inactiveReason).font(.caption2).foregroundStyle(MLTheme.muted) }
    }
    private var inactiveReason: String? { componentID.flatMap { ComponentFieldApplicability.inactiveReason(componentID: $0, field: field.key, values: configuration) } }
}
