import KwipMLCore

enum RegistryFieldCompatibility {
    static func normalized(_ value: JSONValue, for field: ComponentField) -> JSONValue? {
        guard field.isValidDefinition, field.accepts(value) else { return nil }
        if value == .null {return .null}
        switch field.kind {
        case .numberList:
            guard case .array(let items)=value else {return nil}
            return .array(items.compactMap { $0.doubleValue.map(JSONValue.number) })
        case .integerList:
            guard case .array(let items)=value else {return nil}
            return .array(items.compactMap {item in item.integerValue.map {.integer(Int64($0))}})
        case .number: return value.doubleValue.map(JSONValue.number)
        case .boolean, .choice, .text: return value
        case .integer: return value.integerValue.map { .integer(Int64($0)) }
        }
    }
    static func canExpose(_ field: ComponentField, to target: ComponentField) -> Bool {
        CompositeDefinitionValidation.canExpose(field, to: target)
    }
}
