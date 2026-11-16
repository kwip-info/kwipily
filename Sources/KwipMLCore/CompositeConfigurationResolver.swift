import Foundation

/// Resolves declared values on a copy without requiring a completed topology.
/// Products reference exposed integer parameters only, never expressions or other products.
public enum CompositeConfigurationResolver {
    public static func resolve(_ definition: CompositeDefinition, overrides: [String: JSONValue] = [:],
                               definitions: [CompositeDefinition], registry: ComponentRegistry) throws -> CompositeGraph {
        func invalid(_ message: String) -> CompositeContractError { .init(.invalidParameter, message) }
        let products = definition.integerProducts ?? []
        guard definition.parameters.count <= 128, products.count <= 128, definitions.count <= 64 else {
            throw CompositeContractError(.resourceLimit, "Block configuration binding limits exceeded.")
        }
        let keys = definition.parameters.map { $0.field.key }
        guard Set(keys).count == keys.count, keys.allSatisfy({ !$0.isEmpty && $0.unicodeScalars.count <= 128 }),
              Set(overrides.keys).isSubset(of: Set(keys)) else { throw invalid("Exposed parameter names must be unique; undeclared overrides are not accepted.") }
        var lookup: [CompositeDefinitionReference: CompositeDefinition] = [:]
        for item in definitions {
            guard lookup[item.reference] == nil else { throw invalid("Duplicate nested Block definition reference.") }
            lookup[item.reference] = item
        }
        var body = try CompositeMigration.toCurrentBody(definition.body, registry: registry)
        let identities = body.nodes.map(\.id) + body.instances.map(\.id)
        guard Set(identities).count == identities.count else { throw invalid("Configuration targets require unique component identities.") }
        var targeted = Set<CompositeParameterTarget>()
        var values: [String: JSONValue] = [:]
        var fields: [String: ComponentField] = [:]
        func targetField(_ target: CompositeParameterTarget) throws -> ComponentField {
            if let node = body.nodes.first(where: { $0.id == target.componentID }),
               let field = registry.component(type: node.type)?.fields.first(where: { $0.key == target.field }) { return field }
            if let instance = body.instances.first(where: { $0.id == target.componentID }),
               let nested = lookup[instance.definition] {
                let matches = nested.parameters.filter { $0.field.key == target.field }
                if matches.count == 1 { return matches[0].field }
            }
            throw invalid("Configuration binding targets an absent primitive field or nested exposed parameter.")
        }
        func assign(_ value: JSONValue, to target: CompositeParameterTarget) throws {
            guard targeted.insert(target).inserted else { throw invalid("An internal field cannot receive duplicate direct or derived bindings.") }
            if let index = body.nodes.firstIndex(where: { $0.id == target.componentID }) { body.nodes[index].config[target.field] = value }
            else if let index = body.instances.firstIndex(where: { $0.id == target.componentID }) { body.instances[index].overrides[target.field] = value }
            else { throw invalid("Missing configuration target.") }
        }
        for parameter in definition.parameters {
            let field = parameter.field, value = overrides[parameter.field.key] ?? parameter.field.defaultValue
            guard field.isValidDefinition, field.accepts(value), (1...256).contains(parameter.targets.count) else {
                throw invalid("Exposed parameter \(field.key) must match its declared type, range and bounded targets.")
            }
            values[field.key] = value; fields[field.key] = field
            for target in parameter.targets {
                let destination = try targetField(target)
                guard CompositeDefinitionValidation.canExpose(field, to: destination), destination.accepts(value) else {
                    throw invalid("Exposed parameter range or value is incompatible with its target.")
                }
                try assign(value, to: target)
            }
        }
        for product in products {
            guard (1...256).contains(product.targets.count),
                  let lhs = fields[product.lhs], let rhs = fields[product.rhs],
                  lhs.kind == .integer, rhs.kind == .integer, !lhs.allowsNull, !rhs.allowsNull,
                  let left = values[product.lhs]?.integerValue, let right = values[product.rhs]?.integerValue else {
                throw invalid("Integer products require two declared nonnullable integer parameters and bounded targets.")
            }
            let (value, overflow) = left.multipliedReportingOverflow(by: right)
            guard !overflow else { throw invalid("Integer product exceeds the supported integer range.") }
            let encoded = JSONValue.integer(Int64(value))
            for target in product.targets {
                let destination = try targetField(target)
                guard destination.isValidDefinition, destination.kind == .integer, !destination.allowsNull, destination.accepts(encoded) else {
                    throw invalid("Integer product must fit its nonnullable integer target's bounds.")
                }
                try assign(encoded, to: target)
            }
        }
        return body
    }
}
