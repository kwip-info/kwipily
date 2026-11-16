import Foundation

public enum CompositeBindingEditing {
    /// Stop deriving selected fields while preserving their current default-context values.
    /// Other direct/derived bindings and immutable source definitions are untouched.
    public static func releasingIntegerProduct(at index: Int, from definition: CompositeDefinition,
        definitions: [CompositeDefinition], registry: ComponentRegistry) throws -> CompositeDefinition {
        guard let products = definition.integerProducts, products.indices.contains(index) else {
            throw CompositeContractError(.invalidParameter, "The derived binding no longer exists.")
        }
        let resolved = try CompositeConfigurationResolver.resolve(definition, definitions: definitions, registry: registry)
        var result = definition
        for target in products[index].targets {
            if let i = result.body.nodes.firstIndex(where: { $0.id == target.componentID }),
               let value = resolved.nodes.first(where: { $0.id == target.componentID })?.config[target.field] {
                result.body.nodes[i].config[target.field] = value
            } else if let i = result.body.instances.firstIndex(where: { $0.id == target.componentID }),
                      let value = resolved.instances.first(where: { $0.id == target.componentID })?.overrides[target.field] {
                result.body.instances[i].overrides[target.field] = value
            } else { throw CompositeContractError(.invalidParameter, "The derived target is unavailable.") }
        }
        result.integerProducts?.remove(at: index)
        if result.integerProducts?.isEmpty == true { result.integerProducts = nil }
        return result
    }
}
