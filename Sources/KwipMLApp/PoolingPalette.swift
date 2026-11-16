import KwipMLCore

/// One creation surface; saved/assessed primitive identities retain their exact contracts.
enum PoolingPalette {
    static let type="kwip.pooling"
    static func variants(allowedTypes:Set<String>?=nil)->[PoolingConfiguration.Variant] {
        PoolingConfiguration.Variant.allCases.filter {allowedTypes?.contains($0.rawValue) ?? true}
    }
    static func defaultVariant(allowedTypes:Set<String>?=nil)->PoolingConfiguration.Variant? {
        let offered=variants(allowedTypes:allowedTypes)
        return offered.contains(.maxPool2d) ? .maxPool2d:offered.first
    }
    static func consolidated(entries:[ComponentRibbonEntry])->[ComponentRibbonEntry] {
        let actual=entries.filter {PoolingConfiguration.componentIDs.contains($0.typeIcon)}
        guard !actual.isEmpty else {return entries}
        let allowed=Set(actual.map(\.typeIcon))
        guard let selected=defaultVariant(allowedTypes:allowed),let prototype=actual.first(where:{$0.typeIcon==selected.rawValue}) else {return entries}
        let entry=ComponentRibbonEntry(id:prototype.id,title:"Pooling",shortTitle:"Pooling",
            purpose:"Apply a selected pooling operation. "+variants(allowedTypes:allowed).map(\.title).joined(separator:", ")+".",
            category:"Pooling",typeIcon:type)
        var inserted=entries.contains {$0.typeIcon==type}
        return entries.compactMap {value in
            guard PoolingConfiguration.componentIDs.contains(value.typeIcon) else {return value}
            guard !inserted else {return nil};inserted=true;return entry
        }
    }
    /// Consolidate only the already-eligible insertion catalog. The selected actual
    /// primitive keeps its ports, fields and ID for the existing atomic insertion command.
    static func consolidated(components:[ComponentDefinition])->[ComponentDefinition] {
        let allowed=Set(components.map(\.id)).intersection(PoolingConfiguration.componentIDs)
        guard let selected=defaultVariant(allowedTypes:allowed),var prototype=components.first(where:{$0.id==selected.rawValue}) else {return components}
        prototype.title="Pooling"
        var inserted=false
        return components.compactMap {component in
            guard PoolingConfiguration.componentIDs.contains(component.id) else {return component}
            guard !inserted else {return nil};inserted=true;return prototype
        }
    }

}
