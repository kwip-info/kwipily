import KwipMLCore

/// One creation surface; saved/assessed primitive identities retain their exact contracts.
enum ActivationPalette {
    static let type="kwip.activation"
    static func functions(allowedTypes:Set<String>?=nil)->[ActivationConfiguration.Function] {
        ActivationConfiguration.Function.allCases.filter {allowedTypes?.contains($0.rawValue) ?? true}
    }
    static func defaultFunction(allowedTypes:Set<String>?=nil)->ActivationConfiguration.Function? {
        let offered=functions(allowedTypes:allowedTypes)
        return offered.contains(.relu) ? .relu:offered.first
    }
    static func consolidated(entries:[ComponentRibbonEntry])->[ComponentRibbonEntry] {
        let actual=entries.filter {ActivationConfiguration.componentIDs.contains($0.typeIcon)}
        guard !actual.isEmpty else {return entries}
        let allowed=Set(actual.map(\.typeIcon))
        guard let selected=defaultFunction(allowedTypes:allowed),let prototype=actual.first(where:{$0.typeIcon==selected.rawValue}) else {return entries}
        let entry=ComponentRibbonEntry(id:prototype.id,title:"Activation",shortTitle:"Activation",
            purpose:"Apply a selected activation function. "+functions(allowedTypes:allowed).map(\.title).joined(separator:", ")+".",
            category:"Activations",typeIcon:type)
        var inserted=entries.contains {$0.typeIcon==type}
        return entries.compactMap {value in
            guard ActivationConfiguration.componentIDs.contains(value.typeIcon) else {return value}
            guard !inserted else {return nil};inserted=true;return entry
        }
    }
    /// Consolidate only the already-eligible insertion catalog. The selected actual
    /// primitive keeps its ports, fields and ID for the existing atomic insertion command.
    static func consolidated(components:[ComponentDefinition])->[ComponentDefinition] {
        let allowed=Set(components.map(\.id)).intersection(ActivationConfiguration.componentIDs)
        guard let selected=defaultFunction(allowedTypes:allowed),var prototype=components.first(where:{$0.id==selected.rawValue}) else {return components}
        prototype.title="Activation"
        var inserted=false
        return components.compactMap {component in
            guard ActivationConfiguration.componentIDs.contains(component.id) else {return component}
            guard !inserted else {return nil};inserted=true;return prototype
        }
    }

}
