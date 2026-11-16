import KwipMLCore

/// One creation surface; saved/assessed primitive identities retain their exact contracts.
enum ReshapePalette {
    static let type="kwip.reshape"
    static func variants(allowedTypes:Set<String>?=nil)->[ReshapeAuthoringConfiguration.Variant] {
        ReshapeAuthoringConfiguration.Variant.allCases.filter {allowedTypes?.contains($0.rawValue) ?? true}
    }
    static func defaultVariant(allowedTypes:Set<String>?=nil)->ReshapeAuthoringConfiguration.Variant? {
        let offered=variants(allowedTypes:allowedTypes)
        return offered.contains(.dimensions) ? .dimensions:offered.first
    }
    static func consolidated(entries:[ComponentRibbonEntry])->[ComponentRibbonEntry] {
        let actual=entries.filter {ReshapeAuthoringConfiguration.componentIDs.contains($0.typeIcon)}
        guard !actual.isEmpty else {return entries}
        let allowed=Set(actual.map(\.typeIcon))
        guard let selected=defaultVariant(allowedTypes:allowed),let prototype=actual.first(where:{$0.typeIcon==selected.rawValue}) else {return entries}
        let entry=ComponentRibbonEntry(id:prototype.id,title:"Reshape",shortTitle:"Reshape",
            purpose:"Change sample dimensions while preserving batch and values. "+variants(allowedTypes:allowed).map(\.title).joined(separator:", ")+". "+actual.map(\.title).joined(separator:", ")+". "+variants(allowedTypes:allowed).map {$0.rawValue.replacingOccurrences(of:"_",with:"")}.joined(separator:", ")+".",
            category:"Tensor layout",typeIcon:type)
        var inserted=entries.contains {$0.typeIcon==type}
        return entries.compactMap {value in
            guard ReshapeAuthoringConfiguration.componentIDs.contains(value.typeIcon) else {return value}
            guard !inserted else {return nil};inserted=true;return entry
        }
    }
    /// Consolidate only the already-eligible insertion catalog. The selected actual
    /// primitive keeps its ports, fields and ID for the existing atomic insertion command.
    static func consolidated(components:[ComponentDefinition])->[ComponentDefinition] {
        let allowed=Set(components.map(\.id)).intersection(ReshapeAuthoringConfiguration.componentIDs)
        guard let selected=defaultVariant(allowedTypes:allowed),var prototype=components.first(where:{$0.id==selected.rawValue}) else {return components}
        prototype.title="Reshape"
        var inserted=false
        return components.compactMap {component in
            guard ReshapeAuthoringConfiguration.componentIDs.contains(component.id) else {return component}
            guard !inserted else {return nil};inserted=true;return prototype
        }
    }

}
