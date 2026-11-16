import KwipMLCore

/// One creation surface; saved/assessed primitive identities retain their exact contracts.
enum RecurrentCellPalette {
    static let type="kwip.recurrentCell"
    static func methods(allowedTypes:Set<String>?=nil)->[RecurrentCellConfiguration.Method] {
        RecurrentCellConfiguration.Method.allCases.filter {allowedTypes?.contains($0.rawValue+"_cell") ?? true}
    }
    static func defaultMethod(allowedTypes:Set<String>?=nil)->RecurrentCellConfiguration.Method? {
        let offered=methods(allowedTypes:allowedTypes)
        return offered.contains(.rnn) ? .rnn:offered.first
    }
    static func consolidated(entries:[ComponentRibbonEntry])->[ComponentRibbonEntry] {
        let actual=entries.filter {RecurrentCellConfiguration.componentIDs.contains($0.typeIcon)}
        guard !actual.isEmpty else {return entries}
        let allowed=Set(actual.map(\.typeIcon))
        guard let selected=defaultMethod(allowedTypes:allowed),let prototype=actual.first(where:{$0.typeIcon==selected.rawValue+"_cell"}) else {return entries}
        let entry=ComponentRibbonEntry(id:prototype.id,title:"Recurrent Cell",shortTitle:"Recurrent Cell",
            purpose:"Process one step with explicit hidden/cell state. "+methods(allowedTypes:allowed).map(\.title).joined(separator:", ")+". "+actual.map(\.title).joined(separator:", ")+". "+methods(allowedTypes:allowed).map {$0.rawValue.replacingOccurrences(of:"_",with:"")}.joined(separator:", ")+".",
            category:"Layers",typeIcon:type)
        var inserted=entries.contains {$0.typeIcon==type}
        return entries.compactMap {value in
            guard RecurrentCellConfiguration.componentIDs.contains(value.typeIcon) else {return value}
            guard !inserted else {return nil};inserted=true;return entry
        }
    }
    /// Consolidate only the already-eligible insertion catalog. The selected actual
    /// primitive keeps its ports, fields and ID for the existing atomic insertion command.
    static func consolidated(components:[ComponentDefinition])->[ComponentDefinition] {
        let allowed=Set(components.map(\.id)).intersection(RecurrentCellConfiguration.componentIDs)
        guard let selected=defaultMethod(allowedTypes:allowed),var prototype=components.first(where:{$0.id==selected.rawValue+"_cell"}) else {return components}
        prototype.title="Recurrent Cell"
        var inserted=false
        return components.compactMap {component in
            guard RecurrentCellConfiguration.componentIDs.contains(component.id) else {return component}
            guard !inserted else {return nil};inserted=true;return prototype
        }
    }

}
