import KwipMLCore

enum TensorOperationPalette {
    static let consolidatedFamilies:[TensorOperationAuthoring.Family]=[.axes,.parts]
    static let replacedTypes=Set(consolidatedFamilies.flatMap(\.types))
    static func type(_ family:TensorOperationAuthoring.Family)->String {"kwip.tensor."+family.rawValue}
    static func consolidated(entries:[ComponentRibbonEntry])->[ComponentRibbonEntry] {
        var result=entries
        for family in consolidatedFamilies {
            let actual=result.filter {family.types.contains($0.typeIcon)}
            guard let selected=family.types.first(where:{type in actual.contains {$0.typeIcon==type}}),let prototype=actual.first(where:{$0.typeIcon==selected}) else {continue}
            let combined=ComponentRibbonEntry(id:prototype.id,title:family.title,shortTitle:family.title,purpose:actual.map(\.title).joined(separator:", ")+". "+family.types.filter {type in actual.contains {$0.typeIcon==type}}.map(TensorOperationAuthoring.methodTitle).joined(separator:", ")+". Preserve independent observations.",category:prototype.category,typeIcon:type(family))
            var inserted=result.contains {$0.typeIcon==type(family)}
            result=result.compactMap {value in guard family.types.contains(value.typeIcon) else {return value};guard !inserted else {return nil};inserted=true;return combined}
        }
        return result
    }
    static func consolidated(components:[ComponentDefinition])->[ComponentDefinition] {
        var result=components
        for family in consolidatedFamilies {
            guard let selected=family.types.first(where:{type in result.contains {$0.id==type}}),var prototype=result.first(where:{$0.id==selected}) else {continue}
            prototype.title=family.title;var inserted=false
            result=result.compactMap {value in guard family.types.contains(value.id) else {return value};guard !inserted else {return nil};inserted=true;return prototype}
        }
        return result
    }
}
