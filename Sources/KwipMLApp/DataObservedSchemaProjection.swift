import KwipMLCore
/// Cached-only presentation. Invocation/archive validation owns evidence acceptance.
enum DataObservedSchemaProjection {
    static func columns(_ port:JSONValue,isCurrent:Bool)->[DataColumnSchema]? {
        guard isCurrent,case .object(let output)=port,case .object(let preview)=output["preview"],case .array(let columns)=preview["columns"] else {return nil}
        var result:[DataColumnSchema]=[]
        for value in columns {
            guard case .object(let fields)=value,case .string(let name)=fields["name"],case .string(let dtype)=fields["dtype"],case .bool(let nullable)=fields["nullable"] else {return nil}
            result.append(.init(name:name,dtype:dtype,nullable:nullable))
        }
        return result
    }
}
