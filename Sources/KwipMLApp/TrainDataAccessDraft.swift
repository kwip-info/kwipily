import Foundation
import KwipMLCore

/// Inspector-only text drafts. Omitted policies stay omitted until explicitly enabled.
struct TrainDataAccessDraft: Equatable {
    var accessEnabled:Bool
    var batchingEnabled:Bool
    var kind:String
    var indices:String
    var maximumRows:String
    var batchSize:String
    var shuffle:Bool
    var dropLast:Bool
    var seed:String
    var workers:String
    var prefetch:String
    var timeout:String
    var collation:String
    init(configuration:[String:JSONValue]) throws {
        let access:DatasetConfiguration
        if let value=configuration["datasetAccess"] {
            guard case .object(let values)=value else {throw DataFlowContractError("Data access must be a complete configuration.")}
            access=try .init(values:values);accessEnabled=true
        } else {access=try .init(values:DatasetConfiguration.defaults);accessEnabled=false}
        let batching:DataLoaderConfiguration
        if let value=configuration["batching"] {
            guard case .object(let values)=value else {throw DataFlowContractError("Batching must be a complete configuration.")}
            batching=try .init(values:values);batchingEnabled=true
        } else {batching=try .init(values:DataLoaderConfiguration.defaults);batchingEnabled=false}
        kind=access.kind;indices=access.indices.map(String.init).joined(separator:", ");maximumRows=String(access.maximumRows)
        batchSize=String(batching.batchSize);shuffle=batching.shuffle;dropLast=batching.dropLast;seed=String(batching.seed)
        workers=String(batching.numberOfWorkers);prefetch=String(batching.prefetchFactor ?? 1);timeout=String(batching.timeoutSeconds);collation=batching.collation
    }
    func configuration(preserving original:[String:JSONValue]) throws -> [String:JSONValue] {
        func integer(_ text:String,_ name:String)throws->Int64 {
            guard let value=Int64(text.trimmingCharacters(in:.whitespacesAndNewlines)) else {throw DataFlowContractError("\(name) requires an exact whole number.")}
            return value
        }
        var result=original
        if accessEnabled {
            let selected:[JSONValue]
            if kind == "subset" {
                let parts=indices.split(whereSeparator:{$0 == "," || $0.isWhitespace})
                selected=try parts.map{.integer(try integer(String($0),"Selected position"))}
            } else {selected=[]}
            let values:[String:JSONValue] = ["kind":.string(kind),"indices":.array(selected),"max_rows":.integer(kind == "boundedIterable" ? try integer(maximumRows,"Maximum training rows"):100000)]
            result["datasetAccess"] = .object(try DatasetConfiguration(values:values).values)
        } else {result.removeValue(forKey:"datasetAccess")}
        if batchingEnabled {
            let count=try integer(workers,"Reader processes")
            guard count == 0 || (Double(timeout)?.isFinite == true) else {throw DataFlowContractError("Timeout requires a finite number of seconds.")}
            let values:[String:JSONValue] = ["batch_size":.integer(try integer(batchSize,"Batch size")),"shuffle":.bool(shuffle),"drop_last":.bool(dropLast),"seed":.integer(try integer(seed,"Batch ordering seed")),"num_workers":.integer(count),"prefetch_factor":count == 0 ? .null:.integer(try integer(prefetch,"Prefetch")),"timeout":.number(count == 0 ? 0:Double(timeout)!),"collation":.string(collation)]
            result["batching"] = .object(try DataLoaderConfiguration(values:values).values)
        } else {result.removeValue(forKey:"batching")}
        if accessEnabled,kind == "boundedIterable",batchingEnabled,shuffle {throw DataFlowContractError("A finite prefix keeps source order. Turn off Shuffle or choose indexed rows.")}
        return result
    }
}
