import Foundation

/// Disposable derived UI work only. Inputs are compared before reuse; nothing is
/// persisted or used as scientific/provenance authority.
final class BoundedPresentationCache<Input:Equatable,Output> {
    private struct Entry {let input:Input;let output:Output;let cost:Int}
    private let lock=NSLock()
    private let maximumEntries:Int,maximumCost:Int
    private var entries:[String:Entry]=[:],order:[String]=[]
    private var cost=0
    init(maximumEntries:Int,maximumCost:Int) {self.maximumEntries=max(1,maximumEntries);self.maximumCost=max(1,maximumCost)}
    func value(key:String,input:Input,cost requestedCost:Int,build:()->Output)->Output {
        lock.lock();defer {lock.unlock()}
        if let entry=entries[key],entry.input==input {return entry.output}
        let output=build()
        if let old=entries.removeValue(forKey:key) {cost-=old.cost;order.removeAll {$0==key}}
        let entryCost=max(1,requestedCost)
        guard entryCost<=maximumCost else {return output}
        while !order.isEmpty && (entries.count>=maximumEntries || cost>maximumCost-entryCost) {
            let expired=order.removeFirst()
            if let entry=entries.removeValue(forKey:expired) {cost-=entry.cost}
        }
        entries[key]=Entry(input:input,output:output,cost:entryCost);order.append(key);cost+=entryCost
        return output
    }
    var retainedCount:Int {lock.lock();defer {lock.unlock()};return entries.count}
    var retainedCost:Int {lock.lock();defer {lock.unlock()};return cost}
}
