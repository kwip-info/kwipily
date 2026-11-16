import Foundation
import CoreFoundation

/// Counts from complete captured outputs; never extrapolates from preview samples.
struct PartitionResultSummary {
    struct Partition {let role:String,rows:Int,groups:Int?}
    let inputRows:Int,excludedRows:Int
    let partitions:[Partition]
    let grouped:Bool
    let temporal:Bool
    init?(evidence:[String:Any]) {
        func integer(_ value:Any?,range:ClosedRange<Int64>)->Int64? {
            guard let number=value as? NSNumber,CFGetTypeID(number) != CFBooleanGetTypeID(),
                  let exact=Int64(number.stringValue),range.contains(exact) else {return nil}
            return exact
        }
        func count(_ value:Any?)->Int? {integer(value,range:0...5000000).map(Int.init)}
        guard evidence["kind"] as? String == "partition",let algorithm=evidence["algorithm"] as? String,
              ["kwip.groupSplit.v1","kwip.temporalSplit.v1","kwip.partition.random.v1","kwip.partition.entity.v1","kwip.partition.time.v1"].contains(algorithm),
              let input=count(evidence["inputRows"]),input>0,let excluded=count(evidence["excludedRows"]),
              let entries=evidence["partitions"] as? [[String:Any]],entries.count==3 else {return nil}
        let isGroup=["kwip.groupSplit.v1","kwip.partition.entity.v1"].contains(algorithm)
        guard !isGroup || excluded==0 else {return nil}
        var values:[Partition]=[],previousTime:Int64?
        for (entry,role) in zip(entries,["training","validation","test"]) {
            guard entry["role"] as? String == role,let rows=count(entry["rows"]),rows>0 else {return nil}
            var groups:Int?
            if isGroup {
                guard let measured=count(entry["groups"]),measured>0,measured<=rows else {return nil}
                groups=measured
            } else if ["kwip.temporalSplit.v1","kwip.partition.time.v1"].contains(algorithm) {
                let range:ClosedRange<Int64> = -62135596800000000...253402300799999999
                guard let minimum=integer(entry["minTimeMicroseconds"],range:range),let maximum=integer(entry["maxTimeMicroseconds"],range:range),minimum<=maximum,
                      previousTime.map({minimum>$0}) ?? true else {return nil}
                previousTime=maximum
            }
            values.append(.init(role:role,rows:rows,groups:groups))
        }
        guard values.reduce(excluded,{$0+$1.rows})==input else {return nil}
        inputRows=input;excludedRows=excluded;partitions=values;grouped=isGroup;temporal=["kwip.temporalSplit.v1","kwip.partition.time.v1"].contains(algorithm)
    }
}
