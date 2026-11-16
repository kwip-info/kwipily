import Foundation
import CoreFoundation

/// Measured population counts, independent of the bounded table preview.
struct AggregateResultSummary {
    struct Reducer {let output:String,op:String,nullInputRows:Int}
    let inputRows:Int,outputRows:Int,droppedKeyRows:Int
    let reducers:[Reducer]
    var retainedRows:Int {inputRows-droppedKeyRows}
    init?(evidence:[String:Any]) {
        func count(_ value:Any?)->Int? {
            guard let value=value as? NSNumber,CFGetTypeID(value) != CFBooleanGetTypeID(),value.doubleValue.isFinite,
                  value.doubleValue.rounded(.towardZero)==value.doubleValue,(0...100000).contains(value.doubleValue) else {return nil}
            return value.intValue
        }
        guard evidence["kind"] as? String == "aggregate",let input=count(evidence["inputRows"]),
              let output=count(evidence["outputRows"]),let dropped=count(evidence["droppedKeyRows"]),
              input>0,output>0,dropped<input,output<=input-dropped else {return nil}
        var measured:[Reducer]=[],names=Set<String>()
        if let raw=evidence["reducers"] {
            guard let entries=raw as? [[String:Any]],entries.count<=128 else {return nil}
            for entry in entries {
                guard let name=entry["output"] as? String,!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.count<=200,names.insert(name).inserted,
                      let op=entry["op"] as? String,["countRows","countNonNull","sum","sumDecimal","mean","min","max","joinText"].contains(op),
                      let nulls=count(entry["nullInputRows"]),nulls<=input-dropped,op != "countRows" || nulls==0 else {return nil}
                measured.append(.init(output:name,op:op,nullInputRows:nulls))
            }
        }
        inputRows=input;outputRows=output;droppedKeyRows=dropped;reducers=measured
    }
}
