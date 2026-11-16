import Foundation
import CoreFoundation

/// Explanatory counts from measured join evidence, never inferred from a preview sample.
struct JoinResultSummary {
    let leftRows:Int, rightRows:Int, matchedPairs:Int, outputRows:Int
    let leftUnmatched:Int, rightUnmatched:Int
    let retainedUnmatchedRows:Int, discardedUnmatchedRows:Int
    init?(evidence:[String:Any]) {
        guard evidence["kind"] as? String == "join",let mode=evidence["mode"] as? String,
              ["inner","left","right","full"].contains(mode) else {return nil}
        func count(_ key:String)->Int? {
            guard let value=evidence[key] as? NSNumber,CFGetTypeID(value) != CFBooleanGetTypeID(),
                  value.doubleValue.isFinite,value.doubleValue.rounded(.towardZero)==value.doubleValue,
                  (0...100000).contains(value.doubleValue) else {return nil}
            return value.intValue
        }
        guard let left=count("leftRows"),let right=count("rightRows"),let pairs=count("matchedPairs"),
              let output=count("outputRows"),let lu=count("leftUnmatched"),let ru=count("rightUnmatched"),
              lu<=left,ru<=right else {return nil}
        let kept=(["left","full"].contains(mode) ? lu:0)+(["right","full"].contains(mode) ? ru:0)
        guard output==pairs+kept else {return nil}
        leftRows=left;rightRows=right;matchedPairs=pairs;outputRows=output
        leftUnmatched=lu;rightUnmatched=ru;retainedUnmatchedRows=kept;discardedUnmatchedRows=lu+ru-kept
    }
}
