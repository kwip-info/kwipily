import Foundation

/// Empty configuration means every current/future source column, never zero columns.
enum TableOutputColumnSelection {
    static func effective(_ configured:[String],available:[String])->[String] {
        configured.isEmpty ? available:configured
    }
    static func removable(_ name:String,configured:[String],available:[String])->[String]? {
        let current=effective(configured,available:available)
        guard current.count>1,current.contains(name) else {return nil}
        return current.filter {$0 != name}
    }
    static func moving(_ name:String,to target:Int,configured:[String],available:[String])->[String]? {
        var current=effective(configured,available:available)
        guard let index=current.firstIndex(of:name),current.indices.contains(target),target != index else {return nil}
        current.remove(at:index);current.insert(name,at:target);return current
    }
}
