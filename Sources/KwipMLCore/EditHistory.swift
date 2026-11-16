import Foundation

/// Value snapshots keep UI edits reversible without coupling history to views.
public struct EditHistory<Value: Equatable> {
    private var past: [Value] = []
    private var future: [Value] = []
    public init() {}
    public var canUndo: Bool { !past.isEmpty }
    public var canRedo: Bool { !future.isEmpty }
    public mutating func record(before: Value, after: Value) {
        guard before != after else { return }
        past.append(before)
        if past.count > 100 { past.removeFirst() }
        future.removeAll()
    }
    public mutating func undo(current: Value) -> Value? {
        guard let value = past.popLast() else { return nil }
        future.append(current)
        return value
    }
    public mutating func redo(current: Value) -> Value? {
        guard let value = future.popLast() else { return nil }
        past.append(current)
        return value
    }
}
