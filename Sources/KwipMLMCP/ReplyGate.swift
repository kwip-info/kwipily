import Foundation

/// XPC interruption and reply can race. Resolve a one-shot callback exactly once.
public final class MCPReplyGate<Value>:@unchecked Sendable {
    private let lock=NSLock()
    private var callback:((Value)->Void)?
    public init(_ callback:@escaping(Value)->Void){self.callback=callback}
    public func resolve(_ value:Value){lock.lock();let next=callback;callback=nil;lock.unlock();next?(value)}
}
