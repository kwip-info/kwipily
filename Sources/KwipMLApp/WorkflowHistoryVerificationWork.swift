import Foundation

/// Debug builds of the deeply nested archive verifier can exceed the cooperative
/// executor's small stack. Keep verification off the UI actor on a short-lived,
/// explicitly bounded stack; cancellation never publishes a successful result.
enum WorkflowHistoryVerificationWork {
    private final class Cancellation:@unchecked Sendable {
        let lock=NSLock()
        var cancelled=false
        func cancel() {lock.lock();cancelled=true;lock.unlock()}
        var isCancelled:Bool {lock.lock();defer {lock.unlock()};return cancelled}
    }
    static func run<Value>(discard:@escaping (Value)->Void = {_ in},_ operation:@escaping ()throws->Value)async throws->Value {
        try await runCancellable(discard:discard) {_ in try operation()}
    }
    /// Blocking recovery also needs cancellation inside its transaction, before
    /// a ledger mutation; Task cancellation does not propagate into a raw Thread.
    static func runCancellable<Value>(discard:@escaping (Value)->Void = {_ in},_ operation:@escaping (@escaping ()throws->Void)throws->Value)async throws->Value {
        try Task.checkCancellation()
        let cancellation=Cancellation()
        let value:Value=try await withTaskCancellationHandler(operation:{
            try await withCheckedThrowingContinuation {continuation in
                let thread=Thread {
                    let result:Result<Value,Error>
                    do {
                        guard !cancellation.isCancelled else {throw CancellationError()}
                        let value=try operation {if cancellation.isCancelled {throw CancellationError()}}
                        if cancellation.isCancelled {discard(value);throw CancellationError()}
                        result = .success(value)
                    } catch {result = .failure(error)}
                    continuation.resume(with:result)
                }
                thread.name="Kwip ML archive verification"
                thread.qualityOfService = .userInitiated
                thread.stackSize=8*1024*1024
                thread.start()
            }
        },onCancel:{cancellation.cancel()})
        do {try Task.checkCancellation();return value}
        catch {discard(value);throw error}
    }
}
