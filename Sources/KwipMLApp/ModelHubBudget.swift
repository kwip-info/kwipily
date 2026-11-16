import Foundation
import KwipMLCore

/// One captured deadline covers the entire download, not a fresh allowance per range.
enum ModelHubBudget {
    static func read(remainingSeconds:Double,operation:@escaping @Sendable ()async throws->Data)async throws->Data {
        guard remainingSeconds.isFinite,remainingSeconds>0 else {throw WorkflowArchiveError("The model download exhausted its captured time allowance.")}
        return try await withThrowingTaskGroup(of:Data.self) {group in
            group.addTask {try await operation()}
            group.addTask {
                try await Task.sleep(for:.seconds(remainingSeconds))
                throw WorkflowArchiveError("The model download exhausted its captured time allowance. Increase the execution time allowance or retry on a faster connection.")
            }
            defer {group.cancelAll()}
            guard let value=try await group.next() else {throw CancellationError()}
            return value
        }
    }
}
