import Foundation
import KwipMLCore

/// Old progress IDs remain readable; retired recipe construction is not a current lesson surface.
enum RetiredTrainingLessonDestination {
    static let operations:Set<String> = Set(OptimizerLearningTask.allCases.map(\.rawValue))
        .union(SchedulerLearningTask.allCases.map(\.rawValue))
        .union(LossLearningTask.allCases.map(\.rawValue))
        .union(GradientLearningTask.allCases.map(\.rawValue))
        .union(InitializationLearningTask.allCases.map(\.rawValue))
        .union(DataLoadingLearningTask.allCases.map(\.rawValue))
    static func contains(_ operation:String?) -> Bool {operation.map(operations.contains) ?? false}
}
