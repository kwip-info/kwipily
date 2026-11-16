import Foundation

/// Ordered table columns fill each model sample in row-major order.
/// Structural validity is independent of the selected model's shape and dtype.
public enum TrainingColumnMapping {
    public static let maximumColumns=128
    public static func validate(features:[String],target:String)throws {
        guard (1...maximumColumns).contains(features.count),Set(features).count==features.count,
              !features.contains(target),
              (features+[target]).allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count<=200}) else {
            throw DataFlowContractError("Select 1–128 distinct ordered input columns and a separate target, each with a nonblank name of at most 200 characters.")
        }
    }
}
