import Foundation
import KwipMLCore

/// Explicit per-operation allowances; no policy is saved until Apply.
struct ExecutionWorkloadDraft: Equatable {
    let allowsInheritance: Bool
    private let original: ExecutionWorkloadPolicy?
    var selection: String
    var values: [String: String]

    init(application: ExecutionWorkloadPolicy?, project: ExecutionWorkloadPolicy?, isProject: Bool) {
        allowsInheritance = isProject
        original = isProject ? project : application
        selection = original?.preset.rawValue ?? (isProject ? "inherit" : "legacy")
        let seed = original ?? application ?? (try? ExecutionWorkloadPolicy(preset: .legacy))
        values = seed?.limits.mapValues(String.init) ?? [:]
    }

    mutating func choose(_ value: String) throws {
        if value == "inherit", allowsInheritance { selection = value; return }
        guard let preset = ExecutionWorkloadPolicy.Preset(rawValue: value) else {
            throw ExecutionBudgetError.invalid("Choose a workload profile.")
        }
        if preset != .custom { values = try ExecutionWorkloadPolicy(preset: preset).limits.mapValues(String.init) }
        selection = value
    }

    func policy() throws -> ExecutionWorkloadPolicy? {
        if selection == "inherit", allowsInheritance { return nil }
        guard let preset = ExecutionWorkloadPolicy.Preset(rawValue: selection) else {
            throw ExecutionBudgetError.invalid("Choose a workload profile.")
        }
        let policy: ExecutionWorkloadPolicy
        if preset == .custom {
            var limits: [String: Int64] = [:]
            for setting in ExecutionWorkloadPolicy.Setting.allCases {
                let text = values[setting.rawValue, default: ""].trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty, text.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }), let value = Int64(text) else {
                    throw ExecutionBudgetError.invalid("Workload allowances must be positive whole numbers.")
                }
                limits[setting.rawValue] = value
            }
            policy = try .init(preset: .custom, limits: limits)
        } else { policy = try .init(preset: preset) }
        // Opening/saving an unrelated field must not migrate old preferences.
        if !allowsInheritance, original == nil, preset == .legacy { return nil }
        return policy
    }
}
