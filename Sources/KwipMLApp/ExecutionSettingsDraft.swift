import Foundation
import KwipMLCore

/// Text edits stay local until validated and explicitly applied. Binary units
/// round-trip exact persisted bytes, including values not aligned to a MiB.
enum ExecutionSettingsScope { case all, execution, storage }

struct ExecutionSettingsDraft: Equatable {
    var workload: ExecutionWorkloadDraft
    private let applicationCapture: TrainingSampleCapturePolicy
    private let projectCapture: TrainingSampleCapturePolicy?
    private let applicationDerivativeCapture: DerivativeCapturePolicy
    private let projectDerivativeCapture: DerivativeCapturePolicy?
    var seconds: String
    var threads: String
    var memoryGiB: String
    var storageGiB: String
    var preparationCacheMaximum: Int { ExecutionPreferences.preparationCacheByteBounds(workload: try? workload.policy()).upperBound }
    var preparationCacheBytes: String
    var overrideSeconds: Bool
    var overrideThreads: Bool
    var memoryMode: String
    var storageLimited: Bool

    init(global: ExecutionPreferences, project: ProjectExecutionOverrides? = nil) {
        workload = .init(application: global.workloadPolicy, project: project?.workloadPolicy, isProject: project != nil)
        applicationCapture = global.sampleCapture
        projectCapture = project?.sampleCapture
        applicationDerivativeCapture = global.derivativeCapture
        projectDerivativeCapture = project?.derivativeCapture
        seconds = String(project?.maximumRunSeconds ?? global.maximumRunSeconds)
        threads = String(project?.cpuThreads ?? global.cpuThreads)
        overrideSeconds = project?.maximumRunSeconds != nil
        overrideThreads = project?.cpuThreads != nil
        var memory = global.maximumResidentBytes
        if let project {
            switch project.memory {
            case .inherit: memoryMode = "inherit"
            case .unlimited: memoryMode = "unlimited"
            case .limit(let bytes): memoryMode = "limit"; memory = bytes
            }
        } else { memoryMode = memory == nil ? "unlimited" : "limit" }
        memoryGiB = Self.amount(Int64(memory ?? 1_073_741_824))
        storageGiB = Self.amount(global.managedStorageLimitBytes ?? 10_737_418_240)
        storageLimited = global.managedStorageLimitBytes != nil
        preparationCacheBytes = String(global.preparationCacheLimitBytes)
    }
    func application() throws -> ExecutionPreferences {
        guard ["unlimited", "limit"].contains(memoryMode) else { throw invalid("Choose a resident-memory mode.") }
        return try .init(maximumRunSeconds: integer(seconds, title: "Time allowance"),
            cpuThreads: integer(threads, title: "CPU threads"),
            maximumResidentBytes: memoryMode == "limit" ? Int(bytes(memoryGiB, title: "Resident memory")) : nil,
            managedStorageLimitBytes: storageLimited ? bytes(storageGiB, title: "Managed storage") : nil, sampleCapture: applicationCapture, preparationCacheLimitBytes: integer(preparationCacheBytes,title:"Preparation cache"), derivativeCapture: applicationDerivativeCapture, workloadPolicy: workload.policy())
    }
    func application(merging latest: ExecutionPreferences, scope: ExecutionSettingsScope) throws -> ExecutionPreferences {
        let edited = try application()
        return try .init(maximumRunSeconds: scope == .storage ? latest.maximumRunSeconds : edited.maximumRunSeconds,
            cpuThreads: scope == .storage ? latest.cpuThreads : edited.cpuThreads,
            maximumResidentBytes: scope == .storage ? latest.maximumResidentBytes : edited.maximumResidentBytes,
            managedStorageLimitBytes: scope == .execution ? latest.managedStorageLimitBytes : edited.managedStorageLimitBytes,
            sampleCapture: latest.sampleCapture,
            preparationCacheLimitBytes: scope == .execution ? latest.preparationCacheLimitBytes : edited.preparationCacheLimitBytes, derivativeCapture: latest.derivativeCapture, workloadPolicy: scope == .storage ? latest.workloadPolicy : edited.workloadPolicy)
    }
    func project() throws -> ProjectExecutionOverrides {
        let memory: ProjectExecutionOverrides.MemoryOverride
        switch memoryMode {
        case "inherit": memory = .inherit
        case "unlimited": memory = .unlimited
        case "limit": memory = .limit(Int(try bytes(memoryGiB, title: "Resident memory")))
        default: throw invalid("Choose a resident-memory mode.")
        }
        return try .init(maximumRunSeconds: overrideSeconds ? integer(seconds, title: "Time allowance") : nil,
            cpuThreads: overrideThreads ? integer(threads, title: "CPU threads") : nil, memory: memory, sampleCapture: projectCapture, derivativeCapture: projectDerivativeCapture, workloadPolicy: workload.policy())
    }
    private static func amount(_ bytes: Int64) -> String {
        let unit: Int64 = 1_073_741_824
        let whole = bytes / unit, remainder = bytes % unit
        guard remainder != 0 else { return String(whole) }
        let fraction = NSDecimalNumber(decimal: Decimal(remainder) / Decimal(unit)).stringValue
        return String(whole) + String(fraction.dropFirst())
    }
    private func integer(_ raw: String, title: String) throws -> Int {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.range(of: "^[0-9]+$", options: .regularExpression) != nil, let value = Int(text) else {
            throw invalid("\(title) must be a whole number.")
        }
        return value
    }
    private func bytes(_ raw: String, title: String) throws -> Int64 {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text.range(of: "^(?:[0-9]+(?:\\.[0-9]*)?|\\.[0-9]+)$", options: .regularExpression) != nil,
              !text.isEmpty else {
            throw invalid("\(title) must be a positive decimal in GiB, using a period.")
        }
        let parts = text.split(separator: ".", omittingEmptySubsequences: false)
        guard let whole = Decimal(string: parts[0].isEmpty ? "0" : String(parts[0]), locale: Locale(identifier: "en_US_POSIX")) else {
            throw invalid("\(title) is too large.")
        }
        var digits = parts.count == 2 ? String(parts[1]) : ""
        while digits.last == "0" { digits.removeLast() }
        guard digits.count <= 30 else { throw invalid("\(title) has more precision than a whole byte supports.") }
        let fractionText = digits.isEmpty ? "0" : "0." + digits
        guard let fraction = Decimal(string: fractionText, locale: Locale(identifier: "en_US_POSIX")) else {
            throw invalid("\(title) has invalid precision.")
        }
        let scaledFraction = fraction * Decimal(1_073_741_824)
        let fractionalBytes = NSDecimalNumber(decimal: scaledFraction).int64Value
        guard scaledFraction >= 0, scaledFraction < Decimal(1_073_741_824), Decimal(fractionalBytes) == scaledFraction else {
            throw invalid("\(title) must resolve to a whole number of bytes.")
        }
        let scaled = whole * Decimal(1_073_741_824) + Decimal(fractionalBytes)
        guard !scaled.isNaN, scaled <= Decimal(Int64.max) else { throw invalid("\(title) is too large.") }
        let value = NSDecimalNumber(decimal: scaled).int64Value
        guard value > 0, Decimal(value) == scaled else { throw invalid("\(title) must resolve to a whole number of bytes.") }
        return value
    }
    private func invalid(_ message: String) -> ExecutionBudgetError { .invalid(message) }
}
