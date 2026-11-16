import Foundation

/// Deterministic source-order working population. Full consumers override this per branch.
public struct PreparationSampling: Codable, Equatable {
    public enum Mode: String, Codable, CaseIterable { case all, rowCount, percentage }
    public var schemaVersion: Int = 1
    public var mode: Mode
    public var rowCount: Int?
    public var percentage: Double?
    public static let all = Self(mode: .all)
    public init(mode: Mode, rowCount: Int? = nil, percentage: Double? = nil) {
        self.mode = mode; self.rowCount = rowCount; self.percentage = percentage
    }
    public func validated() throws -> Self {
        guard schemaVersion == 1 else { throw DataFlowContractError("Unsupported sampling policy.") }
        switch mode {
        case .all: guard rowCount == nil, percentage == nil else { throw DataFlowContractError("Full data has no sample limit.") }
        case .rowCount: guard let rowCount, (1...5_000_000).contains(rowCount), percentage == nil else { throw DataFlowContractError("Sample records must be between 1 and 5,000,000.") }
        case .percentage: guard let percentage, percentage.isFinite, percentage > 0, percentage <= 100, rowCount == nil else { throw DataFlowContractError("Sample percentage must be greater than zero and at most 100.") }
        }
        return self
    }
    public func transportJSON() throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(validated()), as: UTF8.self)
    }
    public var title: String {
        switch mode { case .all: return "All data"; case .rowCount: return "First \(rowCount ?? 0) records"; case .percentage: return "First \(percentage ?? 0)%" }
    }
}
