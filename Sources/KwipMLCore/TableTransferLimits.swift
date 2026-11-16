/// Matched with runner/table_transfer_policy.py by the cross-runtime contract test.
/// File/row admission and display paging are distinct from Working data selection.
public enum TableTransferLimits {
    public static let artifactBytes = 50_000_000
    public static let pageBytes = 1_000_000
    public static let pageRows = 100
    public static let columns = 128
    public static let rows = 5_000_000
    public static let cells = 5_000_000
}

/// Total-table allowances are captured separately from the bounded page decoder.
public struct TableTransferCapacity {
    public let artifactBytes: Int
    public let columns: Int
    public let rows: Int
    public let cells: Int
    public init(workload: ExecutionWorkloadPolicy? = nil) {
        artifactBytes = workload.map { Int($0.limit(.retainedBytes)) } ?? TableTransferLimits.artifactBytes
        columns = workload.map { Int($0.limit(.tableColumns)) } ?? TableTransferLimits.columns
        rows = workload.map { Int($0.limit(.tableRows)) } ?? TableTransferLimits.rows
        cells = workload.map { Int($0.limit(.tableCells)) } ?? TableTransferLimits.cells
    }
}
