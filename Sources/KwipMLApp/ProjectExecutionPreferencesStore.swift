import Foundation
import KwipMLCore

/// Local execution overrides keyed by project identity. Portable authored graphs
/// and workspace layouts are separate records and are never reset here.
final class ProjectExecutionPreferencesStore {
    private let workspace: URL
    private static let maximumBytes = 4096
    private struct Envelope: Codable {
        let schemaVersion: Int
        let projectID: String
        let overrides: ProjectExecutionOverrides
    }

    init(workspace: URL) { self.workspace = workspace }

    func load(projectID: String) throws -> ProjectExecutionOverrides {
        let parts = try Self.parts(projectID)
        guard let bytes = try DiagnosticFileIO.readIfPresent(root: workspace, parts: parts, limit: Self.maximumBytes) else {
            return .factory
        }
        return try Self.decode(bytes, projectID: projectID)
    }

    func save(_ overrides: ProjectExecutionOverrides, projectID: String) throws {
        let parts = try Self.parts(projectID)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let bytes = try encoder.encode(Envelope(schemaVersion: 1, projectID: projectID, overrides: overrides))
        _ = try Self.decode(bytes, projectID: projectID)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        try DiagnosticFileIO.writeAtomic(bytes, root: workspace, parts: parts)
    }

    @discardableResult func reset(projectID: String) throws -> ProjectExecutionOverrides {
        let next = try load(projectID: projectID).resettingResources()
        try save(next, projectID: projectID)
        return next
    }

    private static func parts(_ projectID: String) throws -> [String] {
        guard UUID(uuidString: projectID)?.uuidString == projectID else {
            throw WorkflowArchiveError("Project execution preferences require a canonical project identity.")
        }
        return ["project-execution-preferences", projectID + ".json"]
    }

    private static func decode(_ bytes: Data, projectID: String) throws -> ProjectExecutionOverrides {
        guard bytes.count <= maximumBytes,
              let object = try JSONSerialization.jsonObject(with: bytes) as? [String: Any],
              Set(object.keys) == ["schemaVersion", "projectID", "overrides"] else {
            throw WorkflowArchiveError("Project execution preferences require a bounded record with supported fields.")
        }
        let value = try JSONDecoder().decode(Envelope.self, from: bytes)
        guard value.schemaVersion == 1, value.projectID == projectID else {
            throw WorkflowArchiveError("Project execution preferences belong to another project or an unsupported version.")
        }
        return value.overrides
    }
}
