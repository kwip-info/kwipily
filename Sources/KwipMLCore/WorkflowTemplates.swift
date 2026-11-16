import Foundation

/// Presentation/provenance metadata, separate from any grants or recorded run data.
public struct WorkflowTemplateOrigin: Codable, Equatable {
    public let id: String
    public let version: Int
    public init(id: String, version: Int) { self.id = id; self.version = version }
}

public struct WorkflowTemplate: Codable, Equatable, Identifiable {
    public let id: String
    public let version: Int
    public let title: String
    public let category: String
    public let summary: String
    public let setup: [String]
    public let limitations: String
    public let attribution: String
    public let sourceURL: String
    public let resource: String
    public let proofCase: String
    public let suggestedWorkload: ExecutionWorkloadPolicy.Preset?
    public let suggestCompactHistory: Bool?
}

public enum WorkflowTemplates {
    private static let cachedCatalog: Result<[WorkflowTemplate], Error> = Result { try loadCatalog() }
    public static func catalog() throws -> [WorkflowTemplate] { try cachedCatalog.get() }
    private static func loadCatalog() throws -> [WorkflowTemplate] {
        let data = try resource("WorkflowTemplateCatalog.json")
        let entries = try JSONDecoder().decode([WorkflowTemplate].self, from: data)
        guard !entries.isEmpty, entries.count <= 64,
              Set(entries.map(\.id)).count == entries.count,
              entries.allSatisfy({ validID($0.id) && $0.version > 0 && $0.version <= 10000 &&
                  $0.resource == "Template-" + $0.id + ".json" && !$0.setup.isEmpty && $0.setup.count <= 8 &&
                  URL(string: $0.sourceURL)?.scheme == "https" &&
                  $0.suggestedWorkload != .custom }) else {
            throw DataFlowContractError("The workflow template catalog is invalid.")
        }
        return entries
    }

    /// Fresh managed source only. Native file/model/output grants and recorded evidence
    /// are never part of the template. The shared project owner performs persistence.
    public static func instantiate(id: String) throws -> WorkflowDocument {
        guard validID(id), let entry = try catalog().first(where: { $0.id == id }) else {
            throw DataFlowContractError("This workflow template is unavailable.")
        }
        var source = try WorkflowCodec.decode(resource(entry.resource))
        guard source.drafts.isEmpty, source.definitions.isEmpty, source.guidedBuild == nil else {
            throw DataFlowContractError("A template cannot contain private drafts or lesson state.")
        }
        source.projectID = UUID().uuidString
        source.revision = 0
        source.name = entry.title
        source.schemaVersion = max(source.schemaVersion, 24)
        source.templateOrigin = .init(id: entry.id, version: entry.version)
        try WorkflowCodec.validate(source)
        return source
    }

    private static func validID(_ id: String) -> Bool {
        !id.isEmpty && id.count <= 80 && id.allSatisfy { $0.isASCII && ($0.isLowercase || $0.isNumber || $0 == "-") }
    }
    private static func resource(_ file: String) throws -> Data {
        let name = file as NSString
        guard file == name.lastPathComponent,
              let url = Bundle.module.url(forResource: name.deletingPathExtension, withExtension: name.pathExtension),
              let size = try url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              size <= WorkflowCodec.maximumBytes else {
            throw DataFlowContractError("The workflow template resource is unavailable.")
        }
        return try Data(contentsOf: url)
    }
}
