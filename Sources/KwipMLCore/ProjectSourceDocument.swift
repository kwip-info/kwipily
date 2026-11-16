import Foundation

/// Managed source formats remain distinct; this sum type never implies an authoring migration.
public enum ProjectSourceDocument: Equatable {
    case legacyFlat(GraphDocument)
    case blockModel(ModelAuthoringDocument)
    case workflow(WorkflowDocument)

    public var name: String {
        switch self {
        case .legacyFlat(let graph): return graph.name
        case .blockModel(let model): return model.source.name
        case .workflow(let workflow): return workflow.name
        }
    }
    public var projectID: String? {
        switch self {
        case .legacyFlat: return nil
        case .blockModel(let model): return model.projectID
        case .workflow(let workflow): return workflow.projectID
        }
    }
}

public enum ProjectSourceCodec {
    public static let maximumBytes = 2_000_000

    public static func decode(_ data: Data) throws -> ProjectSourceDocument {
        guard data.count <= maximumBytes else { throw ProjectLibraryError.resourceLimit }
        guard case .object(let root) = try JSONDecoder().decode(JSONValue.self, from: data) else {
            throw ProjectLibraryError.unsupportedSchema
        }
        if let kind = root["kind"] {
            switch kind {
            case .string("workflowProject"): return .workflow(try WorkflowCodec.decode(data))
            case .string("modelProject"):
                guard case .blockModel(let model) = try ModelProjectCodec.decode(data) else {
                    throw ProjectLibraryError.documentKindMismatch
                }
                return .blockModel(model)
            default: throw ProjectLibraryError.unsupportedSchema
            }
        }
        guard case .legacyFlat(let graph) = try ModelProjectCodec.decode(data) else {
            throw ProjectLibraryError.documentKindMismatch
        }
        return .legacyFlat(graph)
    }

    public static func encode(_ document: ProjectSourceDocument) throws -> Data {
        switch document {
        case .legacyFlat(let graph): return try ModelProjectCodec.encode(.legacyFlat(graph))
        case .blockModel(let model): return try ModelProjectCodec.encode(.blockModel(model))
        case .workflow(let workflow): return try WorkflowCodec.encode(workflow)
        }
    }
}
