import Foundation

/// Local immutable provenance binding; runnable graph remains a schema-v1 primitive architecture.
public struct CompositeSourceReference: Codable, Equatable {
    public var path: String
    public var sha256: String
    public var expandedSemanticSHA256: String
    public init(path: String, sha256: String, expandedSemanticSHA256: String) {
        self.path = path; self.sha256 = sha256; self.expandedSemanticSHA256 = expandedSemanticSHA256
    }
}

public struct CompositeSourceSnapshot: Codable, Equatable {
    public var nativeExpansionContractVersion: Int
    public var sourceCompositeDocument: CompositeProjectDocument
    public var expandedGraph: GraphDocument
    public var nodeOrigins: [String: CompositeNodeOrigin]
    public var consumedDefinitions: [CompositeDefinitionReference]
    public func verifiedExpansion(registry: ComponentRegistry, context: GraphValidationContext = .bundledDataset) throws -> CompositeExpansion {
        let expectedVersion = sourceCompositeDocument.schemaVersion >= 3 ? 2 : 1
        guard nativeExpansionContractVersion == expectedVersion else { throw CompositeContractError(.unsupportedSchema, "Source expansion version does not match its source contract.") }
        let expansion = try CompositeExpander.expand(sourceCompositeDocument, registry: registry, context: context)
        // Preserve immutable schema-1 snapshot bytes while comparing their current meaning.
        // Current/native snapshots remain structurally exact; ownership and pins never migrate.
        let capturedGraph = expandedGraph.schemaVersion == 1 && expectedVersion == 1
            ? try GraphMigration.toCurrent(expandedGraph, registry: registry) : expandedGraph
        guard Self.sameEncoding(expansion.graph,capturedGraph), expansion.nodeOrigins == nodeOrigins, expansion.consumedDefinitions == consumedDefinitions else { throw CompositeContractError(.invalidExpandedGraph, "Captured graph, composition plan or source ownership differs from independent re-expansion.") }
        return expansion
    }
    /// Codable JSON is the durable identity. JSONValue deliberately distinguishes
    /// integer/number in memory, but JSON round trips can decode 0.0 as integer 0.
    /// Compare the exact finite serialized structure, never just expanded math.
    public static func ==(lhs:Self,rhs:Self)->Bool {sameEncoding(lhs,rhs)}
    private static func sameEncoding<T:Encodable>(_ lhs:T,_ rhs:T)->Bool {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        guard let a=try? encoder.encode(lhs),let b=try? encoder.encode(rhs) else {return false}
        return a==b
    }
    public init(project: CompositeProjectDocument, expansion: CompositeExpansion) {
        nativeExpansionContractVersion = project.schemaVersion >= 3 ? 2 : 1; sourceCompositeDocument = project
        expandedGraph = expansion.graph; nodeOrigins = expansion.nodeOrigins; consumedDefinitions = expansion.consumedDefinitions
    }
}
