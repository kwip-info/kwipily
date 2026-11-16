import Foundation
import CryptoKit

/// Captured executable values, independent of later edits to an authoring document.
/// The graph is the worker input; provenance retains the exact source and existing expansion mapping.
public struct ModelExecutionSnapshot:Equatable {
    public let graph:GraphDocument
    public let provenance:CompositeSourceSnapshot?
    public let origins:[String:CompositeNodeOrigin]
    public let authoringRevision:UInt64?
    public let semanticSHA256:String
}

public enum ModelExecutionSnapshotBuilder {
    public static func build(_ document:ModelProjectDocument,registry:ComponentRegistry,context:GraphValidationContext = .bundledDataset)throws->ModelExecutionSnapshot {
        try ModelProjectCodec.validate(document)
        switch document {
        case .legacyFlat(let graph):
            guard graph.nodes.allSatisfy({$0.parameterSettings == nil}) else {throw GraphConfigurationError("Execute layer parameter settings through the universal workflow so they are captured as training recipes.")}
            try validateExecution(graph,registry:registry,context:context)
            let origins=Dictionary(uniqueKeysWithValues:graph.nodes.map {($0.id,CompositeNodeOrigin(instancePath:[],localNodeID:$0.id,definition:nil))})
            return ModelExecutionSnapshot(graph:graph,provenance:nil,origins:origins,authoringRevision:nil,semanticSHA256:try semanticDigest(graph,registry:registry))
        case .blockModel(let authoring):
            guard authoring.source.root.nodes.allSatisfy({$0.parameterSettings == nil}),authoring.source.definitions.allSatisfy({$0.body.nodes.allSatisfy({$0.parameterSettings == nil})}) else {throw GraphConfigurationError("Execute layer parameter settings through the universal workflow so they are captured as training recipes.")}
            guard authoring.drafts.isEmpty else {throw CompositeContractError(.invalidExpandedGraph,"Apply or discard the unresolved Block draft before creating an execution snapshot.")}
            // Empty or unfinished placed Blocks must not disappear during expansion and
            // accidentally leave an otherwise runnable parent looking complete.
            for reference in Set(authoring.source.root.instances.map(\.definition)) {
                guard let definition = authoring.source.definitions.first(where: { $0.reference == reference }) else { throw CompositeContractError(.missingDefinition, "Placed Block definition is missing.") }
                try CompositeDefinitionValidation.validate(definition, definitions: authoring.source.definitions, registry: registry)
            }
            let expansion=try CompositeExpander.expand(authoring.source,registry:registry,context:context)
            try validateExecution(expansion.graph,registry:registry,context:context)
            return ModelExecutionSnapshot(graph:expansion.graph,provenance:CompositeSourceSnapshot(project:authoring.source,expansion:expansion),origins:expansion.nodeOrigins,authoringRevision:authoring.revision,semanticSHA256:try semanticDigest(expansion.graph,registry:registry))
        }
    }
    private static func validateExecution(_ graph:GraphDocument,registry:ComponentRegistry,context:GraphValidationContext)throws {
        if let plan = graph.compositionPlan { try plan.validate(graph: graph, registry: registry) }
        let validation=GraphValidator.validate(graph,registry:registry,context:context)
        guard validation.isValid else {throw CompositeContractError(.invalidExpandedGraph,"Model is not executable: "+validation.issues.map(\.message).joined(separator:" "),validationIssues:validation.issues)}
    }
    private static func semanticDigest(_ graph:GraphDocument,registry:ComponentRegistry)throws->String {
        // Same canonical projection/hash as the existing App CompositeArchitectureIdentity.
        SHA256.hash(data:try GraphSemantics.canonicalData(graph,registry:registry)).map{String(format:"%02x",$0)}.joined()
    }
}
