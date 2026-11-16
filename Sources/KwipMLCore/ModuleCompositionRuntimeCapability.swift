import Foundation

/// Exact independent native composition support. Primitive descriptors grant no such support.
public struct ModuleCompositionRuntimeCapability: Equatable {
    public let descriptor: JSONValue
    private init(descriptor: JSONValue) { self.descriptor = descriptor }
    public static func decode(_ data: Data) throws -> Self {
        guard data.count <= 4096,
              let value = try? JSONDecoder().decode(JSONValue.self, from: data),
              value == expected else {
            throw GraphConfigurationError("Verified native module composition support is unavailable. Refresh the runtime.")
        }
        return Self(descriptor: value)
    }
    private static let expected = try! JSONDecoder().decode(JSONValue.self, from: Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8))
    public static func verify(_ data: Data) throws { _ = try decode(data) }
}
