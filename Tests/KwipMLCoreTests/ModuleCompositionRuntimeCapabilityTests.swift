import XCTest
@testable import KwipMLCore

final class ModuleCompositionRuntimeCapabilityTests: XCTestCase {
    private var exact: Data { Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8) }

    func testExactDescriptorRetainsImmutableSeparateEvidence() throws {
        let capability = try ModuleCompositionRuntimeCapability.decode(exact)
        XCTAssertEqual(capability.descriptor, try JSONDecoder().decode(JSONValue.self, from: exact))
        XCTAssertNoThrow(try ModuleCompositionRuntimeCapability.verify(exact))
    }
    func testEveryMissingChangedForeignOrOversizedFieldRefuses() throws {
        let value = try JSONDecoder().decode([String: JSONValue].self, from: exact)
        for key in value.keys {
            var missing = value; missing.removeValue(forKey: key)
            XCTAssertThrowsError(try ModuleCompositionRuntimeCapability.decode(JSONEncoder().encode(missing)), key)
            var changed = value; changed[key] = .null
            XCTAssertThrowsError(try ModuleCompositionRuntimeCapability.verify(JSONEncoder().encode(changed)), key)
        }
        var foreign = value; foreign["unreviewed"] = .bool(true)
        XCTAssertThrowsError(try ModuleCompositionRuntimeCapability.decode(JSONEncoder().encode(foreign)))
        XCTAssertThrowsError(try ModuleCompositionRuntimeCapability.decode(Data(repeating: 32, count: 4097)))
        XCTAssertThrowsError(try ModuleCompositionRuntimeCapability.decode(Data("null".utf8)))
    }
    func testVersionsModesMembersAndBackendGuaranteesCannotBePartiallyGranted() throws {
        let value = try JSONDecoder().decode([String: JSONValue].self, from: exact)
        let mutations: [(String, JSONValue)] = [("version", .object(["major": 1, "minor": 0, "patch": 0])), ("planSchemaVersion", .bool(true)), ("captureFormatVersion", 5), ("workflowSchemaVersion", 10), ("compilerContractVersion", 9), ("modes", .array([.string("architecture-export")])), ("sourceMembers", .array([.string("torch.nn.Module")])), ("exactResumeDevices", .array([.string("cpu"), .string("mps")])), ("sharedParameters", true), ("maximumParameters", 2_000_000)]
        for (key, replacement) in mutations {
            var changed = value; changed[key] = replacement
            XCTAssertThrowsError(try ModuleCompositionRuntimeCapability.decode(JSONEncoder().encode(changed)), key)
        }
    }
}
