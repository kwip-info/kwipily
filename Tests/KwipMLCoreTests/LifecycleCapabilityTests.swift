import XCTest
import KwipMLCore

final class LifecycleCapabilityTests: XCTestCase {
    func testArtifactOnlyDeclarationUsesExistingActionsAndConservativeDeviceGuarantee() throws {
        let value = LifecycleCapabilityBridge.declaration()
        XCTAssertTrue(CapabilityContractValidator.validate(value).isValid)
        XCTAssertEqual(value.id, "checkpoint_lifecycle")
        XCTAssertEqual(value.packageID, "kwip.lifecycle")
        XCTAssertEqual(value.ports.map(\.id), ["savedWeights", "savedContinuation", "parentWeights", "parentContinuation"])
        XCTAssertTrue(value.ports.allSatisfy { $0.schema.kind == .artifact && $0.schema.tensor == nil })
        XCTAssertEqual(value.lifecycle.resume, .weightsOnly)
        XCTAssertEqual(value.devices, [.cpu, .mps])
        XCTAssertTrue(value.fields.isEmpty)
        XCTAssertNil(value.shapeRuleID)
        XCTAssertFalse(value.lifecycle.phases.contains(.forward))
        XCTAssertFalse(value.lifecycle.phases.contains(.train))
        let bindings = BundledLifecycleCapabilities.entry.bindings
        XCTAssertEqual(bindings.map(\.id), ["weights-save", "continuation-save", "weights-restore", "continuation-restore"])
        XCTAssertEqual(bindings.map(\.trainingMode), [nil, nil, .fineTune, .exactResume])
        XCTAssertEqual(bindings.map(\.surface), [.history, .history, .train, .train])
        XCTAssertEqual(bindings.map(\.action), [.exportTrainedPackage, .retainCompletedCheckpoint, .selectParent, .selectParent])
        let additiveIDs:Set<String>=["bag_input","embedding_bag","dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(try .builtIn()).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count, 83)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        XCTAssertFalse(fixtureModelComponentsBeforeTextExtensions(try .builtIn()).contains { $0.id == value.id })
    }
    func testExactRuntimeDiscoveryIndependentOfTeachingPublicationAndMissingSupport() throws {
        let data = try JSONEncoder().encode(BundledLifecycleCapabilities.entry.runtimeContract)
        XCTAssertNoThrow(try LifecycleRuntimeCapability.verify(data))
        XCTAssertTrue(LifecycleRuntimeCapability(descriptor: data).isAvailable)
        XCTAssertFalse(LifecycleRuntimeCapability().isAvailable)
        XCTAssertFalse(LifecycleRuntimeCapability(descriptor: Data("{}".utf8)).isAvailable)
        XCTAssertFalse(LifecycleRuntimeCapability(descriptor: Data(repeating: 32, count: 4097)).isAvailable)
        guard case .object(let value) = BundledLifecycleCapabilities.entry.runtimeContract else { return XCTFail("Missing reviewed object") }
        for key in value.keys {
            var missing = value; missing.removeValue(forKey: key)
            let bytes = try JSONEncoder().encode(JSONValue.object(missing))
            XCTAssertFalse(LifecycleRuntimeCapability(descriptor: bytes).isAvailable, key)
        }
    }
    func testContradictoryTrustDeviceTimingTraceAndUnknownFieldsRefuse() throws {
        guard case .object(let original) = BundledLifecycleCapabilities.entry.runtimeContract else { return XCTFail("Missing reviewed object") }
        let changes: [String: JSONValue] = ["externalImport": .bool(true), "periodicSave": .bool(true),
            "cpuResume": .string("weightsOnly"), "mpsResume": .string("epochBoundaryExactCPU"),
            "exactResumeDevices": .array([.string("cpu"), .string("mps")]),
            "saveTiming": .string("midBatch"), "sourceTrust": .string("arbitraryPath"),
            "schedulerTracePolicy": .string("optional"), "checkpointSchemaVersions": .array([.bool(true)]),
            "forwardAdapter": .bool(true)]
        for (key, change) in changes {
            var value = original; value[key] = change
            XCTAssertThrowsError(try LifecycleRuntimeCapability.verify(JSONEncoder().encode(JSONValue.object(value))), key)
        }
    }
}
