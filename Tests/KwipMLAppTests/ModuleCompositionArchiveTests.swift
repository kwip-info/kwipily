import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModuleCompositionArchiveTests: XCTestCase {
    // Independent unit-fixture contract literal; this does not claim an actual runtime probe.
    private let descriptor = try! JSONDecoder().decode(JSONValue.self, from: Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8))
    private let pin = String(repeating: "a", count: 64)
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func plan(selected: Bool, version: Int = 11) throws -> WorkflowExecutionPlan {
        var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        document.schemaVersion = version
        if selected {
            document.definitions[0].composition = .init(kind: .graph, children: document.definitions[0].body.nodes.map { .init(localID: $0.id) })
        }
        return try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress,
            modelRegistry: .builtIn(), dataRegistry: .load(from: repository.appendingPathComponent("runner/dataflow-operators.json")), contractVersion: version == 11 ? 10 : 9)
    }
    private func capture(_ plan: WorkflowExecutionPlan, version: Int = 11, format: Int? = 6) throws -> WorkflowArchiveCapture {
        func file(_ path: String) -> WorkflowArchiveFile { .init(path: path, bytes: 10, sha256: pin) }
        return WorkflowArchiveCapture(schemaVersion: version, compilerContractVersion: version == 11 ? 10 : 9,
            projectID: plan.projectID, directoryID: UUID().uuidString, runID: UUID().uuidString,
            requestIdentity: pin, revision: plan.revision, train: plan.train,
            workflow: file("workflow.json"), modelRegistry: file("model-registry.json"), dataRegistry: file("data-registry.json"),
            modelRegistrySHA256: pin, dataRegistrySHA256: pin, runtimeEvidence: file("runtime-evidence.json"),
            sources: plan.sourceAddresses.enumerated().map { .init(address: $0.element, path: "inputs/source-\($0.offset).csv", bytes: 10, sha256: pin) },
            plan: try WorkflowArchivePlan(plan, compilerContractVersion: version == 11 ? 10 : 9), captureFormatVersion: format, moduleCompositionDescriptor: version == 11 ? descriptor : nil)
    }
    func testSelectedCompositionCaptureSixRetainsExactSourcePlanOriginsAndPins() throws {
        let p = try plan(selected: true), value = try capture(p)
        XCTAssertNotNil(p.model.graph.compositionPlan)
        XCTAssertEqual(value.plan.modelProvenance?.nativeExpansionContractVersion, 2)
        XCTAssertEqual(value.plan.modelProvenance?.sourceCompositeDocument.schemaVersion, 3)
        let bytes = try WorkflowArchiveCodec.encodeCapture(value)
        let restored = try WorkflowArchiveCodec.decodeCapture(bytes)
        XCTAssertEqual(restored, value)
        XCTAssertTrue(try restored.plan.matches(p, compilerContractVersion: 10))
        let expansion = try XCTUnwrap(restored.plan.modelProvenance).verifiedExpansion(registry: .builtIn(), context: .preparedWorkflow)
        XCTAssertEqual(expansion.graph, restored.plan.modelGraph)
        XCTAssertEqual(Set(expansion.nodeOrigins.keys), Set(restored.plan.modelOrigins.map(\.nodeID)))
    }
    func testNoPlanSchemaElevenStillUsesCaptureSixAndExpansionTwoWithoutInventingBudget() throws {
        let p = try plan(selected: false), value = try capture(p)
        XCTAssertNil(p.model.graph.compositionPlan)
        XCTAssertEqual(p.model.provenance?.nativeExpansionContractVersion, 2)
        XCTAssertEqual(p.model.provenance?.sourceCompositeDocument.schemaVersion, 3)
        XCTAssertNil(value.executionBudget)
        XCTAssertNil(value.derivativeCapture)
        let bytes = try WorkflowArchiveCodec.encodeCapture(value)
        XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(bytes), value)
        for format: Int? in [nil, 2, 3, 4, 5] {
            var wrong = value; wrong.captureFormatVersion = format
            XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(wrong), "schema11 format \(String(describing: format))")
        }
    }
    func testCaptureSixKeepsGradientPermissionAndInitializationIndependent() throws {
        var p = try plan(selected: true)
        let owner = try XCTUnwrap(p.model.graph.nodes.first { $0.type == "linear" })
        p.gradientPolicyRecipe = .init(configuration: try .init(values: ["accumulation_steps": 2]), source: .init(domain: .control, localID: "gradient"))
        p.trainabilityRecipe = .init(configuration: try .init(values: ["owners": .array([.object(["node": .string(owner.id), "parameter": .string("weight"), "mode": .string("frozen")])])]), source: .init(domain: .control, localID: "trainability"))
        p.initializationRecipe = .init(configuration: try .init(values: ["seed": 42, "rules": .array([.object(["owner": .object(["node": .string(owner.id), "parameter": .string("weight")]), "algorithm": .string("zeros"), "settings": .object([:])])])]), source: .init(domain: .control, localID: "initialization"))
        var value = try capture(p)
        value.executionBudget = try ExecutionPreferences.factory.resolve().budget
        value.derivativeCapture = .factory
        XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(value)), value)
        value.derivativeCapture = nil
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(value))
    }
    func testLegacyCaptureBytesAndNoPlanGraphRemainUnchanged() throws {
        let p = try plan(selected: false, version: 10), value = try capture(p, version: 10, format: nil)
        XCTAssertNil(p.model.graph.compositionPlan)
        XCTAssertEqual(p.model.provenance?.nativeExpansionContractVersion, 1)
        let bytes = try WorkflowArchiveCodec.encodeCapture(value)
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(bytes)), bytes)
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("captureFormatVersion"))
        var wrong = value; wrong.captureFormatVersion = 6
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(wrong))
    }
    private func ownershipFixture() -> (GraphDocument, [String:JSONValue]) {
        let nestedID = CompositionPlan.containerID(path: ["group"])
        let root = CompositionPlan.Container(id: "root", kind: .graph, path: [], children: [.init(kind: .container, id: nestedID, key: "0")], inputs: [.init(id: "input", targets: [.init(node: "learn", port: "in")])], outputs: [.init(id: "output", source: .init(node: "learn", port: "out"))])
        let namedMap = CompositionPlan.Container(id: nestedID, kind: .namedMap, path: ["group"], children: [.init(kind: .node, id: "learn", key: "affine")], inputs: [.init(id: "in", targets: [.init(node: "learn", port: "in")])], outputs: [.init(id: "out", source: .init(node: "learn", port: "out"))])
        let plan = CompositionPlan(containers: [root, namedMap])
        let graph = GraphDocument(schemaVersion: 2, nodes: [.init(id: "input", type: "input", config: ["features": 2]), .init(id: "learn", type: "linear", config: ["in_features": 2, "out_features": 2, "bias": .bool(true)]), .init(id: "output", type: "output")], edges: [.init(id: "enter", source: "input", target: "learn"), .init(id: "leave", source: "learn", target: "output")], compositionPlan: plan)
        func owner(_ parameter: String, _ shape: [Int]) -> JSONValue { .object(["node": .string("learn"), "parameter": .string(parameter), "name": .string("composition_root.blocks.0.affine.operation." + parameter), "shape": .array(shape.map { .integer(Int64($0)) }), "dtype": .string("float32")]) }
        let owners = [owner("weight", [2,2]), owner("bias", [2])]
        let state = owners.map { item -> JSONValue in guard case .object(var fields) = item else { return item }; fields["kind"] = .string("parameter"); return .object(fields) }
        let encodedPlan = try! JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(plan))
        return (graph, ["graph": try! JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(graph)), "checkpoint": .object(["contract": .object(["composition": .object(["schemaVersion": 1, "plan": encodedPlan, "parameterOwnership": .array(owners), "stateOwnership": .array(state)])])])])
    }
    func testCheckpointCompositionOwnershipRejectsRepinnedNamesShapesOrderAndPlan() throws {
        let (graph, manifest) = ownershipFixture(), registry = try ComponentRegistry.builtIn()
        XCTAssertNoThrow(try WorkflowLifecycleVerifier.validateComposition(manifest, graph: graph, registry: registry, context: .preparedWorkflow))
        for mutation in ["absent", "name", "shape", "order", "state", "plan", "foreign"] {
            var bad = manifest
            guard case .object(var checkpoint) = bad["checkpoint"], case .object(var contract) = checkpoint["contract"], case .object(var composition) = contract["composition"], case .array(var owners) = composition["parameterOwnership"] else { return XCTFail("Fixture") }
            if mutation == "name" || mutation == "shape" { guard case .object(var owner) = owners[0] else { return XCTFail("Fixture") }; owner[mutation == "name" ? "name" : "shape"] = mutation == "name" ? .string("layers.learn.weight") : .array([1,4]); owners[0] = .object(owner); composition["parameterOwnership"] = .array(owners) }
            if mutation == "order" { composition["parameterOwnership"] = .array(Array(owners.reversed())) }
            if mutation == "state" { composition["stateOwnership"] = .array([]) }
            if mutation == "plan" { composition["plan"] = .null }
            if mutation == "foreign" { composition["foreign"] = .bool(true) }
            contract["composition"] = mutation == "absent" ? nil : .object(composition)
            checkpoint["contract"] = .object(contract); bad["checkpoint"] = .object(checkpoint)
            XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateComposition(bad, graph: graph, registry: registry, context: .preparedWorkflow), mutation)
        }
        var unselected = graph; unselected.compositionPlan = nil
        XCTAssertThrowsError(try WorkflowLifecycleVerifier.validateComposition(manifest, graph: unselected, registry: registry, context: .preparedWorkflow))
    }
    func testIndependentSnapshotReexpansionRejectsChangedConsumedSource() throws {
        let p = try plan(selected: true), archived = try WorkflowArchivePlan(p, compilerContractVersion: 10)
        try archived.validateModelSnapshot(registry: .builtIn(), context: .preparedWorkflow)
        var raw = try XCTUnwrap(JSONSerialization.jsonObject(with: WorkflowArchiveCodec.encode(archived)) as? [String:Any])
        var provenance = try XCTUnwrap(raw["modelProvenance"] as? [String:Any])
        var source = try XCTUnwrap(provenance["sourceCompositeDocument"] as? [String:Any])
        var definitions = try XCTUnwrap(source["definitions"] as? [[String:Any]])
        definitions[0]["title"] = "Repinned changed display title"
        // Changing display copy is still different captured source, but does not by itself alter numerical identity.
        var body = try XCTUnwrap(definitions[0]["body"] as? [String:Any]), nodes = try XCTUnwrap(body["nodes"] as? [[String:Any]])
        let index = try XCTUnwrap(nodes.firstIndex { $0["type"] as? String == "linear" })
        var config = try XCTUnwrap(nodes[index]["config"] as? [String:Any]); config["bias"] = false; nodes[index]["config"] = config
        body["nodes"] = nodes; definitions[0]["body"] = body; source["definitions"] = definitions; provenance["sourceCompositeDocument"] = source; raw["modelProvenance"] = provenance
        let bad = try JSONDecoder().decode(WorkflowArchivePlan.self, from: JSONSerialization.data(withJSONObject: raw))
        XCTAssertThrowsError(try bad.validateModelSnapshot(registry: .builtIn(), context: .preparedWorkflow))
    }

    func testCaptureDescriptorCannotBeMissingMalformedOrAddedToLegacy() throws {
        var selected = try capture(plan(selected: true))
        selected.moduleCompositionDescriptor = nil
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(selected))
        selected.moduleCompositionDescriptor = .object(["version":.string("1.0.0")])
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(selected))
        var old = try capture(plan(selected: false, version: 10), version: 10, format: nil)
        old.moduleCompositionDescriptor = descriptor
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(old))
    }

}
