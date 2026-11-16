import Foundation
import XCTest
@testable import KwipMLCore

final class GradientWorkflowContextTests: XCTestCase {
    private let policyAddress = WorkflowNodeAddress(domain: .control, localID: "selected-influence")
    private let trainabilityAddress = WorkflowNodeAddress(domain: .control, localID: "selected-owners")
    private func registries() throws -> (ComponentRegistry, DataFlowOperatorRegistry) {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        return try (.builtIn(), .load(from: root.appendingPathComponent("runner/dataflow-operators.json")))
    }
    private func compile(_ document: WorkflowDocument, version: Int = 9) throws -> WorkflowExecutionPlan {
        let (model, data) = try registries()
        return try WorkflowCompiler.compileHistoricalEvidence(document, train: WorkflowExamples.trainAddress, modelRegistry: model, dataRegistry: data, contractVersion: version)
    }
    private func request(output: String, input: String, target: JSONValue? = nil, port: String = "out", upstream: JSONValue = .null) -> JSONValue {
        .object(["id": .string("effect"), "output": .object(["node": .string(output), "port": .string(port)]), "partition": .string("validation"), "row_start": 0, "row_count": 1, "reduction": .string(upstream == .null ? "sum" : "upstream"), "upstream": upstream, "targets": .array([target ?? .object(["kind": .string("input"), "node": .string(input), "port": .string("out")])]), "unavailable": .string("reportUnavailable")])
    }
    private func fixture() throws -> (WorkflowDocument, String, String) {
        var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        let old = try compile(document, version: 2)
        let linear = try XCTUnwrap(old.model.graph.nodes.first { $0.type == "linear" }).id
        let input = try XCTUnwrap(old.model.graph.nodes.first { $0.type == "input" }).id
        document.schemaVersion = 10
        let gradient = try GradientPolicyConfiguration(values: ["accumulation_steps": 2, "requests": .array([request(output: linear, input: input)])])
        let owners = try TrainabilityConfiguration(values: ["owners": .array([.object(["node": .string(linear), "parameter": .string("weight"), "mode": .string("frozen")])])])
        document.nodes += [.init(address: policyAddress, payload: .gradientPolicyRecipe(type: "gradient_policy", configurationVersion: 1, config: gradient.values)), .init(address: trainabilityAddress, payload: .trainabilityRecipe(type: "trainability", configurationVersion: 1, config: owners.values))]
        document.edges += [.init(id: "gradient-to-train", from: .init(address: policyAddress, port: "out"), to: .init(address: WorkflowExamples.trainAddress, port: "gradientPolicy"), kind: .gradientPolicyRecipe), .init(id: "owners-to-train", from: .init(address: trainabilityAddress, port: "out"), to: .init(address: WorkflowExamples.trainAddress, port: "trainability"), kind: .trainabilityRecipe)]
        return (document, linear, input)
    }
    private func withRequest(_ source: WorkflowDocument, _ value: JSONValue, scale: String = "mean") throws -> WorkflowDocument {
        var document = source
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address == policyAddress })
        document.nodes[index].payload = .gradientPolicyRecipe(type: "gradient_policy", configurationVersion: 1, config: try GradientPolicyConfiguration(values: ["gradient_scale": .string(scale), "requests": .array([value])]).values)
        return document
    }
    func testQualifiedRecipesRetainModelAndDistinctOptionalControlPorts() throws {
        let (document, _, _) = try fixture(), (model, data) = try registries()
        let plan = try compile(document)
        XCTAssertEqual(plan.gradientPolicyRecipe?.source, policyAddress)
        XCTAssertEqual(plan.trainabilityRecipe?.source, trainabilityAddress)
        XCTAssertEqual(plan.gradientPolicyRecipe?.configuration.accumulationSteps, 2)
        XCTAssertEqual(plan.model.graph, try compile(WorkflowExamples.preparedParallelClassifier(device: "cpu"), version: 2).model.graph)
        let train = try XCTUnwrap(document.nodes.first { $0.address == WorkflowExamples.trainAddress })
        let ports = try WorkflowPortContracts.resolvePlanPorts(train, document: document, modelRegistry: model, dataRegistry: data,historicalEvidence:true)
        for (name, kind) in [("gradientPolicy", WorkflowEdgeKind.gradientPolicyRecipe), ("trainability", .trainabilityRecipe)] {
            let port = try XCTUnwrap(ports.inputs.first { $0.id == name })
            XCTAssertEqual(port.kind, kind); XCTAssertFalse(port.required); XCTAssertEqual(port.maximumConnections, 1)
        }
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)), document)
    }
    func testForeignOwnersNoninputTargetsAndMissingOutputPortsRefuse() throws {
        let (document, linear, input) = try fixture()
        for value in [request(output: "foreign", input: input), request(output: linear, input: input, port: "hidden"), request(output: linear, input: "foreign"), request(output: linear, input: linear), request(output: linear, input: input, target: .object(["kind": .string("parameter"), "node": .string("foreign"), "parameter": .string("weight")]))] {
            XCTAssertThrowsError(try compile(withRequest(document, value)))
        }
        var wrong = document
        let index = try XCTUnwrap(wrong.nodes.firstIndex { $0.address == trainabilityAddress })
        wrong.nodes[index].payload = .trainabilityRecipe(type: "trainability", configurationVersion: 1, config: ["owners": .array([.object(["node": .string("foreign"), "parameter": .string("weight"), "mode": .string("frozen")])])])
        XCTAssertThrowsError(try compile(wrong))
    }
    func testExactKnownUpstreamGeometryAndNativeMembershipBoundary() throws {
        let (document, linear, input) = try fixture(), (registry, _) = try registries()
        let plan = try compile(document)
        let validation = GraphValidator.validate(plan.model.graph, registry: registry, context: .preparedWorkflow)
        let layout = try XCTUnwrap(validation.outputTensorContracts[.init(linear, "out")]?.layout)
        let shape = [1] + layout.sample.dimensions
        let good: JSONValue = .object(["dtype": .string("float32"), "shape": .array(shape.map { .integer(Int64($0)) }), "values": .array(Array(repeating: .number(1), count: layout.sample.elementCount))])
        XCTAssertNoThrow(try compile(withRequest(document, request(output: linear, input: input, upstream: good))))
        let wrong: JSONValue = .object(["dtype": .string("float32"), "shape": .array([1, 999]), "values": .array(Array(repeating: 0, count: 999))])
        XCTAssertThrowsError(try compile(withRequest(document, request(output: linear, input: input, upstream: wrong))))
        let native = request(output: linear, input: input, target: .object(["kind": .string("parameter"), "node": .string(linear), "parameter": .string("native.nested.actual_parameter")]))
        XCTAssertNoThrow(try compile(withRequest(document, native)), "Native membership is verified after materialization by the worker, not guessed in Core.")
    }
    func testMalformedKindsVersionsDuplicateWiresAndObjectiveMismatchRefuse() throws {
        let (document, linear, input) = try fixture()
        XCTAssertThrowsError(try compile(withRequest(document, request(output: linear, input: input), scale: "sum")), "Implicit objective is mean; policy cannot override it.")
        var wrong = document; wrong.edges[wrong.edges.count - 1].kind = .gradientPolicyRecipe
        XCTAssertThrowsError(try compile(wrong))
        var duplicate = document; var edge = try XCTUnwrap(document.edges.first { $0.kind == .gradientPolicyRecipe }); edge.id = "duplicate-policy"; duplicate.edges.append(edge)
        XCTAssertThrowsError(try compile(duplicate))
        var bad = document; let index = try XCTUnwrap(bad.nodes.firstIndex { $0.address == policyAddress }); bad.nodes[index].payload = .gradientPolicyRecipe(type: "trainability", configurationVersion: 1, config: [:]); XCTAssertThrowsError(try compile(bad))
        bad = document; bad.nodes[index].payload = .gradientPolicyRecipe(type: "gradient_policy", configurationVersion: 2, config: [:]); XCTAssertThrowsError(try compile(bad))
    }
    func testRetiredGradientEditingRefusesWhileExactUpgradeKeepsCanonicalConfig() throws {
        let (document, _, _) = try fixture(), (model, data) = try registries()
        let old = try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        let node = try XCTUnwrap(document.nodes.first { $0.address == policyAddress })
        XCTAssertThrowsError(try WorkflowEditing.apply(.insert(node),to:old,modelRegistry:model,dataRegistry:data))
        XCTAssertThrowsError(try WorkflowEditing.apply(.configure(address:policyAddress,config:try GradientPolicyConfiguration(values:["accumulation_steps":3]).values),to:document,modelRegistry:model,dataRegistry:data))
        let migrated=try RetiredTrainingRecipeMigration.migrate(document)
        let plan=try WorkflowCompiler.compile(migrated,train:WorkflowExamples.trainAddress,modelRegistry:model,dataRegistry:data,contractVersion:13)
        XCTAssertEqual(plan.gradientPolicyRecipe?.configuration.accumulationSteps,2)
        XCTAssertEqual(plan.gradientPolicyRecipe?.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(migrated)),migrated)
    }
}
