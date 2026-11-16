import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowDerivativeGeometryTests: XCTestCase {
    private func policy(_ change: (inout [String: JSONValue]) -> Void = { _ in }) throws -> GradientPolicyConfiguration {
        var values = GradientLearningTask.selectedInfluence.gradientValues
        guard case .array(var requests) = values["requests"], case .object(var request) = requests[0] else { throw WorkflowPhaseError("Fixture request") }
        change(&request); requests[0] = .object(request); values["requests"] = .array(requests)
        return try GradientPolicyConfiguration(values: values)
    }
    private func owner(_ name: String = "layers.first.weight", frozen: Bool = false) -> JSONValue {
        .object(["node": .string("first"), "parameter": .string("weight"), "name": .string(name),
                 "shape": .array([3, 2]), "dtype": .string("float32"), "requiresGrad": .bool(!frozen)])
    }
    private func identity(_ policy: GradientPolicyConfiguration, owners: [JSONValue]) -> WorkflowGradientPolicyIdentity {
        .init(gradientPolicy: policy, trainability: nil, fullOwnership: owners, optimizerOwnership: [], microbatchesPerEpoch: 2, optimizerUpdatesPerEpoch: 2)
    }
    private func resolve(_ p: GradientPolicyConfiguration, owners: [JSONValue]? = nil, graph: GraphDocument? = nil) throws -> [DerivativeRequestGeometry] {
        try WorkflowDerivativeGeometry.resolve(policy: p, graph: graph ?? GradientLearningExamples.modelGraph(widths: [2, 3, 2]),
            registry: .builtIn(), context: .explicitObjective, identity: identity(p, owners: owners ?? [owner()]))
    }
    func testActualConfiguredModelGeometryPreservesTargetOrderAndNativeParameterShape() throws {
        let p = try policy(), result = try resolve(p)
        XCTAssertEqual(result.count, 1)
        let request = try XCTUnwrap(result.first)
        XCTAssertEqual(request.requestID, "selected-effect"); XCTAssertEqual(request.outputShape, [3, 2])
        XCTAssertEqual(request.targets.map(\.name), ["input.out", "layers.first.weight"])
        XCTAssertEqual(request.targets.map(\.shape), [[3, 2], [3, 2]])
        XCTAssertEqual(request.targets.map(\.dtype), [.float32, .float32])
        XCTAssertEqual(request.targets.map(\.requiresGrad), [true, true])
        if case .object(let selected) = p.requests[0], case .array(let targets) = selected["targets"] {
            XCTAssertEqual(request.targets.map(\.target), targets)
        } else { XCTFail("Missing selected target identities") }
        let one = try policy { $0["row_count"] = 1 }
        let changed = try XCTUnwrap(resolve(one).first)
        XCTAssertEqual(changed.outputShape, [1, 2]); XCTAssertEqual(changed.targets.map(\.shape), [[1, 2], [3, 2]])
    }
    func testFrozenNativeFlagIsRetainedAndEmptyRequestsNeedNoInventedGeometry() throws {
        let p = try policy { $0["unavailable"] = .string("reportUnavailable") }
        let result = try XCTUnwrap(resolve(p, owners: [owner(frozen: true)]).first)
        XCTAssertFalse(try XCTUnwrap(result.targets.last).requiresGrad)
        XCTAssertEqual(try XCTUnwrap(result.targets.last).shape, [3, 2])
        let empty = try GradientPolicyConfiguration()
        XCTAssertTrue(try resolve(empty).isEmpty)
    }
    func testMissingConfiguredPortsAndIntermediateInputTargetsRefuse() throws {
        let missing = try policy { $0["output"] = .object(["node": .string("second"), "port": .string("not-an-output")]) }
        XCTAssertThrowsError(try resolve(missing))
        let intermediate = try policy { $0["targets"] = .array([.object(["kind": .string("input"), "node": .string("first"), "port": .string("out")])]) }
        XCTAssertThrowsError(try resolve(intermediate))
        let inputPort = try policy { $0["targets"] = .array([.object(["kind": .string("input"), "node": .string("input"), "port": .string("in")])]) }
        XCTAssertThrowsError(try resolve(inputPort))
    }
    func testOwnershipAliasesMissingOwnersAndDuplicateGraphIDsRefuse() throws {
        let p = try policy()
        XCTAssertThrowsError(try resolve(p, owners: []))
        XCTAssertThrowsError(try resolve(p, owners: [owner(), owner()]))
        XCTAssertThrowsError(try resolve(p, owners: [owner("input.out")]))
        var graph = try GradientLearningExamples.modelGraph(widths: [2, 3, 2]); graph.nodes.append(graph.nodes[0])
        XCTAssertThrowsError(try resolve(p, graph: graph))
        guard case .object(var malformed) = owner() else { throw WorkflowPhaseError("Fixture owner") }; malformed.removeValue(forKey: "requiresGrad")
        XCTAssertThrowsError(try resolve(p, owners: [.object(malformed)]))
    }
}
