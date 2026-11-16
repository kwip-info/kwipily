import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BagLaunchArgumentsTests: XCTestCase {
    private let reference = PreparedDataReference(manifestPath: "/tmp/preparation/manifest.json", manifestSHA256: String(repeating: "a", count: 64), graphDigest: String(repeating: "b", count: 64), trainingNode: "split", trainingPort: "training", validationNode: "split", validationPort: "validation")
    private func selection(_ plan: WorkflowExecutionPlan) throws -> WorkflowPreparedSelection {
        if let recipe = plan.lossRecipe {
            return .objective(.init(reference: reference, objectiveRecipe: recipe.configuration.recipeValue, targetMapping: recipe.targetMapping, scalarization: plan.objectiveScalarization))
        }
        return .legacy(.init(manifestPath: reference.manifestPath, manifestSHA256: reference.manifestSHA256, graphDigest: reference.graphDigest, trainingNode: reference.trainingNode, trainingPort: reference.trainingPort, validationNode: reference.validationNode, validationPort: reference.validationPort, featureColumns: plan.featureColumns, targetColumn: try XCTUnwrap(plan.targetColumn)))
    }
    private func options(_ args: [String]) -> [String: String] {
        Dictionary(uniqueKeysWithValues: stride(from: 0, to: args.count, by: 2).map { (args[$0], args[$0 + 1]) })
    }
    func testBothTargetModesAndAllPacketLayoutsEmitCanonicalMappingWithoutDenseFeatureFlag() throws {
        let fixture = BagWorkflowArchiveTests(), registry = try fixture.registry()
        for packed in [false, true] { for weighted in [false, true] { for objective in [false, true] {
            let plan = try fixture.compile(fixture.fixture(packed: packed, weighted: weighted, objective: objective))
            let mapping = try XCTUnwrap(plan.bagInput)
            let args = try WorkflowPhaseArtifacts.preparedArguments(selection(plan), bagInput: mapping, model: plan.model.graph, registry: registry)
            let values = options(args)
            XCTAssertNil(values["--feature-columns-json"])
            XCTAssertEqual(values["--bag-input-mapping-json"], String(decoding: try WorkflowPhaseArtifacts.encode(mapping.value), as: UTF8.self))
            XCTAssertEqual(values["--prepared-manifest-sha256"], reference.manifestSHA256)
            XCTAssertEqual(values["--prepared-training-node"], reference.trainingNode)
            XCTAssertEqual(values["--target-column"], objective ? nil : "label")
            XCTAssertEqual(values["--objective-target-mapping-json"] != nil, objective)
        } } }
    }
    func testMappingMustMatchActualSourceAndSelectedTargets() throws {
        let fixture = BagWorkflowArchiveTests(), registry = try fixture.registry()
        let plan = try fixture.compile(fixture.fixture(packed: true, weighted: true))
        let chosen = try selection(plan)
        let fixed = try BagInputMapping(value: .object(["schemaVersion": 1, "indicesColumns": .array([.string("i0"), .string("i1")])]))
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.preparedArguments(chosen, bagInput: fixed, model: plan.model.graph, registry: registry))
        var dense = plan.model.graph; dense.nodes[0] = .init(id: "bags", type: "input", config: ["features": 2])
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.preparedArguments(chosen, bagInput: XCTUnwrap(plan.bagInput), model: dense, registry: registry))
        let foreign = WorkflowPreparedSelection.legacy(.init(manifestPath: reference.manifestPath, manifestSHA256: reference.manifestSHA256, graphDigest: reference.graphDigest, trainingNode: reference.trainingNode, trainingPort: reference.trainingPort, validationNode: reference.validationNode, validationPort: reference.validationPort, featureColumns: ["foreign", "i1"], targetColumn: "label"))
        XCTAssertThrowsError(try WorkflowPhaseArtifacts.preparedArguments(foreign, bagInput: XCTUnwrap(plan.bagInput), model: plan.model.graph, registry: registry))
    }
    func testPhaseAdmissionAcceptsCanonicalBagMappingAndRefusesMixedDuplicateAndNoncanonicalFlags() throws {
        let workspace = URL(fileURLWithPath: "/tmp/kwip-bag-launch"), project = UUID().uuidString, run = UUID().uuidString
        let folder = workspace.appendingPathComponent("workflow-runs/\(project)/\(run)/training")
        let raw = #"{"indicesColumns":["i0","i1"],"schemaVersion":1}"#
        let args = ["--graph", folder.appendingPathComponent("request.json").path, "--output", folder.appendingPathComponent("artifacts").path, "--prepared-dataflow", reference.manifestPath, "--prepared-manifest-sha256", reference.manifestSHA256, "--prepared-graph-digest", reference.graphDigest, "--prepared-training-node", "split", "--prepared-training-port", "training", "--prepared-validation-node", "split", "--prepared-validation-port", "validation", "--target-column", "label", "--bag-input-mapping-json", raw]
        try WorkflowPhaseArguments.validate(args, workspace: workspace, projectID: project, directoryID: run, phase: .training)
        for bad in [args + ["--feature-columns-json", #"["i0","i1"]"#], args + ["--bag-input-mapping-json", raw], Array(args.dropLast()) + [" " + raw], Array(args.dropLast()) + [String(repeating: " ", count: 32769)], Array(args.dropLast()) + [#"{"indicesColumns":["i0","i1"],"schemaVersion":1,"schemaVersion":1}"#]] {
            XCTAssertThrowsError(try WorkflowPhaseArguments.validate(bad, workspace: workspace, projectID: project, directoryID: run, phase: .training))
        }
    }
}
