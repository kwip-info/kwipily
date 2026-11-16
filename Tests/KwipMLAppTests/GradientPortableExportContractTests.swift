import XCTest
import KwipMLCore
@testable import KwipMLApp

final class GradientPortableExportContractTests: XCTestCase {
    func testSelectedHelpersKeepExactBytesAndBoundedUTF8() throws {
        let gradient = Data("# sanitized accumulation helper\n".utf8), trainability = Data("# native flags helper\n".utf8)
        let files = ["gradient_policy.py": gradient, "trainability_recipe.py": trainability]
        XCTAssertEqual(try WorkflowHistoryReader.portablePolicyHelpers(gradientSelected: true, trainabilitySelected: true, artifacts: files), files)
        XCTAssertThrowsError(try WorkflowHistoryReader.portablePolicyHelpers(gradientSelected: true, trainabilitySelected: false, artifacts: [:]))
        XCTAssertThrowsError(try WorkflowHistoryReader.portablePolicyHelpers(gradientSelected: true, trainabilitySelected: false, artifacts: ["gradient_policy.py": Data(repeating: 32, count: 131073)]))
        XCTAssertThrowsError(try WorkflowHistoryReader.portablePolicyHelpers(gradientSelected: true, trainabilitySelected: false, artifacts: ["gradient_policy.py": Data([255])]))
        XCTAssertEqual(try WorkflowHistoryReader.portablePolicyHelpers(gradientSelected: true, trainabilitySelected: false, artifacts: ["gradient_policy.py": Data(repeating: 32, count: 131072)]).count, 1)
    }
    func testAbsentPoliciesAndForeignDerivativeArtifactRefuseWithoutInventedFiles() throws {
        XCTAssertEqual(try WorkflowHistoryReader.portablePolicyHelpers(gradientSelected: false, trainabilitySelected: false, artifacts: [:]), [:])
        let manifest = WorkflowTrainedExportManifest(runID: "run", requestIdentity: "request", sourceRevision: 0, modelManifestSHA256: String(repeating: "a", count: 64), files: [])
        guard case .object(let raw) = try JSONDecoder().decode(JSONValue.self, from: WorkflowArchiveCodec.encode(manifest)) else { return XCTFail("Missing legacy manifest") }
        XCTAssertNil(raw["gradientPolicy"]); XCTAssertNil(raw["trainability"])
        for files in [["gradient_policy.py": Data("x".utf8)], ["derivative-results.json": Data("{}".utf8)]] {
            XCTAssertThrowsError(try WorkflowHistoryReader.portablePolicyHelpers(gradientSelected: false, trainabilitySelected: false, artifacts: files))
        }
    }
    func testPortablePublicProvenancePinsFullRecipeWithoutRequestsOrUpstreamArrays() throws {
        let source = WorkflowNodeAddress(domain: .control, localID: "policy")
        let recipe = GradientPolicyRecipe(configuration: try GradientPolicyConfiguration(values: GradientLearningTask.selectedInfluence.gradientValues), source: source)
        let policy = try WorkflowExportGradientPolicy(recipe)
        let bytes = try WorkflowArchiveCodec.encode(policy)
        guard case .object(let fields) = try JSONDecoder().decode(JSONValue.self, from: bytes) else { return XCTFail("Missing policy provenance") }
        XCTAssertEqual(Set(fields.keys), ["source", "recipeSHA256", "accumulationSteps", "gradientScale"])
        XCTAssertEqual(fields["recipeSHA256"], .string(WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(recipe.configuration.recipeValue))))
        let flags = TrainabilityRecipe(configuration: try TrainabilityConfiguration(values: GradientLearningTask.keepSettings.trainabilityValues), source: .init(domain: .control, localID: "flags"))
        let exported = try WorkflowExportTrainability(flags)
        XCTAssertEqual(exported.owners, flags.configuration.owners)
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("requests"))
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("upstream"))
    }
    func testInferenceTemplateAllowsPolicyHelpersWithoutExecutingThem() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source = try String(contentsOf: root.appendingPathComponent("Sources/KwipMLApp/Resources/inference.py"), encoding: .utf8)
        let allowed = try XCTUnwrap(source.components(separatedBy: "ALLOWED = ").dropFirst().first?.components(separatedBy: "\n").first)
        XCTAssertTrue(allowed.contains("'gradient_policy.py'")); XCTAssertTrue(allowed.contains("'trainability_recipe.py'"))
        XCTAssertFalse(source.contains("import gradient_policy")); XCTAssertFalse(source.contains("import trainability_recipe"))
    }
}
