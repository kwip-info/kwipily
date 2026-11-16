import XCTest
import KwipMLCore
@testable import KwipMLApp

final class InitializationCaptureTransportTests: XCTestCase {
    private let registryPin = String(repeating: "a", count: 64)
    private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func configuration(node: String = "dense", seed: Int64 = 42) throws -> InitializationConfiguration {
        try .init(values: ["seed": .integer(seed), "rules": .array([.object([
            "owner": .object(["node": .string(node), "parameter": .string("weight")]),
            "algorithm": .string("normal"), "settings": .object(["mean": 0, "std": 0.2])])])])
    }
    private func plan(selected: Bool = true, seed: Int64 = 42) throws -> WorkflowExecutionPlan {
        var document = try WorkflowExamples.preparedParallelClassifier(device: "cpu")
        document.schemaVersion = 8
        var plan = try WorkflowCompiler.compile(document, train: WorkflowExamples.trainAddress, modelRegistry: .builtIn(), dataRegistry: .load(from: repository.appendingPathComponent("runner/dataflow-operators.json")), contractVersion: 7)
        if selected {
            let owner = try XCTUnwrap(plan.model.graph.nodes.first { $0.type == "linear" })
            plan.initializationRecipe = .init(configuration: try configuration(node: owner.id, seed: seed), source: .init(domain: .control, localID: "starting-settings"))
        }
        return plan
    }
    private func object<T: Encodable>(_ value: T) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: WorkflowArchiveCodec.encode(value)) as? [String: Any])
    }
    private func capture(_ plan: WorkflowExecutionPlan, schema: Int = 9, compiler: Int = 8) throws -> WorkflowArchiveCapture {
        func file(_ path: String) -> WorkflowArchiveFile { .init(path: path, bytes: 10, sha256: registryPin) }
        return .init(schemaVersion: schema, compilerContractVersion: compiler, projectID: plan.projectID, directoryID: UUID().uuidString, runID: UUID().uuidString, requestIdentity: registryPin, revision: plan.revision, train: plan.train, workflow: file("workflow.json"), modelRegistry: file("model-registry.json"), dataRegistry: file("data-registry.json"), modelRegistrySHA256: registryPin, dataRegistrySHA256: registryPin, runtimeEvidence: file("runtime-evidence.json"), sources: plan.sourceAddresses.enumerated().map { .init(address: $0.element, path: "inputs/source-\($0.offset).csv", bytes: 10, sha256: registryPin) }, plan: try WorkflowArchivePlan(plan, compilerContractVersion: compiler))
    }
    private func wire(_ configuration: InitializationConfiguration) throws -> String { String(decoding: try WorkflowPhaseArtifacts.encode(configuration.recipeValue), as: UTF8.self) }
    func testSchemaNineCompilerEightArchiveRetainsQualifiedRecipeAndDetectsIdentityChanges() throws {
        let selected = try plan(), archive = try WorkflowArchivePlan(selected, compilerContractVersion: 8)
        let raw = try XCTUnwrap(object(archive)["initializationRecipe"] as? [String: Any])
        XCTAssertEqual(Set(raw.keys), ["source", "configuration"])
        let restored = try JSONDecoder().decode(WorkflowArchivePlan.self, from: WorkflowArchiveCodec.encode(archive))
        XCTAssertTrue(try restored.matches(selected, compilerContractVersion: 8))
        XCTAssertFalse(try restored.matches(plan(seed: 41), compilerContractVersion: 8))
        var otherSource = selected
        otherSource.initializationRecipe = .init(configuration: try XCTUnwrap(selected.initializationRecipe).configuration, source: .init(domain: .control, localID: "other-start"))
        XCTAssertFalse(try restored.matches(otherSource, compilerContractVersion: 8))
        let captured = try capture(selected)
        XCTAssertEqual(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(captured)), captured)
        XCTAssertEqual(WorkflowArchiveCodec.trainingCompilerContractVersion(forWorkflowSchema: 9), 8)
    }
    func testOldAbsenceIsOmittedAndOldCompilersCannotDropSelectedPolicy() throws {
        let implicit = try plan(selected: false)
        for version in [4, 5, 6, 7, 8] {
            XCTAssertNil(try object(WorkflowArchivePlan(implicit, compilerContractVersion: version))["initializationRecipe"])
            if version < 8 { XCTAssertThrowsError(try WorkflowArchivePlan(plan(), compilerContractVersion: version)) }
        }
        let old = try capture(implicit, schema: 8, compiler: 7)
        let bytes = try WorkflowArchiveCodec.encodeCapture(old)
        XCTAssertFalse(String(decoding: bytes, as: UTF8.self).contains("initializationRecipe"))
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(bytes)), bytes)
    }
    func testArchiveRefusesForeignSourceUnknownMissingControlsAndVersionPairs() throws {
        let valid = try object(capture(plan()))
        for mutation in ["unknown", "missingSettings", "foreignSource", "oldCompiler"] {
            var raw = valid
            var archived = try XCTUnwrap(raw["plan"] as? [String: Any])
            var recipe = try XCTUnwrap(archived["initializationRecipe"] as? [String: Any])
            if mutation == "oldCompiler" { raw["compilerContractVersion"] = 7 }
            if mutation == "unknown" { recipe["unchecked"] = true }
            if mutation == "foreignSource" {
                var source = try XCTUnwrap(recipe["source"] as? [String: Any]); source["domain"] = "model"; recipe["source"] = source
            }
            if mutation == "missingSettings" {
                var configuration = try XCTUnwrap(recipe["configuration"] as? [String: Any]); var config = try XCTUnwrap(configuration["config"] as? [String: Any]); var rules = try XCTUnwrap(config["rules"] as? [[String: Any]]); var settings = try XCTUnwrap(rules[0]["settings"] as? [String: Any]); settings.removeValue(forKey: "std"); rules[0]["settings"] = settings; config["rules"] = rules; configuration["config"] = config; recipe["configuration"] = configuration
            }
            archived["initializationRecipe"] = recipe; raw["plan"] = archived
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject: raw)), mutation)
        }
    }
    func testWorkflowPhaseAcceptsCanonicalInitializationOnlyForTrainingWithBoundedUniqueFlag() throws {
        let root = URL(fileURLWithPath: "/private/kwip-init-phase"), project = UUID().uuidString, directory = UUID().uuidString
        let run = root.appendingPathComponent("workflow-runs/\(project)/\(directory)")
        let base = ["--graph", run.appendingPathComponent("training/request.json").path, "--output", run.appendingPathComponent("training/artifacts").path]
        let raw = try wire(configuration())
        func validate(_ extra: [String]) throws { try WorkflowPhaseArguments.validate(base + extra, workspace: root, projectID: project, directoryID: directory, phase: .training) }
        XCTAssertNoThrow(try validate(["--initialization-recipe-json", raw]))
        XCTAssertThrowsError(try validate(["--initialization-recipe-json", raw, "--initialization-recipe-json", raw]))
        XCTAssertThrowsError(try validate(["--initialization-recipe-json", String(repeating: " ", count: 32769)]))
        XCTAssertThrowsError(try validate(["--initialization-recipe-json", "{}"] ))
        var badValue = try JSONDecoder().decode([String: JSONValue].self, from: Data(raw.utf8))
        guard case .object(var badConfig) = badValue["config"] else { return XCTFail("Missing canonical initialization config") }
        badConfig["seed"] = .bool(true); badValue["config"] = .object(badConfig)
        XCTAssertThrowsError(try validate(["--initialization-recipe-json", String(decoding: WorkflowPhaseArtifacts.encode(badValue), as: UTF8.self)]))
        let prep = ["--execute-dataflow", run.appendingPathComponent("preparation/request.json").path, "--source-bindings", run.appendingPathComponent("preparation/bindings.json").path, "--output", run.appendingPathComponent("preparation/artifacts").path, "--initialization-recipe-json", raw]
        XCTAssertThrowsError(try WorkflowPhaseArguments.validate(prep, workspace: root, projectID: project, directoryID: directory, phase: .preparation))
    }
    func testStandaloneTrainingAdmitsFullRecipeWithoutRelaxingOtherInlineBoundsOrModes() throws {
        let root = URL(fileURLWithPath: "/private/tmp/kwip/workflow-operations/\(UUID().uuidString)/\(UUID().uuidString)")
        let base = ["--graph", root.appendingPathComponent("input/graph.json").path, "--output", root.appendingPathComponent("artifacts").path, "--epochs", "1", "--seed", "42", "--device", "cpu", "--batch-size", "4", "--learning-rate", "0.03", "--sample-limit", "0", "--split-seed", "42", "--validation-percent", "20"]
        let files = [WorkflowArchiveFile(path: "input/graph.json", bytes: 2, sha256: registryPin)]
        func validate(_ extra: [String]) throws { try WorkflowModelTrainingToolContract.validate(arguments: base + extra, files: files, directory: root) }
        XCTAssertNoThrow(try validate(["--initialization-recipe-json", wire(configuration())]))
        let large = try configuration(node: String(repeating: "n", count: 9000))
        XCTAssertNoThrow(try validate(["--initialization-recipe-json", wire(large)]), "Only the new recipe gets its32768-byte limit.")
        XCTAssertThrowsError(try validate(["--initialization-recipe-json", "{}"] ))
        XCTAssertThrowsError(try validate(["--initialization-recipe-json", String(repeating: " ", count: 32769)]))
        XCTAssertThrowsError(try validate(["--initialization-recipe-json", wire(configuration()), "--preview"]))
        XCTAssertThrowsError(try validate(["--sample-capture-json", String(repeating: "x", count: 8193)]))
    }
}
