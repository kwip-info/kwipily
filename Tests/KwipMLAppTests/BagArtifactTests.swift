import XCTest
import Darwin
@testable import KwipMLCore
@testable import KwipMLApp

final class BagArtifactTests: XCTestCase {
    struct Fixture {
        let retention:Bool
        let root: URL
        let selection: WorkflowPreparedSelection
        let mapping: BagInputMapping
        let requirement: BagWorkflowInputRequirement
        let manifest: [String: JSONValue]
        let predictions: Data
        let target: BagArtifactTarget
        let device: ContractDevice
    }
    func fixture(objective: Bool = false, capture: Bool = true, activation: Bool = true, device: String = "cpu", retention:Bool=true) throws -> Fixture {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let root = folder.resolvingSymlinksInPath()
        let process = Process(), log = Pipe()
        process.executableURL = repo.appendingPathComponent(".venv/bin/python"); process.currentDirectoryURL = repo
        process.arguments = ["-B", "-c", Self.script, root.path, objective ? "yes" : "no", capture ? "yes" : "no", activation ? "yes" : "no", device,retention ? "yes":"no"]
        process.standardOutput = log; process.standardError = log
        try process.run(); let deadline = Date().addingTimeInterval(40)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.025) }
        if process.isRunning { _ = Darwin.kill(process.processIdentifier, SIGKILL) }; process.waitUntilExit()
        let output = String(decoding: log.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        XCTAssertEqual(process.terminationStatus, 0, output)
        let manifest = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: root.appendingPathComponent("training/artifacts/manifest.json")))
        let summary = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: root.appendingPathComponent("fixture.json")))
        func decode<T: Decodable>(_ type: T.Type, _ value: JSONValue) throws -> T { try JSONDecoder().decode(type, from: WorkflowPhaseArtifacts.encode(value)) }
        let reference = try decode(PreparedDataReference.self, XCTUnwrap(summary["reference"]))
        let mapping = try BagInputMapping(value: XCTUnwrap(summary["mapping"]))
        guard case .object(let config) = summary["sourceConfiguration"] else { throw WorkflowPhaseError("Missing actual source configuration") }
        let requirement = BagWorkflowInputRequirement(sourceNodeID: "source", configuration: try BagSourceConfiguration(values: config), numEmbeddings: 5, embeddingDim: 2)
        let selection: WorkflowPreparedSelection, target: BagArtifactTarget
        if objective {
            let configuration = try LossConfiguration.decodeRecipe(XCTUnwrap(summary["recipe"]))
            let roles = try decode(LossTargetMapping.self, XCTUnwrap(summary["targetMapping"]))
            let selected = PreparedObjectiveSelection(reference: reference, objectiveRecipe: configuration.recipeValue, targetMapping: roles)
            let recipe = try LossRecipe(configuration: configuration, source: .init(domain: .control, localID: "objective"), targetMapping: roles)
            selection = .objective(selected); target = .objective(selected, recipe)
        } else {
            selection = .legacy(.init(manifestPath: reference.manifestPath, manifestSHA256: reference.manifestSHA256, graphDigest: reference.graphDigest, trainingNode: reference.trainingNode, trainingPort: reference.trainingPort, validationNode: reference.validationNode, validationPort: reference.validationPort, featureColumns: mapping.indicesColumns, targetColumn: "label")); target = .binary("label")
        }
        return Fixture(retention:retention,root: root, selection: selection, mapping: mapping, requirement: requirement, manifest: manifest, predictions: retention ? try Data(contentsOf: root.appendingPathComponent("training/artifacts/predictions.json")):Data(), target: target, device: device == "mps" ? .mps : .cpu)
    }
    func context(_ f: Fixture, capture: Bool = true, activation: Bool = true) throws -> WorkflowBagArtifactContext {
        let evidence = try WorkflowBagPreparedEvidence.preparedEvidence(root: f.root, selection: f.selection, mapping: f.mapping, requirement: f.requirement)
        return .init(requirement: f.requirement, mapping: f.mapping, preparation: evidence, runID: try XCTUnwrap(f.manifest["runID"]?.stringValue), device: f.device, outputContract: try .init(sample: [2], dtype: .float32), target: f.target, policy: .init(activationValues: activation && f.retention, predictionRows: capture && f.retention,retainComponentOutputs:f.retention ? nil:false,retainDiagnosticDetails:f.retention ? nil:false), explicitSampleCapture: true)
    }
    private func preview(_ f: Fixture, context c: WorkflowBagArtifactContext) throws -> BagPredictionPreview {
        try WorkflowBagArtifacts.predictions(data: f.predictions, metadata: XCTUnwrap(f.manifest["predictionPreview"]), context: c)
    }
    func testActualBinaryPreparedPacketIdentityAndEmptyWholeBagRows() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let c = try context(f); XCTAssertEqual(c.preparation.training.count, 16); XCTAssertEqual(c.preparation.validation.count, 8)
        try WorkflowBagArtifacts.validateIdentity(manifest: f.manifest, context: c)
        let p = try preview(f, context: c); XCTAssertEqual(p.rows.count, 8); XCTAssertTrue(p.rows.contains { $0.input.indices.isEmpty })
        XCTAssertEqual(p.rows.map(\.input), c.preparation.validation.prefix.map(\.input))
        XCTAssertTrue(p.rows.allSatisfy { $0.label != nil && $0.predicted != nil && $0.output == nil })
    }
    func testActualObjectivePacketRowsRetainOutputsTargetsAndUnreducedValues() throws {
        let f = try fixture(objective: true); defer { try? FileManager.default.removeItem(at: f.root) }
        let c = try context(f); try WorkflowBagArtifacts.validateIdentity(manifest: f.manifest, context: c)
        let p = try preview(f, context: c)
        XCTAssertEqual(p.rows.count, 8); XCTAssertTrue(p.rows.contains { $0.input.indices.isEmpty })
        XCTAssertTrue(p.rows.allSatisfy { $0.label == nil && $0.decoded == nil && $0.output?.shape == [2] && $0.objectiveOutput?.shape == [2] && $0.targets["target"]?.shape == [] })
    }
    func testActualMPSObjectivePacketUsesSameTypedReader() throws {
        let f = try fixture(objective: true, device: "mps"); defer { try? FileManager.default.removeItem(at: f.root) }
        let c = try context(f); try WorkflowBagArtifacts.validateIdentity(manifest: f.manifest, context: c)
        let p = try preview(f, context: c)
        XCTAssertEqual(p.rows.count, 8); XCTAssertTrue(p.rows.contains { $0.input.indices.isEmpty })
        XCTAssertTrue(p.rows.allSatisfy { $0.output?.shape == [2] && $0.targets["target"]?.shape == [] })
    }
    func testAllFourIndependentCapturePermissionsRemainHonest() throws {
        for capture in [false, true] { for activation in [false, true] {
            let f = try fixture(capture: capture, activation: activation); defer { try? FileManager.default.removeItem(at: f.root) }
            let c = try context(f, capture: capture, activation: activation); try WorkflowBagArtifacts.validateIdentity(manifest: f.manifest, context: c)
            let p = try preview(f, context: c); XCTAssertEqual(p.rows.count, capture ? 8 : 0); XCTAssertEqual(p.captureStatus, capture ? .captured : .omittedByPolicy)
        } }
    }
    func testRepinnedWrongRowRoleIdentityAndPrivacyAreRefused() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }; let c = try context(f)
        let original = try JSONDecoder().decode([String: JSONValue].self, from: f.predictions)
        for key in ["sourceID", "configuration", "ports"] {
            var value = original
            if case .object(var input) = value["predictionInput"] { input[key] = .null; value["predictionInput"] = .object(input) }
            XCTAssertThrowsError(try changedPreview(value, f: f, context: c), key)
        }
        var changed = original
        if case .array(var rows) = changed["predictionRows"] { rows.swapAt(0, 1); changed["predictionRows"] = .array(rows) }
        XCTAssertThrowsError(try changedPreview(changed, f: f, context: c))
        XCTAssertThrowsError(try preview(f, context: context(f, capture: false)))
        var manifest = f.manifest
        if case .object(var p) = manifest["provenance"] { p["modelInput"] = .object(["dtype": .string("float32"), "sampleShape": .array([2])]); manifest["provenance"] = .object(p) }
        XCTAssertThrowsError(try WorkflowBagArtifacts.validateIdentity(manifest: manifest, context: c))
        for key in ["mapping", "sourceConfiguration", "targetMapping", "lineage", "observationIDs", "membershipDigest"] {
            var altered = f.manifest
            if case .object(var provenance) = altered["provenance"], case .object(var sources) = provenance["bagSources"], case .object(var training) = sources["training"] {
                training[key] = .null; sources["training"] = .object(training); provenance["bagSources"] = .object(sources); altered["provenance"] = .object(provenance)
                if case .object(var checkpoint) = altered["checkpoint"], case .object(var contract) = checkpoint["contract"], case .object(var data) = contract["data"], case .object(var bag) = data["bagInput"] {
                    bag["sources"] = .object(sources); data["bagInput"] = .object(bag); contract["data"] = .object(data); checkpoint["contract"] = .object(contract); altered["checkpoint"] = .object(checkpoint)
                }
            }
            XCTAssertThrowsError(try WorkflowBagArtifacts.validateIdentity(manifest: altered, context: c), key)
        }
    }
    private func changedPreview(_ object: [String: JSONValue], f: Fixture, context: WorkflowBagArtifactContext) throws -> BagPredictionPreview {
        let bytes = try WorkflowPhaseArtifacts.encode(object)
        let metadata: JSONValue = .object(["schemaVersion": 3, "file": .string("predictions.json"), "sha256": .string(WorkflowPhaseArtifacts.hash(bytes)), "bytes": .integer(Int64(bytes.count))])
        return try WorkflowBagArtifacts.predictions(data: bytes, metadata: metadata, context: context)
    }
    func testChangedPreparedBytesRefuseBeforeContextCreation() throws {
        let f = try fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let root = f.root.appendingPathComponent("preparation/artifacts"), manifest = try JSONDecoder().decode([String: JSONValue].self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
        guard case .array(let entries) = manifest["artifacts"], let found = entries.first(where: { if case .object(let v) = $0 { return v["nodeID"] == .string("p") && v["port"] == .string("validation") }; return false }), case .object(let value) = found else { return XCTFail("Missing actual selected table") }
        let path = root.appendingPathComponent(try XCTUnwrap(value["path"]?.stringValue)); var bytes = try Data(contentsOf: path); bytes.append(32); try bytes.write(to: path)
        XCTAssertThrowsError(try context(f))
    }
    private static let script = #"""
import hashlib,json,shutil,sys
from pathlib import Path
from runner import engine
from runner.tests.test_prepared_bag_loader import fixture,options,load
from runner.tests.test_bag_source_graph import graph,registry
from runner.loss_configuration import canonical_configuration
root=Path(sys.argv[1]);objective,capture,activation=[v=='yes' for v in sys.argv[2:5]];device=sys.argv[5]
source=fixture(root/'source');args=options(source)
recipe=canonical_configuration('loss_mse',{'reduction':'none','target_broadcast':True})
mapping={'schemaVersion':1,'componentID':'loss_mse','inputColumns':['i1','i2'],'roles':[{'role':'target','columns':['desired'],'dtype':'float32','sampleShape':[]}],'tripletSourceGroups':[]}
if objective:
 recipe={'schemaVersion':1,'componentID':'loss_mse','configurationVersion':1,'config':recipe}
 admitted=load(source,target_column=None,objective_mapping=mapping,objective_recipe=recipe,prediction_shape=[2])
else:admitted=load(source)
reg=registry.__wrapped__(root);value=graph();value['nodes'][0]['config']['maximum_items_per_bag']=2
value['name']='Prepared bag artifact fixture'
for node in value['nodes']:node.update(name=node['id'],x=0,y=0)
kwargs=dict(objective_recipe=recipe,target_mapping=mapping) if objective else {}
engine.run(value,root/'training/artifacts',registry=reg,prepared=admitted,device=device,epochs=1,batch_size=3,sample_capture=({'schemaVersion':1,'predictionRows':capture,'activationValues':activation} if sys.argv[6]=='yes' else dict(schemaVersion=3,predictionRows=False,activationValues=False,performanceMeasurements=False,retainComponentOutputs=False,retainDiagnosticDetails=False)),**kwargs)
shutil.copytree(source,root/'preparation/artifacts')
raw=(source/'manifest.json').read_bytes();manifest=json.loads(raw)
summary={'reference':{'manifestPath':str(root/'preparation/artifacts/manifest.json'),'manifestSHA256':hashlib.sha256(raw).hexdigest(),'graphDigest':manifest['graphDigest'],'trainingNode':'p','trainingPort':'training','validationNode':'p','validationPort':'validation'},'mapping':args['mapping'],'sourceConfiguration':args['source_config'],'recipe':recipe,'targetMapping':mapping}
(root/'fixture.json').write_text(json.dumps(summary))
"""#
}
