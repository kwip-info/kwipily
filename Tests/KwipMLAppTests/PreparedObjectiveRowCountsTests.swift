import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparedObjectiveRowCountsTests: XCTestCase {
    private let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    private func fixture(_ task:LossLearningTask, at root:URL) throws -> WorkflowPreparedSelection {
        let registry=try ComponentRegistry.builtIn()
        let dataRegistry=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        let plan=try WorkflowCompiler.compile(WorkflowExamples.preparedLossObjective(task:task,device:"cpu"),train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:dataRegistry,contractVersion:12)
        let graph=root.appendingPathComponent("dataflow.json"),bindings=root.appendingPathComponent("bindings.json")
        try DataFlowCodec.encodeGraph(plan.dataflow).write(to:graph)
        let resource=try XCTUnwrap(BundledLossComponents.entries[task.componentID]?.learning?.sourceResource)
        try WorkflowPhaseArtifacts.encode(["observations":repo.appendingPathComponent("Sources/KwipMLApp/Resources/"+resource+".csv").path]).write(to:bindings)
        let output=root.appendingPathComponent("preparation/artifacts")
        try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        let log=root.appendingPathComponent("prepare.log");FileManager.default.createFile(atPath:log.path,contents:Data())
        let handle=try FileHandle(forWritingTo:log);defer {try? handle.close()}
        let worker=Process();worker.executableURL=repo.appendingPathComponent(".venv/bin/python");worker.currentDirectoryURL=repo
        guard FileManager.default.isExecutableFile(atPath:worker.executableURL!.path) else {throw XCTSkip("Actual local Python is required")}
        worker.arguments=[repo.appendingPathComponent("runner/main.py").path,"--execute-dataflow",graph.path,"--source-bindings",bindings.path,"--output",output.path]
        worker.standardOutput=handle;worker.standardError=handle;try worker.run()
        let deadline=Date().addingTimeInterval(40)
        while worker.isRunning && Date()<deadline {Thread.sleep(forTimeInterval:0.025)}
        if worker.isRunning {worker.terminate();XCTFail("Preparation exceeded its bounded deadline")};worker.waitUntilExit()
        let outputText=String(decoding:try Data(contentsOf:log),as:UTF8.self)
        XCTAssertEqual(worker.terminationStatus,0,outputText)
        return try WorkflowPhaseArtifacts.preparation(expectation:.init(plan:plan,modelRegistrySHA256:WorkflowArchiveCodec.registryHash(registry),dataRegistrySHA256:WorkflowArchiveCodec.registryHash(dataRegistry),compilerContractVersion:12),root:root).selection
    }
    private func temporary() throws -> URL {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true);return root
    }
    private func mutateValidation(_ root:URL,selection:WorkflowPreparedSelection,change: (inout [String:JSONValue])->Void) throws -> WorkflowPreparedSelection {
        guard case .objective(var selected)=selection else {throw WorkflowArchiveError("Expected objective")}
        let prefix=root.appendingPathComponent("preparation/artifacts"),manifestURL=prefix.appendingPathComponent("manifest.json")
        var manifest=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:manifestURL))
        guard case .array(var entries)=manifest["artifacts"],let index=entries.firstIndex(where:{ if case .object(let fields)=$0 {return fields["nodeID"] == .string(selected.reference.validationNode) && fields["port"] == .string(selected.reference.validationPort) && fields["kind"] == .string("table")};return false }),case .object(var entry)=entries[index],case .string(let name)=entry["path"] else {throw WorkflowArchiveError("Missing selected table")}
        let file=prefix.appendingPathComponent(name);var table=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:file));change(&table)
        let bytes=try WorkflowPhaseArtifacts.encode(table);try bytes.write(to:file)
        entry["bytes"] = .integer(Int64(bytes.count));entry["sha256"] = .string(WorkflowArchiveCodec.hash(bytes));entries[index] = .object(entry);manifest["artifacts"] = .array(entries)
        let pin=try WorkflowPhaseArtifacts.encode(manifest);try pin.write(to:manifestURL);selected.reference.manifestSHA256=WorkflowArchiveCodec.hash(pin)
        return .objective(selected)
    }
    func testAllEightActualPreparationsProvidePinnedCountsAndLegacyReaderRemainsCompatible() throws {
        for task in LossLearningTask.allCases {
            let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
            let selected=try fixture(task,at:root)
            let counts=try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:selected)
            XCTAssertEqual(counts.trainingRows,32,task.rawValue);XCTAssertEqual(counts.validationRows,16,task.rawValue)
            XCTAssertTrue(WorkflowArchiveCodec.isHash(counts.trainingTableSHA256));XCTAssertTrue(WorkflowArchiveCodec.isHash(counts.validationTableSHA256))
            XCTAssertEqual(try WorkflowFailedGradientEvidence.trainingRows(root:root,selection:selected),32)
            if task == .mse {
                let r=selected.reference
                let old=PreparedTrainingSelection(manifestPath:r.manifestPath,manifestSHA256:r.manifestSHA256,graphDigest:r.graphDigest,trainingNode:r.trainingNode,trainingPort:r.trainingPort,validationNode:r.validationNode,validationPort:r.validationPort,featureColumns:["signal","context"],targetColumn:"desired")
                XCTAssertEqual(try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:.legacy(old)).validationRows,16)
                XCTAssertEqual(try WorkflowFailedGradientEvidence.trainingRows(root:root,selection:old),32)
            }
        }
    }
    func testActualOrderedObservationIdentitiesSurvivePinnedPreparation() throws {
        let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
        let selected=try fixture(.mse,at:root)
        let prefix=root.appendingPathComponent("preparation/artifacts")
        let manifest=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:prefix.appendingPathComponent("manifest.json")))
        guard case .array(let entries)=manifest["artifacts"] else {return XCTFail("Missing actual preparation inventory")}
        func exactIDs(_ node:String,_ port:String)throws->[String] {
            let entry=try XCTUnwrap(entries.first {if case .object(let v)=$0 {return v["nodeID"]?.stringValue==node && v["port"]?.stringValue==port && v["kind"]?.stringValue=="table"};return false})
            guard case .object(let v)=entry else {throw WorkflowArchiveError("Missing table")}
            let file=try XCTUnwrap(v["path"]?.stringValue)
            let table=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:prefix.appendingPathComponent(file)))
            guard case .object(let lineage)=table["lineage"],case .array(let ids)=lineage["observationIDs"] else {throw WorkflowArchiveError("Missing ordered lineage")}
            return try ids.map {try XCTUnwrap($0.stringValue)}
        }
        let counts=try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:selected)
        let training=try exactIDs(selected.trainingNode,selected.trainingPort),validation=try exactIDs(selected.validationNode,selected.validationPort)
        XCTAssertEqual(counts.trainingRowIDs,training);XCTAssertEqual(counts.validationRowIDs,validation)
        XCTAssertEqual(counts.trainingRowIDs.count,counts.trainingRows);XCTAssertEqual(counts.validationRowIDs.count,counts.validationRows)
        // Preserve loader order, rather than sorting a membership set for a row-range request.
        XCTAssertNotEqual(training,training.sorted())
        XCTAssertEqual(Array(counts.trainingRowIDs.prefix(3)),Array(training.prefix(3)))
        XCTAssertTrue(Set(counts.trainingRowIDs).isDisjoint(with:counts.validationRowIDs))
    }
    func testStaleAndRepinnedConflictingRowsAndPartitionAreRefused() throws {
        let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
        let selection=try fixture(.mse,at:root)
        let original=try Data(contentsOf:root.appendingPathComponent("preparation/artifacts/node-003.json"))
        var changed=original;changed.append(32);try changed.write(to:root.appendingPathComponent("preparation/artifacts/node-003.json"))
        XCTAssertThrowsError(try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:selection))
        try original.write(to:root.appendingPathComponent("preparation/artifacts/node-003.json"))
        let fewer=try mutateValidation(root,selection:selection) {if case .array(var rows)=$0["rows"] {rows.removeLast();$0["rows"] = .array(rows)}}
        XCTAssertThrowsError(try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:fewer))
        try original.write(to:root.appendingPathComponent("preparation/artifacts/node-003.json"))
        let wrongRole=try mutateValidation(root,selection:fewer) {if case .object(var lineage)=$0["lineage"] {lineage["role"] = .string("training");$0["lineage"] = .object(lineage)}}
        XCTAssertThrowsError(try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:wrongRole))
    }
    func testRepinnedTypedColumnAndGraphDigestContradictionsAreRefused() throws {
        let root=try temporary();defer {try? FileManager.default.removeItem(at:root)}
        let selection=try fixture(.ctc,at:root)
        let changed=try mutateValidation(root,selection:selection) {if case .array(var columns)=$0["columns"],case .object(var first)=columns[0] {first["name"] = .string("unselected");columns[0] = .object(first);$0["columns"] = .array(columns)}}
        XCTAssertThrowsError(try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:changed))
        guard case .objective(var selected)=selection else {return}
        selected.reference.graphDigest=String(repeating:"a",count:64)
        XCTAssertThrowsError(try WorkflowFailedGradientEvidence.preparedRowCounts(root:root,selection:.objective(selected)))
    }
}
