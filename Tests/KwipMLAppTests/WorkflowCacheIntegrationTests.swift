import XCTest
import Combine
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCacheIntegrationTests: XCTestCase {
    @MainActor private struct Fixture {
        let root: URL
        let csv: URL
        let workspace: WorkspaceStore
        var host: WorkflowWorkspaceSession { workspace.workflowSession! }
    }
    @MainActor private func fixture(experiments:Bool=false,aliasWorkspace:Bool=false) async throws -> Fixture {
        guard let development = ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {
            throw XCTSkip("Set KWIPML_TEST_DEVELOPMENT_ROOT for actual preparation cache execution.")
        }
        let canonical=try XCTUnwrap(realpath(FileManager.default.temporaryDirectory.path,nil))
        defer {free(canonical)}
        let root = URL(fileURLWithPath:String(cString:canonical)).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var storage=root
        if aliasWorkspace {
            storage=root.appendingPathComponent("workspace-alias")
            try FileManager.default.createSymbolicLink(at:storage,withDestinationURL:root)
        }
        let workspace = WorkspaceStore(rootOverride: URL(fileURLWithPath: development), storageOverride: storage, detectCapabilities: false)
        await verifyRuntimeForIntegration(workspace)
        var document = try WorkflowCodec.migrateToVersion2(WorkflowExamples.preparedParallelClassifier(device: "cpu"))
        let index = try XCTUnwrap(document.nodes.firstIndex { if case .runSetup = $0.payload { return true }; return false })
        guard case .runSetup(var config) = document.nodes[index].payload else { throw XCTSkip("Missing sample setup") }
        config["preparationCachePolicy"] = .string("materialize")
        if experiments {
            document=try WorkflowCodec.migrateToVersion4(document)
            config["experiment"] = .object(["schemaVersion":1,"axes":.array([.object(["field":.string("seed"),"values":.array([7,8])])]),"budget":.object(["maximumRuns":2,"maximumTotalEpochs":6])])
        }
        document.nodes[index].payload = .runSetup(config:config)
        _ = try workspace.projectLibrary.saveSource(.workflow(document), id: document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id: document.projectID), workspace.statusMessage)
        let csv = root.appendingPathComponent("private-observations.csv")
        let resource = URL(fileURLWithPath: development).appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv")
        try FileManager.default.copyItem(at: resource, to: csv)
        let host = try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(csv.path, for: WorkflowExamples.sourceAddress, lease: host.owner.lease(), expectedRevision: host.bindings.revision))
        return .init(root: root, csv: csv, workspace: workspace)
    }
    @MainActor private func policy(_ value: String, host: WorkflowWorkspaceSession) throws {
        let node = try XCTUnwrap(host.owner.document.nodes.first { if case .train = $0.payload { return true }; return false })
        guard case .train(let trainConfig) = node.payload else { return XCTFail("Missing Train") }
        var config=try XCTUnwrap(WorkflowTrainRunSetup.configuration(trainConfig,schemaVersion:host.owner.document.schemaVersion))
        config["preparationCachePolicy"] = .string(value)
        XCTAssertNotNil(host.owner.apply(.configure(address: node.address, config:try WorkflowTrainRunSetup.setting(config,in:trainConfig,schemaVersion:host.owner.document.schemaVersion)), lease: host.owner.lease()))
    }
    @MainActor private func run(_ f: Fixture) async throws -> WorkflowExecutionOutcome {
        try await awaitWorkflowIntegrationIdle(f.workspace,host:f.host)
        let count = f.host.outcomes.count
        let done = expectation(description: "Actual cache workflow terminates")
        let observation = f.host.$outcomes.filter { $0.count > count }.prefix(1).sink { _ in done.fulfill() }
        defer { observation.cancel() }
        guard f.host.run(train: WorkflowExamples.trainAddress) else {throw WorkflowArchiveError(f.host.statusMessage)}
        await fulfillment(of: [done], timeout: 60)
        try await awaitWorkflowIntegrationIdle(f.workspace,host:f.host)
        let outcome = try XCTUnwrap(f.host.outcomes.first)
        XCTAssertFalse(f.workspace.isBusy); XCTAssertNil(f.workspace.cancelDataPreparation)
        XCTAssertNil(outcome.recordError)
        return outcome
    }
    private func reference(_ outcome: WorkflowExecutionOutcome) throws -> WorkflowHistoryReference {
        .init(projectID: outcome.request.document.projectID, directoryID: try XCTUnwrap(UUID(uuidString: outcome.directory.lastPathComponent)))
    }
    private func verified(_ outcome: WorkflowExecutionOutcome, root: URL) throws -> VerifiedWorkflowOutcome {
        switch WorkflowHistoryReader.load(try reference(outcome), root: root) {
        case .verified(let result): return result
        case .unavailable(let error), .incomplete(let error): throw NSError(domain: "CacheIntegration", code: 1, userInfo: [NSLocalizedDescriptionKey:error])
        default: throw NSError(domain: "CacheIntegration", code: 2, userInfo: [NSLocalizedDescriptionKey:"New execution did not produce verified History"])
        }
    }
    private func manifest(_ outcome: WorkflowExecutionOutcome) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: outcome.directory.appendingPathComponent("preparation/artifacts/manifest.json"))) as? [String: Any])
    }
    @MainActor private func materializedValues(_ outcome: WorkflowExecutionOutcome, fixture:Fixture) async throws -> [String: Data] {
        let development=try XCTUnwrap(ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"])
        let runtime=try XCTUnwrap(fixture.workspace.runtime)
        // This small synthetic fixture compares complete logical values and lineage.
        // Segment names/layout are storage evidence, independently pinned by inventory.
        let script=#"""
        import json,sys
        from pathlib import Path
        sys.path.insert(0,sys.argv[1])
        from runner.dataflow_storage import completed_inventory,prepared_table_document
        manifest,files,_=completed_inventory(Path(sys.argv[2]),lazy=True)
        result={}
        for entry in manifest['artifacts']:
            if entry['kind'] not in ('table','state'):continue
            value=prepared_table_document(files,entry) if entry['kind']=='table' else json.loads(files[entry['path']])
            if entry['kind']=='table':
                assert len(value['rows'])<=1000,'Fixture population unexpectedly expanded'
                lineage=value['lineage']
                if 'observationIDPrefix' in lineage:
                    lineage=dict(lineage);prefix=lineage.pop('observationIDPrefix');suffixes=lineage.pop('observationIDSuffixes')
                    lineage['observationIDs']=[prefix+suffix for suffix in suffixes]
                    value={**value,'lineage':lineage}
            result[entry['nodeID']+':'+entry['port']]=value
        print(json.dumps(result,sort_keys=True,allow_nan=False,separators=(',',':')))
        """#
        let done=expectation(description:"Decode pinned fixture tables")
        var code:Int32?,lines:[String]=[],bytes=0,overflow=false
        let process=try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:"/usr/bin/env"),
            ["-u","PYTHONPATH","-u","PYTHONHOME","PYTHONNOUSERSITE=1",runtime.executableURL.path,"-c",script,development,outcome.directory.appendingPathComponent("preparation/artifacts/manifest.json").path],fixture.root,
            {line in bytes+=line.utf8.count;if bytes<=2_000_000 {lines.append(line)}else{overflow=true}},
            {value,error in code=value;if !error.isEmpty{lines.append(String(error.prefix(2000)))};done.fulfill()})
        defer{process.cancel()}
        await fulfillment(of:[done],timeout:15)
        guard code==0,!overflow,lines.count==1,let raw=lines.first?.data(using:.utf8),
              let values=try JSONSerialization.jsonObject(with:raw) as? [String:Any] else {
            throw WorkflowArchiveError("Pinned fixture table decoding failed (exit \(String(describing:code)), bounded output \(!overflow)).")
        }
        return try values.mapValues{try JSONSerialization.data(withJSONObject:$0,options:[.sortedKeys])}
    }
    @MainActor func testCapturedCacheAllowanceRefusesOversizeWithoutChangingEarlierHistory() async throws {
        let f = try await fixture(experiments: true); defer { try? FileManager.default.removeItem(at: f.root) }
        if let destination = ProcessInfo.processInfo.environment["KWIPML_CACHE_NATIVE_FIXTURE"] {
            let folder = URL(fileURLWithPath: destination)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            try WorkflowCodec.encode(f.host.owner.document).write(to: folder.appendingPathComponent("Preparation cache allowance.kwipml"))
            try Data(contentsOf: f.csv).write(to: folder.appendingPathComponent("observations.csv"))
        }
        XCTAssertTrue(f.workspace.setExecutionPreferences(try .init(preparationCacheLimitBytes: 100_000)))
        let base = try WorkflowExecutionRequest.capture(from: f.host.owner, train: WorkflowExamples.trainAddress,
            runtime: XCTUnwrap(f.workspace.runtime), preparationCacheLimitBytes: 100_000, moduleCompositionEvidence:f.workspace.moduleCompositionDiscoveryEvidence)
        let different = try WorkflowExecutionRequest.capture(from: f.host.owner, train: WorkflowExamples.trainAddress,
            runtime: XCTUnwrap(f.workspace.runtime), preparationCacheLimitBytes: 1, moduleCompositionEvidence:f.workspace.moduleCompositionDiscoveryEvidence)
        XCTAssertNotEqual(base.requestIdentity, different.requestIdentity)
        XCTAssertEqual(base.plan, different.plan)
        let first = try await run(f)
        XCTAssertEqual(first.status, .completed, first.error ?? "")
        let evidence = try verified(first, root: f.root)
        XCTAssertEqual(evidence.capture.captureFormatVersion, 8)
        XCTAssertEqual(evidence.capture.preparationCacheLimitBytes, 100_000)
        XCTAssertEqual(evidence.preparationCacheReport?.materializedBytesLimit, 100_000)
        XCTAssertGreaterThan(try XCTUnwrap(evidence.preparationCacheReport?.materializedBytes), 1)
        let original = try Data(contentsOf: first.directory.appendingPathComponent("outcome.json"))
        XCTAssertTrue(f.workspace.setExecutionPreferences(try .init(preparationCacheLimitBytes: 1)))
        XCTAssertEqual(base.preparationCacheLimitBytes, 100_000)
        let second = try await run(f)
        XCTAssertEqual(second.status, .failed)
        XCTAssertEqual(second.request.preparationCacheLimitBytes, 1)
        XCTAssertNil(second.modelManifestSHA256)
        let failed = try verified(second, root: f.root)
        XCTAssertEqual(failed.capture.preparationCacheLimitBytes, 1)
        XCTAssertNil(failed.preparationCacheReport)
        XCTAssertEqual(try Data(contentsOf: first.directory.appendingPathComponent("outcome.json")), original)
        XCTAssertEqual(try verified(first, root: f.root).preparationCacheReport?.materializedBytesLimit, 100_000)
        var downgraded = evidence.capture; downgraded.captureFormatVersion = 3
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(downgraded))
        var missing = evidence.capture; missing.preparationCacheLimitBytes = nil
        XCTAssertNil(try WorkflowArchiveCodec.decodeCapture(WorkflowArchiveCodec.encodeCapture(missing)).preparationCacheLimitBytes,"An optional absent allowance remains unknown, never a fabricated default")
        var invalid = evidence.capture; invalid.preparationCacheLimitBytes = 0
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeCapture(invalid))
    }

    @MainActor func testExperimentSpecificationDoesNotChangeSingleRunAndArchiveReplays() async throws {
        let f=try await fixture(experiments:true);defer {try? FileManager.default.removeItem(at:f.root)}
        let source=f.host.owner.document
        let outcome=try await run(f)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? "")
        let history=try verified(outcome,root:f.root)
        XCTAssertEqual(history.capture.schemaVersion,18)
        XCTAssertEqual(history.capture.compilerContractVersion,16)
        XCTAssertEqual(history.plan.settings.seed,42)
        XCTAssertEqual(history.modelContent?.metrics.count,3)
        XCTAssertEqual(f.host.owner.document,source)
        XCTAssertEqual(try f.workspace.projectLibrary.loadSource(id:source.projectID),.workflow(source))
    }
    @MainActor func testTwoDerivedVariantsExecuteFrozenInputsWithoutCanvasMutation() async throws {
        let f=try await fixture(experiments:true);defer {try? FileManager.default.removeItem(at:f.root)}
        let owner=f.host.owner,original=owner.document
        let runtime=try XCTUnwrap(f.workspace.runtime)
        let base=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,preparationCacheLimitBytes:100_000,moduleCompositionEvidence:f.workspace.moduleCompositionDiscoveryEvidence)
        let owned=try WorkflowGroupOwnership.createGroup(workspace:f.root,projectID:owner.projectID)
        let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:owned.directory) {_ in try Data(contentsOf:f.csv)}
        try Data("original changed after group capture".utf8).write(to:f.csv)
        let requests=try (1...2).map {try base.experimentVariant(ordinal:$0,inputs:inputs,owner:owner)}
        XCTAssertEqual(requests.map(\.preparationCacheLimitBytes), [100_000, 100_000])
        let groupOwner=owned.owner
        defer {groupOwner.close()}
        let group=try WorkflowExperimentGroupStore.create(requests:requests,directory:inputs.directory,groupOwner:groupOwner)
        let done=expectation(description:"Experiment scheduler completes both variants")
        var released=0
        let runner=try WorkflowExperimentRunner(requests:requests,owner:owner,group:group,groupOwner:groupOwner,workspaceDirectory:f.root,
            reservation:.init(acquire:{UUID()},release:{_ in released+=1}),onTerminal:{done.fulfill()})
        XCTAssertTrue(runner.start())
        await fulfillment(of:[done],timeout:60)
        XCTAssertFalse(runner.isRunning);XCTAssertNil(runner.error)
        XCTAssertEqual(released,2);XCTAssertEqual(runner.outcomes.count,2)
        for (index,result) in runner.outcomes.enumerated() {
            XCTAssertEqual(result.status,.completed,result.error ?? "")
            XCTAssertNil(result.recordError)
            let history=try verified(result,root:f.root)
            XCTAssertEqual(history.preparationCacheReport?.materializedBytesLimit, 100_000)
            XCTAssertEqual(history.plan.settings.seed,index+7)
            XCTAssertEqual(history.modelContent?.metrics.count,3)
        }
        XCTAssertEqual(owner.document,original)
        XCTAssertTrue(group.record.isTerminal)
        XCTAssertEqual(group.record.variants.map(\.status),[.completed,.completed])
        XCTAssertEqual(try WorkflowExperimentGroupStore.open(directory:inputs.directory).record,group.record)
        let loaded=try WorkflowExperimentHistoryReader.load(directory:inputs.directory,workspaceDirectory:f.root)
        XCTAssertTrue(loaded.isVerifiedComplete)
        XCTAssertEqual(loaded.variants.count,2)
        let ledger=inputs.directory.appendingPathComponent("group.json"),ledgerBytes=try Data(contentsOf:ledger)
        var substituted=group.record
        substituted.variants[1].runID=substituted.variants[0].runID
        substituted.variants[1].directoryID=substituted.variants[0].directoryID
        substituted.variants[1].outcomeSHA256=substituted.variants[0].outcomeSHA256
        try WorkflowArchiveCodec.encode(substituted).write(to:ledger)
        let wrong=try WorkflowExperimentHistoryReader.load(directory:inputs.directory,workspaceDirectory:f.root)
        XCTAssertFalse(wrong.isVerifiedComplete)
        if case .unavailable=wrong.variants[1].evidence {} else {XCTFail("Substituted variant must not verify")}
        try ledgerBytes.write(to:ledger)
        let outcomeFile=runner.outcomes[0].directory.appendingPathComponent("outcome.json")
        let outcomeBytes=try Data(contentsOf:outcomeFile)
        try (outcomeBytes+Data(" ".utf8)).write(to:outcomeFile)
        let tampered=try WorkflowExperimentHistoryReader.load(directory:inputs.directory,workspaceDirectory:f.root)
        XCTAssertFalse(tampered.isVerifiedComplete)
        if case .unavailable=tampered.variants[0].evidence {} else {XCTFail("Changed outcome bytes must fail the group pin")}
        if case .verified=tampered.variants[1].evidence {} else {XCTFail("Unaffected variant remains inspectable")}

        XCTAssertEqual(try f.workspace.projectLibrary.loadSource(id:original.projectID),.workflow(original))
    }

    @MainActor func testWorkspaceLaunchesExperimentGroupAndReleasesReservation() async throws {
        let f=try await fixture(experiments:true,aliasWorkspace:true);defer {try? FileManager.default.removeItem(at:f.root)}
        let original=f.host.owner.document
        var launchedArguments:[[String]]=[],readiness:[[String:Any]]=[]
        f.host.executionLaunch={executable,arguments,directory,line,exit in
            launchedArguments.append(arguments)
            return try WorkflowExecutionSession.launchWorker(executable,arguments,directory,{event in
                if let bytes=event.data(using:.utf8),let decoded=try? JSONSerialization.jsonObject(with:bytes) as? [String:Any],decoded["event"] as? String == "launchReady" {readiness.append(decoded)}
                line(event)
            },exit)
        }
        let capture=f.host.captureExperimentInputs
        var captures=0
        f.host.captureExperimentInputs={paths,parent in captures+=1;return try await capture(paths,parent)}
        let done=expectation(description:"Workspace group records both outcomes")
        let observation=f.host.$outcomes.filter {$0.count == 2}.prefix(1).sink {_ in done.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress),f.host.statusMessage)
        XCTAssertTrue(f.workspace.isBusy)
        XCTAssertFalse(f.host.runExperiments(train:WorkflowExamples.trainAddress))
        await fulfillment(of:[done],timeout:60)
        let runner=try XCTUnwrap(f.host.experimentRunner)
        XCTAssertEqual(captures,1)
        XCTAssertFalse(runner.isRunning);XCTAssertFalse(f.workspace.isBusy)
        XCTAssertNil(f.workspace.cancelDataPreparation)
        XCTAssertTrue(runner.group.record.isTerminal)
        XCTAssertEqual(runner.group.record.variants.map(\.status),[.completed,.completed])
        XCTAssertEqual(launchedArguments.count,4)
        XCTAssertTrue(launchedArguments.allSatisfy {$0.contains("--kwip-supervise") && !$0.contains("--graph") && !$0.contains("--execute-dataflow")})
        XCTAssertEqual(readiness.count,4)
        XCTAssertEqual(Set(readiness.compactMap {$0["nonce"] as? String}).count,4)
        for ready in readiness {
            let nonce=try XCTUnwrap(ready["nonce"] as? String)
            let intent=try WorkflowLaunchIntent.decode(Data(contentsOf:runner.group.directory.appendingPathComponent("launch-\(nonce).json")))
            XCTAssertEqual(intent.phase.rawValue,ready["phase"] as? String)
            let claim=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:runner.group.directory.appendingPathComponent("claim-\(nonce).json"))) as? [String:Any])
            XCTAssertEqual(claim["state"] as? String,"claimed")
        }
        XCTAssertEqual(f.host.owner.document,original)
        let group=try WorkflowExperimentHistoryReader.load(directory:runner.group.directory,workspaceDirectory:f.root)
        XCTAssertTrue(group.isVerifiedComplete)
        let listing=try WorkflowExperimentHistoryReader.list(root:f.root,projectID:original.projectID)
        XCTAssertEqual(listing.entries.map(\.groupID),[runner.group.record.groupID])
        XCTAssertTrue(listing.issues.isEmpty);XCTAssertFalse(listing.truncated)
        let directory=runner.group.directory,root=f.root
        let background=try await Task.detached {try await WorkflowHistoryVerificationWork.run {try WorkflowExperimentHistoryReader.load(directory:directory,workspaceDirectory:root)}}.value
        XCTAssertTrue(background.isVerifiedComplete)
        let parent=runner.group.directory.deletingLastPathComponent()
        let malformed=parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:malformed,withIntermediateDirectories:false)
        try Data("{}".utf8).write(to:malformed.appendingPathComponent("group.json"))
        let link=parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createSymbolicLink(at:link,withDestinationURL:runner.group.directory)
        let withIssues=try WorkflowExperimentHistoryReader.list(root:f.root,projectID:original.projectID)
        XCTAssertEqual(withIssues.entries.count,1);XCTAssertEqual(withIssues.issues.count,2)
        XCTAssertTrue(try WorkflowExperimentHistoryReader.list(root:f.root,projectID:original.projectID,limit:1).truncated)
        XCTAssertThrowsError(try WorkflowExperimentHistoryReader.list(root:f.root,projectID:"../other"))

    }

    @MainActor func testWorkspaceExperimentCancellationAndChangedCaptureAuthority() async throws {
        for changesDuringCapture in [false,true] {
            let f=try await fixture(experiments:true);defer {try? FileManager.default.removeItem(at:f.root)}
            var launches=0,cancels=0
            f.host.executionLaunch={_,_,_,_,exit in
                launches+=1;return .init(cancel:{cancels+=1;exit(15,"cancelled")})
            }
            if changesDuringCapture {
                let original=f.host.captureExperimentInputs
                f.host.captureExperimentInputs={paths,parent in
                    let snapshot=try await original(paths,parent)
                    XCTAssertTrue(f.host.owner.rename("Changed during capture",lease:f.host.owner.lease()))
                    return snapshot
                }
                let stopped=expectation(description:"Staging rejects changed authority")
                let stageObservation=f.host.$isStagingExperiment.dropFirst().filter {!$0}.prefix(1).sink {_ in stopped.fulfill()}
                defer {stageObservation.cancel()}
                XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress))
                await fulfillment(of:[stopped],timeout:5)
                XCTAssertEqual(launches,0);XCTAssertFalse(f.workspace.isBusy)
                XCTAssertNil(f.workspace.cancelDataPreparation)
            } else {
                let done=expectation(description:"Workspace cancellation terminates group")
                let observation=f.host.$outcomes.filter {!$0.isEmpty}.prefix(1).sink {_ in done.fulfill()}
                defer {observation.cancel()}
                XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress))
                let started=expectation(description:"First worker starts")
                let originalLaunch=f.host.executionLaunch
                f.host.executionLaunch={a,b,c,d,e in
                    let result=try originalLaunch(a,b,c,d,e);started.fulfill();return result
                }
                await fulfillment(of:[started],timeout:5)
                XCTAssertEqual(f.host.statusMessage,"Variant 1 of 2 · Preparing captured sources…")
                f.workspace.cancelRun()
                await fulfillment(of:[done],timeout:5)
                XCTAssertEqual(launches,1);XCTAssertEqual(cancels,1)
                XCTAssertFalse(f.workspace.isBusy);XCTAssertNil(f.workspace.cancelDataPreparation)
                XCTAssertEqual(f.host.experimentRunner?.group.record.variants.map(\.status),[.cancelled,.cancelled])
            }
        }
    }

    @MainActor func testExperimentCancelDuringStagingStartsNoWorker() async throws {
        let f=try await fixture(experiments:true);defer {try? FileManager.default.removeItem(at:f.root)}
        var launches=0
        f.host.executionLaunch={_,_,_,_,_ in launches+=1;return .init(cancel:{})}
        let stopped=expectation(description:"Staging releases workspace")
        let observation=f.host.$isStagingExperiment.dropFirst().filter {!$0}.prefix(1).sink {_ in stopped.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress))
        XCTAssertTrue(f.host.isStagingExperiment);XCTAssertTrue(f.workspace.isBusy)
        f.workspace.cancelRun()
        await fulfillment(of:[stopped],timeout:5)
        XCTAssertEqual(launches,0);XCTAssertNil(f.host.experimentRunner)
        XCTAssertFalse(f.workspace.isBusy);XCTAssertNil(f.workspace.cancelDataPreparation)
        let parent=f.root.appendingPathComponent("workflow-experiments/\(f.host.projectID)")
        // Cancellation can precede async group admission. Never manufacture a
        // group receipt just to represent a request that was not admitted.
        let groups = FileManager.default.fileExists(atPath: parent.path)
            ? try FileManager.default.contentsOfDirectory(at: parent, includingPropertiesForKeys: nil) : []
        XCTAssertLessThanOrEqual(groups.count, 1)
        for directory in groups {
            XCTAssertNoThrow(try WorkflowGroupOwnership.readMetadata(directory: directory, groupID: directory.lastPathComponent))
            let recovery = try WorkflowGroupOwnership.acquireRecovery(directory: directory, groupID: directory.lastPathComponent)
            recovery.close()
        }
        XCTAssertTrue(f.host.outcomes.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath: f.root.appendingPathComponent("workflow-runs/\(f.host.projectID)").path))
    }

    @MainActor func testExperimentCancellationRetainsOwnerUntilWorkerExit() async throws {
        let f=try await fixture(experiments:true);defer {try? FileManager.default.removeItem(at:f.root)}
        let started=expectation(description:"Worker launch accepted")
        var acknowledge:((Int32,String)->Void)?,cancellations=0
        f.host.executionLaunch={_,_,_,_,exit in
            acknowledge=exit;started.fulfill()
            return .init(cancel:{cancellations+=1})
        }
        let stopped=expectation(description:"Terminal outcome after worker exit")
        let observation=f.host.$outcomes.filter {!$0.isEmpty}.prefix(1).sink {_ in stopped.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress),f.host.statusMessage)
        await fulfillment(of:[started],timeout:5)
        let group=try XCTUnwrap(f.host.experimentRunner?.group)
        f.workspace.cancelRun()
        XCTAssertEqual(cancellations,1)
        XCTAssertTrue(f.workspace.isBusy)
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:group.directory,groupID:group.record.groupID)) {
            XCTAssertEqual($0 as? WorkflowFileLeaseError,.busy)
        }
        try XCTUnwrap(acknowledge)(15,"cancelled")
        await fulfillment(of:[stopped],timeout:5)
        XCTAssertFalse(f.workspace.isBusy)
        XCTAssertEqual(group.record.variants.map(\.status),[.cancelled,.cancelled])
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:group.directory,groupID:group.record.groupID)
        recovery.close()
    }

    @MainActor func testGroupLedgerFailurePreservesCompletedOutcomeAndStopsNextVariant() async throws {
        let f=try await fixture(experiments:true);defer {try? FileManager.default.removeItem(at:f.root)}
        var launches=0,changedLedger=false
        f.host.executionLaunch={executable,arguments,directory,line,exit in
            launches+=1
            return try WorkflowExecutionSession.launchWorker(executable,arguments,directory,line,{code,stderr in
                if code == 0,f.host.experimentRunner?.activeSession?.phase == .training,
                   let group=f.host.experimentRunner?.group {
                    do {
                        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:group.directory,groupID:group.record.groupID)) {
                            XCTAssertEqual($0 as? WorkflowFileLeaseError,.busy)
                        }
                        let ledger=group.directory.appendingPathComponent("group.json")
                        try (Data(contentsOf:ledger)+Data(" ".utf8)).write(to:ledger)
                        changedLedger=true
                    } catch {XCTFail(error.localizedDescription)}
                }
                exit(code,stderr)
            })
        }
        let stopped=expectation(description:"Ledger failure stops group after completed worker")
        let observation=f.host.$outcomes.filter {!$0.isEmpty}.prefix(1).sink {_ in stopped.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress),f.host.statusMessage)
        await fulfillment(of:[stopped],timeout:60)
        let runner=try XCTUnwrap(f.host.experimentRunner),outcome=try XCTUnwrap(f.host.outcomes.first)
        XCTAssertTrue(changedLedger);XCTAssertEqual(launches,2)
        XCTAssertFalse(runner.isRunning);XCTAssertNotNil(runner.error)
        XCTAssertEqual(outcome.status,.completed);XCTAssertNil(outcome.recordError)
        XCTAssertEqual(runner.group.record.variants.map(\.status),[.running,.pending])
        XCTAssertFalse(f.workspace.isBusy)
        XCTAssertEqual(try verified(outcome,root:f.root).outcome.status,.completed)
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:runner.group.directory,groupID:runner.group.record.groupID)
        recovery.close()
    }

    @MainActor func testExperimentCancelledAfterCopyRetainsOwnershipUntilAcknowledged() async throws {
        let f=try await fixture(experiments:true);defer {try? FileManager.default.removeItem(at:f.root)}
        let capture=f.host.captureExperimentInputs
        let captured=expectation(description:"Copies ready while host remains responsive")
        var release:CheckedContinuation<Void,Never>?
        var capturedDirectory:URL?
        f.host.captureExperimentInputs={paths,parent in
            let inputs=try await capture(paths,parent)
            capturedDirectory=inputs.directory
            await withCheckedContinuation {continuation in release=continuation;captured.fulfill()}
            return inputs
        }
        var launches=0
        f.host.executionLaunch={_,_,_,_,_ in launches+=1;return .init(cancel:{})}
        let stopped=expectation(description:"Unpublished capture cleaned")
        let observation=f.host.$isStagingExperiment.dropFirst().filter {!$0}.prefix(1).sink {_ in stopped.fulfill()}
        defer {observation.cancel()}
        XCTAssertTrue(f.host.runExperiments(train:WorkflowExamples.trainAddress))
        await fulfillment(of:[captured],timeout:5)
        XCTAssertTrue(f.workspace.isBusy)
        let directory=try XCTUnwrap(capturedDirectory)
        XCTAssertNoThrow(try WorkflowGroupOwnership.readMetadata(directory:directory,groupID:directory.lastPathComponent))
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:directory.lastPathComponent)) {
            XCTAssertEqual($0 as? WorkflowFileLeaseError,.busy)
        }
        f.workspace.cancelRun()
        XCTAssertTrue(f.workspace.isBusy,"Keep the reservation until staging acknowledges cancellation")
        XCTAssertThrowsError(try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:directory.lastPathComponent)) {
            XCTAssertEqual($0 as? WorkflowFileLeaseError,.busy)
        }
        release?.resume()
        await fulfillment(of:[stopped],timeout:5)
        XCTAssertEqual(launches,0);XCTAssertNil(f.host.experimentRunner)
        XCTAssertFalse(f.workspace.isBusy);XCTAssertNil(f.workspace.cancelDataPreparation)
        XCTAssertTrue(FileManager.default.fileExists(atPath:directory.appendingPathComponent("inputs.json").path))
        let recovery=try WorkflowGroupOwnership.acquireRecovery(directory:directory,groupID:directory.lastPathComponent)
        recovery.close()
    }

    @MainActor func testSampleComponentThroughCanvasRunAndVerifiedHistory() async throws {
        let f=try await fixture();defer {try? FileManager.default.removeItem(at:f.root)}
        let canvas=f.host.canvas,owner=f.host.owner
        let entry=try XCTUnwrap(canvas.creationEntries.first {if case .dataOperator("kwip.data.sampleRows",_)=$0.payload {return true};return false})
        XCTAssertTrue(canvas.canAddType(entry.id))
        let before=Set(owner.document.nodes.map(\.address))
        canvas.addNode(type:entry.id,x:300,y:240)
        let sample=try XCTUnwrap(owner.document.nodes.first {!before.contains($0.address)})
        let config:[String:JSONValue]=["count":40,"seed":7]
        XCTAssertNotNil(owner.apply(.configure(address:sample.address,config:config),lease:owner.lease()))
        let edge=try XCTUnwrap(owner.document.edges.first {$0.from.address == WorkflowExamples.sourceAddress})
        XCTAssertNotNil(owner.apply(.reconnect(edgeID:edge.id,from:edge.from,to:.init(address:sample.address,port:"table")),lease:owner.lease()))
        XCTAssertNotNil(owner.apply(.connect(from:.init(address:sample.address,port:"table"),to:edge.to),lease:owner.lease()))
        let authored=owner.document
        let first=try await run(f)
        XCTAssertEqual(first.status,.completed,first.error ?? "")
        let capture=try XCTUnwrap(f.host.snapshot(for:sample.address))
        guard case .object(let table)=capture.ports["table"] else {return XCTFail("Missing measured sample output")}
        guard case .object(let preview)=table["preview"] else {return XCTFail("Missing measured table preview")}
        XCTAssertEqual(preview["rowCount"],40)
        let archive=try verified(first,root:f.root)
        XCTAssertEqual(archive.modelContent?.metrics.count,3)
        XCTAssertEqual(archive.plan.dataflow.nodes.first {$0.id == sample.address.localID}?.config,config)
        XCTAssertEqual(try f.workspace.projectLibrary.loadSource(id:owner.projectID),.workflow(authored))
        try policy("reuse",host:f.host)
        XCTAssertTrue(f.host.selectPreparationCache(reference:try reference(first),for:WorkflowExamples.trainAddress,lease:owner.lease()),f.host.statusMessage)
        let second=try await run(f)
        XCTAssertEqual(second.status,.completed,second.error ?? "")
        let cached=try verified(second,root:f.root)
        XCTAssertTrue(cached.preparationCacheReport?.reusedNodeIDs.contains(sample.address.localID) == true)
        let history = WorkflowHistoryStore(workspace:f.workspace,host:f.host)
        func select(_ outcome:WorkflowExecutionOutcome) async throws {
            let done = expectation(description:"Comparison selection verifies")
            history.select(try reference(outcome))
            let observation = history.$isLoading.filter {!$0}.prefix(1).sink {_ in done.fulfill()}
            await fulfillment(of:[done],timeout:5);observation.cancel()
        }
        XCTAssertFalse(history.pinComparisonBaseline())
        try await select(first)
        XCTAssertTrue(history.pinComparisonBaseline(),history.statusMessage)
        try await select(second)
        let baseline = try XCTUnwrap(history.comparisonBaseline)
        XCTAssertEqual(baseline.outcome.runID,archive.outcome.runID)
        guard case .verified(let candidate)=history.selection else {return XCTFail("Candidate was not verified")}
        let fields = try WorkflowComparisonFields.differences(WorkflowComparisonFields.configuration(baseline),WorkflowComparisonFields.configuration(candidate))
        XCTAssertTrue(fields.contains {$0.path == "/cachePolicy" && $0.baseline == "\"materialize\"" && $0.candidate == "\"reuse\""})
        XCTAssertFalse(fields.contains {$0.path.hasPrefix("/dataComponents") || $0.path == "/modelIdentity" || $0.path.hasPrefix("/sourceContent")})
        XCTAssertEqual(baseline.modelContent?.metrics.count,3)
        let partition=try XCTUnwrap(baseline.modelContent?.partitionEvidence)
        // Independently verify both sampling and three-way allocation from exact
        // captured tables. Training+Validation intentionally excludes protected Test.
        let measured=try await materializedValues(first,fixture:f)
        func capturedTable(_ node:String,_ port:String="table")throws->[String:Any] {
            try XCTUnwrap(JSONSerialization.jsonObject(with:XCTUnwrap(measured[node+":"+port])) as? [String:Any])
        }
        func rows(_ table:[String:Any])throws->[[Any]] {try XCTUnwrap(table["rows"] as? [[Any]])}
        func ids(_ table:[String:Any])throws->[String] {try XCTUnwrap((table["lineage"] as? [String:Any])?["observationIDs"] as? [String])}
        let source=try capturedTable(WorkflowExamples.sourceAddress.localID),sampled=try capturedTable(sample.address.localID),typed=try capturedTable("cast")
        let sourceRows=try rows(source),sourceIDs=try ids(source),sampledRows=try rows(sampled),sampledIDs=try ids(sampled),typedRows=try rows(typed),typedIDs=try ids(typed)
        XCTAssertEqual(sourceRows.count,64);XCTAssertEqual(sampledRows.count,40);XCTAssertEqual(typedIDs,sampledIDs)
        func sampleRank(_ id:String)->String {WorkflowArchiveCodec.hash(Data(("kwip.sample.rows.v1\n7\n"+String(id.utf8.count)+":"+id).utf8))}
        let chosen=Set(sourceIDs.sorted {left,right in let a=sampleRank(left),b=sampleRank(right);return a==b ? left<right:a<b}.prefix(40))
        let positions=sourceIDs.indices.filter {chosen.contains(sourceIDs[$0])}
        XCTAssertEqual(sampledIDs,positions.map {sourceIDs[$0]})
        XCTAssertEqual(try JSONSerialization.data(withJSONObject:sampledRows,options:[.sortedKeys]),try JSONSerialization.data(withJSONObject:positions.map {sourceRows[$0]},options:[.sortedKeys]))
        let split=try XCTUnwrap(archive.plan.dataflow.nodes.first {$0.id=="split"})
        XCTAssertEqual(split.operatorID,"kwip.data.partition");XCTAssertEqual(split.config["strategy"],.string("random"))
        XCTAssertEqual(split.config["seed"],42);XCTAssertEqual(split.config["stratifyColumn"],.string("label"))
        XCTAssertEqual(split.config["validationPercent"],25);XCTAssertEqual(split.config["testPercent"],25)
        let columns=try XCTUnwrap(typed["columns"] as? [[String:Any]]),label=try XCTUnwrap(columns.firstIndex {$0["name"] as? String == "label"})
        var groups:[Int:[Int]]=[:]
        for index in typedRows.indices {groups[try XCTUnwrap(typedRows[index][label] as? NSNumber).intValue,default:[]].append(index)}
        let roles=["training","validation","test"];var expected=Dictionary(uniqueKeysWithValues:roles.map {($0,Set<String>())})
        for (value,indices) in groups {
            let percentages=[50,25,25],n=indices.count
            var counts=percentages.map {n*$0/100}
            let priority=(0..<3).sorted {left,right in let a=n*percentages[left]%100,b=n*percentages[right]%100;return a==b ? left<right:a>b}
            for index in priority.prefix(n-counts.reduce(0,+)) {counts[index]+=1}
            func rank(_ index:Int)throws->String {
                let key:JSONValue = .array([.string("kwip.partition.random.v1"),42,.array([.array([.string("label"),.string("int64"),.integer(Int64(value))])]),.string(typedIDs[index])])
                return WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(key))
            }
            let ranks=try Dictionary(uniqueKeysWithValues:indices.map {($0,try rank($0))})
            let ordered=indices.sorted {left,right in let a=ranks[left]!,b=ranks[right]!;return a==b ? typedIDs[left]<typedIDs[right]:a<b}
            var start=0
            for (role,count) in zip(roles,counts) {expected[role]!.formUnion(ordered[start..<start+count].map {typedIDs[$0]});start+=count}
        }
        var actual:[String:Set<String>]=[:]
        for role in roles {
            let retained=try ids(capturedTable("split",role)),prepared=try ids(capturedTable("fit",role))
            actual[role]=Set(retained)
            XCTAssertEqual(retained,typedIDs.filter {expected[role]!.contains($0)})
            XCTAssertEqual(prepared,retained)
        }
        XCTAssertEqual(actual,expected);XCTAssertEqual(actual["training"]!.union(actual["validation"]!).union(actual["test"]!),chosen)
        XCTAssertTrue(actual["training"]!.isDisjoint(with:actual["validation"]!));XCTAssertTrue(actual["training"]!.isDisjoint(with:actual["test"]!));XCTAssertTrue(actual["validation"]!.isDisjoint(with:actual["test"]!))
        XCTAssertEqual(try rows(capturedTable("split","exclusions")).count,0)
        XCTAssertEqual(partition.trainingCount,expected["training"]!.count);XCTAssertEqual(partition.validationCount,expected["validation"]!.count)
        XCTAssertEqual(partition.trainingMembershipSHA256,WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(expected["training"]!.sorted())))
        XCTAssertEqual(partition.validationMembershipSHA256,WorkflowArchiveCodec.hash(try WorkflowArchiveCodec.encode(expected["validation"]!.sorted())))
        XCTAssertEqual(WorkflowPartitionEvidence.comparison(partition,candidate.modelContent?.partitionEvidence),.sameRowsAndValues)
        XCTAssertNotNil(owner.apply(.configure(address:sample.address,config:["count":40,"seed":8]),lease:owner.lease()))
        let third=try await run(f)
        XCTAssertEqual(third.status,.completed,third.error ?? "")
        try await select(third)
        guard case .verified(let changed)=history.selection else {return XCTFail("Changed-seed candidate was not verified")}
        let seedFields=try WorkflowComparisonFields.differences(WorkflowComparisonFields.configuration(baseline),WorkflowComparisonFields.configuration(changed))
        XCTAssertTrue(seedFields.contains {$0.path == "/dataComponents/\(sample.address.localID)/config/seed" && $0.baseline == "7" && $0.candidate == "8"})
        XCTAssertFalse(changed.preparationCacheReport?.reusedNodeIDs.contains(sample.address.localID) == true)
        XCTAssertEqual(WorkflowPartitionEvidence.comparison(partition,changed.modelContent?.partitionEvidence),.differentRowsOrSource)
        XCTAssertEqual(history.comparisonBaseline?.outcome.runID,baseline.outcome.runID)
        history.clearComparisonBaseline();XCTAssertNil(history.comparisonBaseline)
        XCTAssertTrue(history.pinComparisonBaseline())
        history.invalidate();XCTAssertNil(history.comparisonBaseline)
        XCTAssertFalse(history.pinComparisonBaseline())

    }
    @MainActor func testActualReuseMatchesValuesAndMetricsAndHistorySurvivesPriorDeletion() async throws {
        let f = try await fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let first = try await run(f)
        XCTAssertEqual(first.status, .completed, first.error ?? "")
        let firstHistory = try verified(first, root: f.root)
        let firstReport = try XCTUnwrap(firstHistory.preparationCacheReport)
        XCTAssertEqual(firstReport.policy, "materialize"); XCTAssertEqual(firstReport.executedNodeIDs.count, 6); XCTAssertTrue(firstReport.reusedNodeIDs.isEmpty)
        try policy("reuse", host: f.host)
        XCTAssertTrue(f.host.selectPreparationCache(reference: try reference(first), for: WorkflowExamples.trainAddress, lease: f.host.owner.lease()), f.host.statusMessage)
        let projectID = f.host.owner.projectID
        // A persisted private selection must survive a new host, not only an in-memory second run.
        f.workspace.newProject()
        XCTAssertTrue(f.workspace.beginWorkflowProject(id: projectID), f.workspace.statusMessage)
        let second = try await run(f)
        XCTAssertEqual(second.status, .completed, second.error ?? "")
        let history = try verified(second, root: f.root), report = try XCTUnwrap(history.preparationCacheReport)
        XCTAssertEqual(report.policy, "reuse")
        XCTAssertEqual(report.executedNodeIDs, [WorkflowExamples.sourceAddress.localID]); XCTAssertEqual(report.reusedNodeIDs.count, 5)
        XCTAssertEqual(Set(first.request.plan.dataflow.nodes.map(\.id)),[WorkflowExamples.sourceAddress.localID,"cast","split","fit","training","validation"],"Training-side inspection Views are retained in the selected Train preparation.")
        XCTAssertEqual(Set(firstReport.executedNodeIDs),[WorkflowExamples.sourceAddress.localID,"cast","split","fit","training","validation"])
        XCTAssertEqual(Set(report.reusedNodeIDs),["cast","split","fit","training","validation"])
        XCTAssertEqual(second.request.plan.model.semanticSHA256, first.request.plan.model.semanticSHA256)
        let originalValues=try await materializedValues(first,fixture:f),reusedValues=try await materializedValues(second,fixture:f)
        XCTAssertEqual(reusedValues,originalValues)
        let expected = try XCTUnwrap(firstHistory.modelContent).metrics, actual = try XCTUnwrap(history.modelContent).metrics
        XCTAssertEqual(actual.count, 3); XCTAssertEqual(expected.count, actual.count)
        for (a,b) in zip(actual,expected) {
            XCTAssertTrue(a.trainLoss.isFinite && a.validationLoss.isFinite && a.accuracy.map(\.isFinite) == true)
            XCTAssertEqual(a.trainLoss,b.trainLoss); XCTAssertEqual(a.validationLoss,b.validationLoss); XCTAssertEqual(a.accuracy,b.accuracy)
        }
        let cacheJSON = try XCTUnwrap(try manifest(second)["cache"] as? [String: Any])
        XCTAssertEqual(cacheJSON["sourceManifestSHA256"] as? String, first.preparationManifestSHA256)
        XCTAssertFalse(String(decoding: try WorkflowCodec.encode(f.host.owner.document), as: UTF8.self).contains(first.directory.path))
        try FileManager.default.removeItem(at: first.directory)
        let independent = try verified(second, root: f.root)
        XCTAssertEqual(independent.outcome.status, .completed)
        XCTAssertEqual(independent.preparationCacheReport?.reusedNodeIDs.count, 5)
        XCTAssertTrue(f.host.clearPreparationCache(for: WorkflowExamples.trainAddress, lease: f.host.owner.lease()))
        XCTAssertFalse(f.host.run(train: WorkflowExamples.trainAddress), "Reuse with no private selection must never silently fall back")
        XCTAssertFalse(f.workspace.isBusy)
    }
    @MainActor func testSourceMutationRevalidatesAndRecomputesAllDescendants() async throws {
        let f = try await fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let first = try await run(f); XCTAssertEqual(first.status, .completed, first.error ?? "")
        try policy("reuse", host: f.host)
        XCTAssertTrue(f.host.selectPreparationCache(reference: try reference(first), for: WorkflowExamples.trainAddress, lease: f.host.owner.lease()))
        var rows = try String(contentsOf: f.csv, encoding: .utf8).split(whereSeparator: \.isNewline).map(String.init)
        var cells = rows[1].split(separator: ",").map(String.init)
        cells[0] = String(try XCTUnwrap(Double(cells[0])) + 0.12345)
        rows[1] = cells.joined(separator: ",")
        try (rows.joined(separator: "\n") + "\n").write(to: f.csv, atomically: true, encoding: .utf8)
        let second = try await run(f); XCTAssertEqual(second.status, .completed, second.error ?? "")
        let a = try verified(first, root: f.root), b = try verified(second, root: f.root)
        let report = try XCTUnwrap(b.preparationCacheReport)
        XCTAssertEqual(report.policy, "reuse"); XCTAssertEqual(report.executedNodeIDs.count, 6); XCTAssertTrue(report.reusedNodeIDs.isEmpty)
        XCTAssertEqual(Set(report.executedNodeIDs),[WorkflowExamples.sourceAddress.localID,"cast","split","fit","training","validation"])
        XCTAssertNotEqual(a.capture.sources.first?.sha256, b.capture.sources.first?.sha256)
        let originalValues=try await materializedValues(first,fixture:f),changedValues=try await materializedValues(second,fixture:f)
        XCTAssertNotEqual(originalValues,changedValues)
        XCTAssertEqual(first.request.plan.model.semanticSHA256, second.request.plan.model.semanticSHA256)
    }
    @MainActor func testRequestedArtifactCorruptionCannotReachTrainingOrRewritePriorReceipt() async throws {
        let f = try await fixture(); defer { try? FileManager.default.removeItem(at: f.root) }
        let first = try await run(f); XCTAssertEqual(first.status, .completed, first.error ?? "")
        try policy("reuse", host: f.host)
        XCTAssertTrue(f.host.selectPreparationCache(reference: try reference(first), for: WorkflowExamples.trainAddress, lease: f.host.owner.lease()))
        let retained = try ["outcome.json", "capture.json", "preparation/artifacts/manifest.json"].map { ($0,try Data(contentsOf:first.directory.appendingPathComponent($0))) }
        let entries = try XCTUnwrap(try manifest(first)["artifacts"] as? [[String: Any]])
        let cast = try XCTUnwrap(entries.first { $0["nodeID"] as? String == "cast" && $0["kind"] as? String == "table" })
        let path = first.directory.appendingPathComponent("preparation/artifacts/" + (try XCTUnwrap(cast["path"] as? String)))
        try Data("corrupt requested cache artifact".utf8).write(to: path)
        let before = f.host.outcomes.count
        let done = expectation(description: "Corrupt selected cache is rejected")
        let observation = f.host.$outcomes.filter { $0.count > before }.prefix(1).sink { _ in done.fulfill() }
        defer { observation.cancel() }
        let accepted = f.host.run(train: WorkflowExamples.trainAddress)
        if !accepted { done.fulfill() }
        await fulfillment(of: [done], timeout: 60)
        if accepted {
            let failure = try XCTUnwrap(f.host.outcomes.first)
            XCTAssertEqual(failure.status, .failed); XCTAssertNil(failure.modelManifestSHA256)
            XCTAssertTrue(f.host.execution?.metrics.isEmpty == true)
            XCTAssertFalse(FileManager.default.fileExists(atPath: failure.directory.appendingPathComponent("training/artifacts/manifest.json").path))
        } else {
            XCTAssertEqual(f.host.outcomes.count, before)
            XCTAssertFalse(f.host.statusMessage.isEmpty)
        }
        XCTAssertFalse(f.workspace.isBusy)
        for (name,bytes) in retained { XCTAssertEqual(try Data(contentsOf:first.directory.appendingPathComponent(name)),bytes) }
    }
}
