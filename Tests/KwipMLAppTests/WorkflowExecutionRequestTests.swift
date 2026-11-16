import XCTest
import Darwin
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExecutionRequestTests: XCTestCase {
    private func address(_ domain:WorkflowDomain,_ id:String)->WorkflowNodeAddress {.init(domain:domain,localID:id)}
    private func edge(_ a:WorkflowDomain,_ id:String,_ port:String,_ b:WorkflowDomain,_ target:String,_ input:String,_ kind:WorkflowEdgeKind)->WorkflowEdge {
        .init(id:"\(a)-\(id)-\(port)-\(b)-\(target)-\(input)",from:.init(address:address(a,id),port:port),to:.init(address:address(b,target),port:input),kind:kind)
    }
    private func fixture(block:Bool=false)->WorkflowDocument {CurrentExecutionWorkflowTestFixture.make(block:block)}

    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func dataRegistry() throws -> DataFlowOperatorRegistry {
        try .load(from: repo.appendingPathComponent("runner/dataflow-operators.json"))
    }
    @MainActor func setup(mode:WorkflowTrainingMode?=nil) throws -> (URL, WorkflowAuthoringStore, RuntimeLocation) {
        guard let canonical=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw NSError(domain:NSPOSIXErrorDomain,code:Int(errno))}
        defer {free(canonical)}
        let root = URL(fileURLWithPath:String(cString:canonical),isDirectory:true).appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let library = ProjectLibrary(directory: root.appendingPathComponent("projects"))
        var source = fixture(block: true)
        if let mode {
            source=try WorkflowCodec.migrateToVersion3(source)
            let index=try XCTUnwrap(source.nodes.firstIndex {$0.address==address(.control,"train")})
            source.nodes[index].payload = .train(config:["trainingMode":.string(mode.rawValue)])
        }
        _ = try library.saveSource(.workflow(source), id: source.projectID)
        let runtime = try makeRuntime(root.appendingPathComponent("runtime"))
        return (root, try WorkflowAuthoringStore(library: library, projectID: source.projectID,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry()), runtime)
    }
    @MainActor func testCapturedExecutionBudgetChangesIdentityWithoutChangingGraph() throws {
        let (root, owner, runtime) = try setup()
        defer { try? FileManager.default.removeItem(at: root) }
        let firstPolicy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60)
        let secondPolicy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 120)
        let legacy = try WorkflowExecutionRequest.capture(from: owner, train: address(.control,"train"), runtime: runtime)
        let first = try WorkflowExecutionRequest.capture(from: owner, train: address(.control,"train"), runtime: runtime, executionBudget: firstPolicy)
        let second = try WorkflowExecutionRequest.capture(from: owner, train: address(.control,"train"), runtime: runtime, executionBudget: secondPolicy)
        let repeated = try WorkflowExecutionRequest.capture(from: owner, train: address(.control,"train"), runtime: runtime, executionBudget: firstPolicy)
        XCTAssertNil(legacy.executionBudget)
        XCTAssertEqual(first.executionBudget, firstPolicy)
        XCTAssertEqual(second.executionBudget, secondPolicy)
        XCTAssertNotEqual(first.requestIdentity, second.requestIdentity)
        XCTAssertNotEqual(first.requestIdentity, legacy.requestIdentity)
        XCTAssertEqual(first.requestIdentity, repeated.requestIdentity)
        XCTAssertEqual(first.document, legacy.document)
        XCTAssertEqual(first.plan, legacy.plan)
        XCTAssertTrue(first.isCurrent(in: owner))
        let source = root.appendingPathComponent("budget-source.csv")
        try Data("a,b,label\n1,2,0\n2,3,1\n".utf8).write(to: source)
        for request in [legacy, first] {
            let directory = root.appendingPathComponent(UUID().uuidString)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
            let written = try WorkflowArchiveCaptureWriter.write(request: request, runID: UUID(),
                sources: [address(.data,"input"): try WorkflowCapturedSource.capture(source)], directory: directory,
                modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry)
            let bytes = try Data(contentsOf: directory.appendingPathComponent(written.reference.path))
            let capture = try WorkflowArchiveCodec.decodeCapture(bytes)
            XCTAssertEqual(capture.executionBudget, request.executionBudget)
            XCTAssertEqual(capture.captureFormatVersion, request.executionBudget == nil ? nil : 2)
            XCTAssertEqual(capture.requestIdentity, request.requestIdentity)
            XCTAssertEqual(capture.schemaVersion, request.document.schemaVersion)
        }
    }

    @MainActor func testContinuationRequestPinsParentAndRejectsMissingForeignOrFreshParent()throws {
        for mode in WorkflowTrainingMode.allCases {
            let(root,owner,runtime)=try setup(mode:mode);defer{try? FileManager.default.removeItem(at:root)}
            let parent=WorkflowCheckpointReference(projectID:owner.projectID,directoryID:UUID().uuidString,
                runID:UUID().uuidString,requestIdentity:String(repeating:"a",count:64),modelManifestSHA256:String(repeating:"b",count:64))
            if mode == .fresh {
                XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime,checkpointParent:parent))
                continue
            }
            XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime))
            let first=try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime,checkpointParent:parent)
            XCTAssertEqual(first.checkpointParent,parent);XCTAssertEqual(first.plan.trainingMode,mode)
            let changed=WorkflowCheckpointReference(projectID:owner.projectID,directoryID:parent.directoryID,
                runID:parent.runID,requestIdentity:parent.requestIdentity,modelManifestSHA256:String(repeating:"c",count:64))
            let second=try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime,checkpointParent:changed)
            XCTAssertNotEqual(first.requestIdentity,second.requestIdentity)
            let foreign=WorkflowCheckpointReference(projectID:UUID().uuidString,directoryID:parent.directoryID,
                runID:parent.runID,requestIdentity:parent.requestIdentity,modelManifestSHA256:parent.modelManifestSHA256)
            XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime,checkpointParent:foreign))
            var launches=0
            let run=WorkflowExecutionSession(request:first,owner:owner,sources:[:],directory:root.appendingPathComponent(UUID().uuidString),
                reservation:.init(acquire:{XCTFail("Unintegrated lifecycle must not reserve execution");return nil},release:{_ in}),
                launch:{_,_,_,_,_ in launches += 1;return WorkflowWorkerHandle(cancel:{})},onTerminal:{_ in})
            XCTAssertFalse(run.start());XCTAssertEqual(launches,0)
            XCTAssertTrue(run.statusMessage.contains("continuation"))
        }
    }
    @MainActor func testExperimentRequestsKeepAuthoringSourceAndBindVariantInputs()throws {
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        try WorkflowGuardEvidenceFixture.stage(runtime)
        let specification=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(owner.setExperimentSpecification(specification,for:address(.control,"setup"),lease:owner.lease()))
        let original=owner.document,base=try capture(owner,runtime)
        let file=root.appendingPathComponent("input.csv")
        try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
        let owned=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:owner.projectID)
        defer {owned.owner.close()}
        let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:owned.directory) {_ in try Data(contentsOf:file)}
        let first=try base.experimentVariant(ordinal:1,inputs:inputs,owner:owner)
        let second=try base.experimentVariant(ordinal:2,inputs:inputs,owner:owner)
        XCTAssertEqual(first.plan.settings.seed,7);XCTAssertEqual(second.plan.settings.seed,8)
        XCTAssertEqual(try base.experimentVariants(inputs:inputs,owner:owner),[first,second])
        XCTAssertEqual(first.authoringDocument,original);XCTAssertEqual(owner.document,original)
        XCTAssertNotEqual(first.document,original);XCTAssertTrue(first.isCurrent(in:owner))
        XCTAssertNotEqual(first.requestIdentity,second.requestIdentity)
        XCTAssertEqual(first.experiment?.inputsManifestSHA256,inputs.manifestSHA256)
        XCTAssertEqual(first.experiment?.ordinal,1)
        let wrongSources=[address(.data,"input"):try WorkflowCapturedSource.capture(file)]
        let wrongSession=WorkflowExecutionSession(request:first,owner:owner,sources:wrongSources,
            directory:root.appendingPathComponent(UUID().uuidString),
            reservation:.init(acquire:{XCTFail("Wrong group inputs cannot reserve");return nil},release:{_ in}),groupOwner:owned.owner,
            launch:{_,_,_,_,_ in XCTFail("Wrong group inputs cannot launch");return .init(cancel:{})},onTerminal:{_ in})
        XCTAssertFalse(wrongSession.start())
        XCTAssertTrue(wrongSession.statusMessage.contains("exact captured group inputs"))
        XCTAssertEqual(try base.experimentVariant(ordinal:1,inputs:inputs,owner:owner),first)
        XCTAssertThrowsError(try first.experimentVariant(ordinal:1,inputs:inputs,owner:owner))
        XCTAssertThrowsError(try base.experimentVariant(ordinal:0,inputs:inputs,owner:owner))
        XCTAssertThrowsError(try base.experimentVariant(ordinal:3,inputs:inputs,owner:owner))
        let changedInputs=try WorkflowExperimentInputs.capture(addresses:base.plan.sourceAddresses,parent:root) {_ in try WorkflowCapturedSource.capture(file)}
        XCTAssertNotEqual(try base.experimentVariant(ordinal:1,inputs:changedInputs,owner:owner).requestIdentity,first.requestIdentity)
        let policy = try ResolvedExecutionBudget.resolve(global: .factory, projectMaximumRunSeconds: 60)
        let budgeted = try WorkflowExecutionRequest.capture(from: owner, train: address(.control,"train"), runtime: runtime, executionBudget: policy, sampleCapture: .init(activationValues: false, predictionRows: false))
        let budgetedVariants = try budgeted.experimentVariants(inputs: inputs, owner: owner)
        XCTAssertTrue(budgetedVariants.allSatisfy { $0.executionBudget == policy })
        XCTAssertTrue(budgetedVariants.allSatisfy { $0.sampleCapture == budgeted.sampleCapture })
        XCTAssertTrue(budgetedVariants.allSatisfy { $0.experiment?.sourceRequestIdentity == budgeted.requestIdentity })
        XCTAssertNotEqual(budgetedVariants[0].requestIdentity, first.requestIdentity)
        XCTAssertTrue(owner.rename("Changed",lease:owner.lease()))
        XCTAssertFalse(first.isCurrent(in:owner))
        XCTAssertThrowsError(try base.experimentVariant(ordinal:1,inputs:inputs,owner:owner))
    }

    @MainActor func testBackgroundExperimentDerivationMatchesAndRejectsCancelledCapture() async throws {
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(owner.setExperimentSpecification(spec,for:address(.control,"setup"),lease:owner.lease()))
        let base=try capture(owner,runtime),file=root.appendingPathComponent("input.csv")
        try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
        let inputs=try WorkflowExperimentInputs.capture(addresses:base.plan.sourceAddresses,parent:root) {_ in try WorkflowCapturedSource.capture(file)}
        let expected=try base.experimentVariants(inputs:inputs,owner:owner)
        let actual=try await base.experimentVariantsInBackground(inputs:inputs,owner:owner)
        XCTAssertEqual(actual,expected)
        let task=Task { @MainActor in
            withUnsafeCurrentTask {$0?.cancel()}
            return try await base.experimentVariantsInBackground(inputs:inputs,owner:owner)
        }
        do {_=try await task.value;XCTFail("Cancellation must reject background requests")}
        catch {XCTAssertTrue(error is CancellationError)}
        try Data("changed".utf8).write(to:try XCTUnwrap(inputs.sources.values.first).url)
        do {_=try await base.experimentVariantsInBackground(inputs:inputs,owner:owner);XCTFail("Tampered copies must reject")}
        catch {XCTAssertFalse(error is CancellationError)}
        XCTAssertEqual(owner.document,base.authoringDocument)
    }

    @MainActor func testBackgroundLedgerCreationPersistsAndHonorsPreCancellation() async throws {
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(owner.setExperimentSpecification(spec,for:address(.control,"setup"),lease:owner.lease()))
        let base=try capture(owner,runtime),file=root.appendingPathComponent("input.csv")
        try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
        let inputs=try WorkflowExperimentInputs.capture(addresses:base.plan.sourceAddresses,parent:root) {_ in try WorkflowCapturedSource.capture(file)}
        let requests=try base.experimentVariants(inputs:inputs,owner:owner)
        let groupOwner=try WorkflowGroupOwnership.initialize(directory:inputs.directory,groupID:inputs.directory.lastPathComponent)
        defer {groupOwner.close()}
        let cancelled=Task { @MainActor in
            withUnsafeCurrentTask {$0?.cancel()}
            return try await WorkflowExperimentGroupStore.createInBackground(requests:requests,directory:inputs.directory,groupOwner:groupOwner)
        }
        do {_=try await cancelled.value;XCTFail("Pre-cancelled creation must not write")}
        catch {XCTAssertTrue(error is CancellationError)}
        XCTAssertFalse(FileManager.default.fileExists(atPath:inputs.directory.appendingPathComponent("group.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:inputs.directory.appendingPathComponent("authoring.json").path))
        let store=try await WorkflowExperimentGroupStore.createInBackground(requests:requests,directory:inputs.directory,groupOwner:groupOwner)
        XCTAssertEqual(store.record,try WorkflowExperimentGroupStore.open(directory:inputs.directory).record)
        try store.cancelPending()
        XCTAssertTrue(store.record.isTerminal)
        XCTAssertEqual(store.record,try WorkflowExperimentGroupStore.open(directory:inputs.directory).record)
    }

    @MainActor func testBackgroundArchiveCapturePublishesExactInputsAndRejectsChanges() async throws {
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let request=try capture(owner,runtime),file=root.appendingPathComponent("input.csv")
        let bytes=Data("a,b,label\n1,2,0\n".utf8)
        try bytes.write(to:file)
        let sources=[address(.data,"input"):try WorkflowCapturedSource.capture(file)]
        let directory=root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        let runID=UUID()
        let cancelled=Task { @MainActor in
            withUnsafeCurrentTask {$0?.cancel()}
            return try await WorkflowArchiveCaptureWriter.writeInBackground(request:request,runID:runID,sources:sources,
                directory:directory,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        }
        do {_=try await cancelled.value;XCTFail("Pre-cancelled archive must not write")}
        catch {XCTAssertTrue(error is CancellationError)}
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath:directory.path).isEmpty)
        let result=try await WorkflowArchiveCaptureWriter.writeInBackground(request:request,runID:runID,sources:sources,
            directory:directory,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        let archive=try Data(contentsOf:directory.appendingPathComponent(result.reference.path))
        XCTAssertEqual(WorkflowArchiveCodec.hash(archive),result.reference.sha256)
        XCTAssertEqual(try Data(contentsOf:URL(fileURLWithPath:try XCTUnwrap(result.sourceInputs["input"]))),bytes)
        try Data("changed".utf8).write(to:file)
        let second=root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:second,withIntermediateDirectories:false)
        do {
            _=try await WorkflowArchiveCaptureWriter.writeInBackground(request:request,runID:UUID(),sources:sources,
                directory:second,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
            XCTFail("Changed source must not publish a capture")
        } catch {}
        XCTAssertFalse(FileManager.default.fileExists(atPath:second.appendingPathComponent("capture.json").path))
    }

    @MainActor func testDurableGroupTransitionsCancellationAndReopen()throws {
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(owner.setExperimentSpecification(spec,for:address(.control,"setup"),lease:owner.lease()))
        let base=try capture(owner,runtime),file=root.appendingPathComponent("input.csv")
        try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
        let inputs=try WorkflowExperimentInputs.capture(addresses:base.plan.sourceAddresses,parent:root) {_ in try WorkflowCapturedSource.capture(file)}
        let requests=try (1...2).map {try base.experimentVariant(ordinal:$0,inputs:inputs,owner:owner)}
        let groupOwner=try WorkflowGroupOwnership.initialize(directory:inputs.directory,groupID:inputs.directory.lastPathComponent)
        defer {groupOwner.close()}
        let store=try WorkflowExperimentGroupStore.create(requests:requests,directory:inputs.directory,groupOwner:groupOwner)
        XCTAssertEqual(store.record.variants.map(\.status),[.pending,.pending])
        let run=UUID(),directory=UUID()
        try store.begin(ordinal:1,runID:run,directoryID:directory)
        XCTAssertThrowsError(try store.begin(ordinal:2,runID:UUID(),directoryID:UUID()),"Only one active variant")
        XCTAssertNotNil(store.record.createdAtUnixSeconds)
        let reopened=try WorkflowExperimentGroupStore.open(directory:inputs.directory)
        XCTAssertEqual(reopened.record.createdAtUnixSeconds,store.record.createdAtUnixSeconds)
        XCTAssertEqual(reopened.record,store.record)
        try store.cancelPending()
        XCTAssertTrue(store.record.stopRequested)
        XCTAssertEqual(store.record.variants.map(\.status),[.running,.cancelled])
        XCTAssertThrowsError(try store.begin(ordinal:2,runID:UUID(),directoryID:UUID()))
        try store.finish(ordinal:1,runID:run,directoryID:directory,requestIdentity:requests[0].requestIdentity,status:.cancelled)
        XCTAssertTrue(store.record.isTerminal)
        XCTAssertEqual(try WorkflowExperimentGroupStore.open(directory:inputs.directory).record,store.record)
        XCTAssertThrowsError(try store.finish(ordinal:1,runID:run,directoryID:directory,requestIdentity:requests[0].requestIdentity,status:.completed))
        XCTAssertThrowsError(try reopened.cancelPending(),"Stale writer must not overwrite newer progress")
        let newInputs=try WorkflowExperimentInputs.capture(addresses:base.plan.sourceAddresses,parent:root) {_ in try WorkflowCapturedSource.capture(file)}
        let newRequests=try (1...2).map {try base.experimentVariant(ordinal:$0,inputs:newInputs,owner:owner)}
        let newOwner=try WorkflowGroupOwnership.initialize(directory:newInputs.directory,groupID:newInputs.directory.lastPathComponent)
        defer {newOwner.close()}
        XCTAssertThrowsError(try WorkflowExperimentGroupStore.create(requests:[newRequests[0]],directory:newInputs.directory,groupOwner:newOwner))
        let interrupted=try WorkflowExperimentGroupStore.create(requests:newRequests,directory:newInputs.directory,groupOwner:newOwner)
        try interrupted.begin(ordinal:1,runID:run,directoryID:directory)
        XCTAssertThrowsError(try interrupted.finish(ordinal:1,runID:run,directoryID:directory,requestIdentity:newRequests[0].requestIdentity,status:.completed),"No completion without outcome pin")
        let saved=interrupted.record
        let ledger=newInputs.directory.appendingPathComponent("group.json"),bytes=try Data(contentsOf:ledger)
        try (bytes+Data(" ".utf8)).write(to:ledger)
        XCTAssertThrowsError(try interrupted.recoverInterrupted())
        XCTAssertEqual(interrupted.record,saved,"Failed persistence must not publish state")
        try bytes.write(to:ledger)
        XCTAssertThrowsError(try interrupted.recoverInterrupted(),"Recovery requires explicit recovery authority, not a stale ledger inference")
        XCTAssertEqual(interrupted.record,saved)
        XCTAssertEqual(try Data(contentsOf:ledger),bytes)
        XCTAssertFalse(interrupted.record.isTerminal)
        XCTAssertEqual(try WorkflowExperimentGroupStore.open(directory:newInputs.directory).record,interrupted.record)
    }

    @MainActor func testRunnerStopsBeforeDuringAndAfterLaunchFailure() async throws {
        for mode in ["before","active","launchFailure","authority"] {
            let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
            try WorkflowGuardEvidenceFixture.stage(runtime)
            let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
            XCTAssertTrue(owner.setExperimentSpecification(spec,for:address(.control,"setup"),lease:owner.lease()))
            let base=try capture(owner,runtime),file=root.appendingPathComponent("input.csv")
            try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
            let owned=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:owner.projectID)
            let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:owned.directory) {_ in try Data(contentsOf:file)}
            let requests=try (1...2).map {try base.experimentVariant(ordinal:$0,inputs:inputs,owner:owner)}
            let groupOwner=owned.owner
            defer {groupOwner.close()}
            let group=try WorkflowExperimentGroupStore.create(requests:requests,directory:inputs.directory,groupOwner:groupOwner)
            let done=expectation(description:mode+" terminal")
            var launches=0,cancels=0,releases=0,terminals=0
            let runner=try WorkflowExperimentRunner(requests:requests,owner:owner,group:group,groupOwner:groupOwner,workspaceDirectory:root,
                reservation:.init(acquire:{UUID()},release:{_ in releases+=1}),
                launch:{_,_,_,_,exit in
                    launches+=1
                    if mode == "launchFailure" {throw DataFlowContractError("simulated launch failure")}
                    return .init(cancel:{cancels+=1;exit(15,"cancelled")})
                },canContinue:{mode != "authority"},onTerminal:{terminals+=1;done.fulfill()})
            if mode == "before" {runner.cancel();XCTAssertFalse(runner.start())}
            else {
                _=runner.start()
                if mode == "active" {
                    let deadline=Date().addingTimeInterval(5)
                    while launches == 0 && Date()<deadline {try await Task.sleep(for:.milliseconds(1))}
                    runner.cancel()
                }
            }
            await fulfillment(of:[done],timeout:5)
            XCTAssertFalse(runner.isRunning);XCTAssertEqual(terminals,1)
            XCTAssertEqual(launches,["active","launchFailure"].contains(mode) ? 1:0)
            XCTAssertEqual(releases,launches);XCTAssertEqual(cancels,mode == "active" ? 1:0)
            XCTAssertTrue(group.record.isTerminal)
            XCTAssertEqual(group.record.variants.last?.status,.cancelled)
            XCTAssertEqual(group.record.variants.first?.status,mode == "launchFailure" ? .failed:.cancelled)
            runner.cancel();XCTAssertEqual(terminals,1)
            XCTAssertFalse(owner.isBusy)
        }
    }

    @MainActor func testMaximumBatchCapturePreservesOrderAndRejectsChangedInputs()throws {
        let(root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:(1...32).map {.integer(Int64($0))})],
            budget:.init(maximumRuns:32,maximumTotalEpochs:96))
        XCTAssertTrue(owner.setExperimentSpecification(spec,for:address(.control,"setup"),lease:owner.lease()))
        let original=owner.document,base=try capture(owner,runtime),file=root.appendingPathComponent("input.csv")
        try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
        let inputs=try WorkflowExperimentInputs.capture(addresses:base.plan.sourceAddresses,parent:root) {_ in try WorkflowCapturedSource.capture(file)}
        let batch=try base.experimentVariants(inputs:inputs,owner:owner)
        XCTAssertEqual(batch.map {$0.plan.settings.seed},Array(1...32))
        XCTAssertEqual(batch.map {$0.experiment?.ordinal},Array(1...32).map(Optional.some))
        XCTAssertEqual(Set(batch.map(\.requestIdentity)).count,32)
        XCTAssertTrue(batch.allSatisfy {$0.authoringDocument == original && $0.experimentInputs == inputs})
        XCTAssertEqual(owner.document,original)
        let copied=try XCTUnwrap(inputs.sources.values.first).url
        try Data("changed".utf8).write(to:copied)
        XCTAssertThrowsError(try base.experimentVariants(inputs:inputs,owner:owner))
    }

    private func makeRuntime(_ root: URL) throws -> RuntimeLocation {
        for name in [".venv/bin", "runner", "scripts", "Sources/KwipMLCore/Resources"] {
            try FileManager.default.createDirectory(at: root.appendingPathComponent(name), withIntermediateDirectories: true)
        }
        try Data("fixture interpreter; never executed".utf8).write(to: root.appendingPathComponent(".venv/bin/python"))
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.appendingPathComponent(".venv/bin/python").path)
        for name in ["runner/main.py", "runner/engine.py", "scripts/runtime_launch.py"] {
            try Data("# source fixture\n".utf8).write(to: root.appendingPathComponent(name))
        }
        try JSONEncoder().encode(ComponentRegistry.builtIn()).write(to: root.appendingPathComponent("Sources/KwipMLCore/Resources/components.json"))
        try JSONEncoder().encode(dataRegistry()).write(to: root.appendingPathComponent("runner/dataflow-operators.json"))
        return try RuntimeLocation.resolve(resources: nil, developmentRoot: root, allowDevelopment: true)
    }
    @MainActor func capture(_ owner: WorkflowAuthoringStore, _ runtime: RuntimeLocation) throws -> WorkflowExecutionRequest {
        try .capture(from: owner, train: address(.control,"train"), runtime: runtime)
    }
    @MainActor func testGradientCaptureRequiresVerifiedDeviceCapabilityBeforeRequest() throws {
        let (root,owner,runtime)=try setup();defer {try? FileManager.default.removeItem(at:root)}
        let setup=address(.control,"setup")
        let node=try XCTUnwrap(owner.document.nodes.first {$0.address == setup})
        guard case .runSetup(var config)=node.payload else {return XCTFail("Missing setup")}
        config["device"] = .string("cpu")
        config["gradientCapture"] = .object(["schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":1,"maxCaptures":2])
        XCTAssertNotNil(owner.apply(.configure(address:setup,config:config),lease:owner.lease()))
        XCTAssertThrowsError(try capture(owner,runtime))
        XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime,supportedGradientDevices:["mps"]))
        let request=try WorkflowExecutionRequest.capture(from:owner,train:address(.control,"train"),runtime:runtime,supportedGradientDevices:["cpu"])
        XCTAssertEqual(request.gradientArguments,["--capture-gradients","--gradient-every-epochs","1","--gradient-max-batches-per-epoch","1","--gradient-max-captures","2"])
    }
    @MainActor func testVersionTwoReuseRequiresExplicitPrivateCacheSelection() throws {
        let (root, _, runtime) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        var document = try WorkflowCodec.migrateToVersion2(fixture(block: true))
        let index = try XCTUnwrap(document.nodes.firstIndex { $0.address == address(.control, "setup") })
        guard case .runSetup(var config) = document.nodes[index].payload else { return XCTFail("Expected run setup") }
        config["preparationCachePolicy"] = .string("reuse")
        document.nodes[index].payload = .runSetup(config: config)
        let library = ProjectLibrary(directory: root.appendingPathComponent("version-two-projects"))
        _ = try library.saveSource(.workflow(document))
        let owner = try WorkflowAuthoringStore(library: library, projectID: document.projectID,
            modelRegistry: .builtIn(), dataRegistry: dataRegistry())
        XCTAssertThrowsError(try capture(owner, runtime)) { error in
            XCTAssertTrue(error.localizedDescription.lowercased().contains("selected prior preparation"), error.localizedDescription)
        }
    }
    @MainActor func testFreshIdentityRetainsPreContinuationEncoding()throws {
        let(root,owner,runtime)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let request=try capture(owner,runtime)
        struct HistoricalIdentity:Encodable {
            let schemaVersion:Int
            let priorCache:WorkflowArchiveCacheReference?
            let document:WorkflowDocument
            let train:WorkflowNodeAddress
            let modelRegistrySHA256:String
            let dataRegistrySHA256:String
            let runtimeEvidence:WorkflowRuntimeEvidence
            let executable:String
            let arguments:[String]
            let registry:String
            let workingDirectory:String
        }
        let old=HistoricalIdentity(schemaVersion:request.document.schemaVersion,priorCache:nil,document:request.document,
            train:request.train,modelRegistrySHA256:request.modelRegistrySHA256,dataRegistrySHA256:request.dataRegistrySHA256,
            runtimeEvidence:request.runtimeEvidence,executable:runtime.executableURL.path,arguments:runtime.argumentPrefix,
            registry:runtime.registryURL.path,workingDirectory:runtime.workingDirectoryURL.path)
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let hash=SHA256.hash(data:try encoder.encode(old)).map {String(format:"%02x",$0)}.joined()
        XCTAssertEqual(request.requestIdentity,hash)
        XCTAssertNil(request.checkpointParent)
    }
    @MainActor func testCaptureIsImmutableAndDifferentFromArchitectureCheckpointIdentity() throws {
        let (root, owner, runtime) = try setup(); defer { try? FileManager.default.removeItem(at: root) }
        let first = try capture(owner,runtime), original = owner.document
        XCTAssertEqual(first.document,original); XCTAssertEqual(first.lease,owner.lease())
        XCTAssertTrue(first.plan.model.graph.nodes.contains { $0.type == "linear" })
        XCTAssertNotNil(first.plan.model.provenance)
        XCTAssertNotEqual(first.requestIdentity, first.plan.model.semanticSHA256)
        XCTAssertTrue(first.isCurrent(in:owner))
        XCTAssertNotNil(owner.apply(.configure(address:address(.control,"mapping"),config:["featureColumns":.array([.string("b"),.string("a")]),"targetColumn":.string("label")]),lease:owner.lease()))
        let mapping = try capture(owner,runtime)
        XCTAssertFalse(first.isCurrent(in:owner)); XCTAssertEqual(first.document,original)
        XCTAssertNotEqual(first.requestIdentity,mapping.requestIdentity)
        XCTAssertEqual(first.plan.model.semanticSHA256,mapping.plan.model.semanticSHA256)
        XCTAssertEqual(first.plan.featureColumns,["a","b"]); XCTAssertEqual(mapping.plan.featureColumns,["b","a"])
        XCTAssertNotNil(owner.apply(.configure(address:address(.data,"split"),config:["strategy":.string("random"),"seed":19,"validationPercent":20,"testPercent":20,"stratifyColumn":.string("label")]),lease:owner.lease()))
        let split = try capture(owner,runtime)
        XCTAssertNotEqual(split.requestIdentity,mapping.requestIdentity)
        XCTAssertEqual(split.plan.model.semanticSHA256,mapping.plan.model.semanticSHA256)
        XCTAssertNotNil(owner.apply(.configure(address:address(.control,"setup"),config:["device":.string("cpu"),"epochs":.integer(4),"batchSize":.integer(32),"seed":.integer(17),"learningRate":.number(0.02)]),lease:owner.lease()))
        let settings = try capture(owner,runtime)
        XCTAssertNotEqual(split.requestIdentity,settings.requestIdentity); XCTAssertEqual(settings.plan.settings.epochs,4)
        XCTAssertEqual(first.plan.settings.epochs,3)
    }
    @MainActor func testCaptureRejectsBusyRetiredAndUnresolvedDraftWhileCallbacksRemainBusyCurrent() throws {
        let (root,owner,runtime) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let request = try capture(owner,runtime)
        owner.setExecutionBusy(true)
        XCTAssertTrue(request.isCurrent(in:owner)); XCTAssertThrowsError(try capture(owner,runtime))
        owner.setExecutionBusy(false)
        XCTAssertTrue(owner.beginDraft(root:address(.model,"features"),lease:owner.lease()))
        XCTAssertThrowsError(try capture(owner,runtime)); XCTAssertFalse(request.isCurrent(in:owner))
        XCTAssertTrue(owner.discardDraft(lease:owner.blockLease(root:address(.model,"features"))))
        let current = try capture(owner,runtime)
        owner.invalidate(); XCTAssertFalse(current.isCurrent(in:owner)); XCTAssertThrowsError(try capture(owner,runtime))
    }
    @MainActor func testSourceEvidenceChangesWithoutAuthoringOrArchitectureChange() throws {
        let (root,owner,runtime) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let first = try capture(owner,runtime)
        XCTAssertEqual(first.runtimeEvidence.workerFiles["runner/engine.py"], SHA256.hash(data:Data("# source fixture\n".utf8)).map { String(format:"%02x",$0) }.joined())
        XCTAssertNil(first.runtimeEvidence.manifestSHA256)
        try Data("# changed implementation\n".utf8).write(to:runtime.workingDirectoryURL.appendingPathComponent("runner/engine.py"))
        let changed = try capture(owner,runtime)
        XCTAssertEqual(first.document,changed.document); XCTAssertTrue(first.isCurrent(in:owner))
        XCTAssertNotEqual(first.runtimeEvidence,changed.runtimeEvidence)
        XCTAssertNotEqual(first.requestIdentity,changed.requestIdentity)
        XCTAssertEqual(first.plan.model.semanticSHA256,changed.plan.model.semanticSHA256)
    }
    @MainActor func testRegistryEvidenceComesFromOwnerAndReopenedSessionIsDistinct() throws {
        let (root,owner,runtime) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let first = try capture(owner,runtime)
        let library = ProjectLibrary(directory:root.appendingPathComponent("projects"))
        var data = try dataRegistry()
        data.operators[0].title += " revised label"
        let other = try WorkflowAuthoringStore(library:library,projectID:owner.projectID,
            modelRegistry:.builtIn(),dataRegistry:data)
        XCTAssertThrowsError(try capture(other,runtime)) { XCTAssertTrue($0.localizedDescription.contains("registry")) }
        try JSONEncoder().encode(data).write(to:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let second = try capture(other,runtime)
        XCTAssertFalse(first.isCurrent(in:other))
        XCTAssertEqual(first.document,second.document)
        XCTAssertEqual(first.modelRegistrySHA256,second.modelRegistrySHA256)
        XCTAssertNotEqual(first.dataRegistrySHA256,second.dataRegistrySHA256)
        XCTAssertNotEqual(first.requestIdentity,second.requestIdentity)
    }

    @MainActor func testModelRegistryMustMatchWorkerWhileFormattingOnlyChangesAreAllowed() throws {
        let (root,owner,runtime) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let first = try capture(owner,runtime)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
        try encoder.encode(owner.modelRegistry).write(to:runtime.registryURL)
        try encoder.encode(owner.dataRegistry).write(to:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
        let formatted = try capture(owner,runtime)
        XCTAssertEqual(first.modelRegistrySHA256,formatted.modelRegistrySHA256)
        XCTAssertEqual(first.dataRegistrySHA256,formatted.dataRegistrySHA256)
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:runtime.registryURL)) as? [String:Any])
        var components = try XCTUnwrap(object["components"] as? [[String:Any]])
        components[0]["title"] = "Different runtime component"
        object["components"] = components
        try JSONSerialization.data(withJSONObject:object).write(to:runtime.registryURL)
        XCTAssertThrowsError(try capture(owner,runtime)) { XCTAssertTrue($0.localizedDescription.contains("registry")) }
    }

    @MainActor func testBundledManifestPinsActualInventoryAndRejectsChangedOrUnlistedBytes() throws {
        let (root,owner,development) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let resources = root.appendingPathComponent("Resources")
        let payload = resources.appendingPathComponent("KwipMLRuntime")
        try FileManager.default.createDirectory(at:payload,withIntermediateDirectories:true)
        try FileManager.default.copyItem(at:development.workingDirectoryURL.appendingPathComponent("runner"),to:payload.appendingPathComponent("runner"))
        try FileManager.default.copyItem(at:development.entryURL,to:payload.appendingPathComponent("launch.py"))
        try FileManager.default.copyItem(at:development.registryURL,to:payload.appendingPathComponent("components.json"))
        try FileManager.default.copyItem(at:development.executableURL,to:payload.appendingPathComponent("python"))
        let names = ["launch.py","components.json","runner/main.py","runner/engine.py","runner/dataflow-operators.json"]
        var hashes:[String:String] = [:]
        for name in names {
            hashes[name] = SHA256.hash(data:try Data(contentsOf:payload.appendingPathComponent(name))).map { String(format:"%02x",$0) }.joined()
        }
        let manifest:[String:Any] = ["schemaVersion":1,"pythonExecutable":"python","entryPoint":"launch.py","registry":"components.json",
            "pythonVersion":"3.13.6","packages":["torch":"2.14.1"],"workerIntegrityVersion":1,"workerSourceHashes":hashes,
            "entryHash":hashes["launch.py"]!,"registryHash":hashes["components.json"]!]
        let manifestBytes = try JSONSerialization.data(withJSONObject:manifest,options:[.sortedKeys])
        try manifestBytes.write(to:payload.appendingPathComponent("runtime-manifest.json"))
        let bundled = try RuntimeLocation.resolve(resources:resources)
        let request = try capture(owner,bundled)
        XCTAssertEqual(request.runtimeEvidence.workerFiles,hashes)
        XCTAssertEqual(request.runtimeEvidence.manifestSHA256,SHA256.hash(data:manifestBytes).map { String(format:"%02x",$0) }.joined())
        XCTAssertEqual(request.runtime.argumentPrefix.prefix(2),["-I","-B"])
        var optional = manifest
        optional.removeValue(forKey:"entryHash"); optional.removeValue(forKey:"registryHash")
        try JSONSerialization.data(withJSONObject:optional).write(to:payload.appendingPathComponent("runtime-manifest.json"))
        XCTAssertNoThrow(try capture(owner,bundled), "Absent optional hashes are allowed by the launcher")
        for field in ["entryHash","registryHash"] {
            var invalid = optional; invalid[field] = NSNull()
            try JSONSerialization.data(withJSONObject:invalid).write(to:payload.appendingPathComponent("runtime-manifest.json"))
            XCTAssertThrowsError(try capture(owner,bundled), "Present null must fail like the launcher")
        }
        try manifestBytes.write(to:payload.appendingPathComponent("runtime-manifest.json"))
        try Data("# unlisted\n".utf8).write(to:payload.appendingPathComponent("runner/extra.py"))
        XCTAssertThrowsError(try capture(owner,bundled))
        try FileManager.default.removeItem(at:payload.appendingPathComponent("runner/extra.py"))
        try Data("# changed\n".utf8).write(to:payload.appendingPathComponent("runner/engine.py"))
        XCTAssertThrowsError(try capture(owner,bundled))
    }
    @MainActor func testDevelopmentExclusionsFollowCanonicalPackagerWithoutIgnoringSourceAdditions() throws {
        let (root,owner,runtime) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let first = try capture(owner,runtime)
        let worker = runtime.workingDirectoryURL.appendingPathComponent("runner")
        for name in ["tests","__pycache__",".pytest_cache"] {
            let directory = worker.appendingPathComponent(name)
            try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
            try Data("ignored development file".utf8).write(to:directory.appendingPathComponent("fixture.txt"))
        }
        try Data("ignored bytecode".utf8).write(to:worker.appendingPathComponent("module.pyc"))
        XCTAssertEqual(try capture(owner,runtime).runtimeEvidence,first.runtimeEvidence)
        try Data("new runnable source".utf8).write(to:worker.appendingPathComponent("new_module.py"))
        XCTAssertNotEqual(try capture(owner,runtime).runtimeEvidence,first.runtimeEvidence)
    }

    @MainActor func testPendingNumericEditCannotBeOmittedFromCaptureAndClearRestoresReadiness() throws {
        let (root,owner,runtime) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let contextID = UUID(), node = address(.data,"split")
        XCTAssertTrue(owner.select(.init(nodeAddresses:[node]),lease:owner.lease()))
        XCTAssertTrue(owner.setDataEditPending(contextID:contextID,address:node,key:"seed",pending:true,lease:owner.lease()))
        XCTAssertTrue(owner.hasPendingDataEdits)
        XCTAssertThrowsError(try capture(owner,runtime)) { XCTAssertTrue($0.localizedDescription.lowercased().contains("pending")) }
        // Pending authoring must fail before runtime I/O or compilation can mask it.
        let bytes = try Data(contentsOf:runtime.entryURL)
        try FileManager.default.removeItem(at:runtime.entryURL)
        XCTAssertThrowsError(try capture(owner,runtime)) { XCTAssertTrue($0.localizedDescription.lowercased().contains("pending")) }
        try bytes.write(to:runtime.entryURL)
        owner.clearDataEdits(contextID:contextID)
        XCTAssertFalse(owner.hasPendingDataEdits)
        XCTAssertNoThrow(try capture(owner,runtime))
    }

    @MainActor func testOptInRealRuntimeInventoriesMatchOwnerRegistries() throws {
        let environment = ProcessInfo.processInfo.environment
        let resources = environment["KWIPML_TEST_RUNTIME_RESOURCES"]
        let development = environment["KWIPML_TEST_DEVELOPMENT_ROOT"]
        guard resources != nil || development != nil else {
            throw XCTSkip("Set KWIPML_TEST_RUNTIME_RESOURCES and/or KWIPML_TEST_DEVELOPMENT_ROOT to inspect an explicit real runtime; no worker is launched.")
        }
        var runtimes:[RuntimeLocation] = []
        if let resources {
            runtimes.append(try RuntimeLocation.resolve(resources:URL(fileURLWithPath:resources)))
        }
        if let development {
            runtimes.append(try RuntimeLocation.resolve(resources:nil,developmentRoot:URL(fileURLWithPath:development),allowDevelopment:true))
        }
        for runtime in runtimes {
            let root = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
            defer { try? FileManager.default.removeItem(at:root) }
            let source = fixture(block:true), library = ProjectLibrary(directory:root)
            _ = try library.saveSource(.workflow(source),id:source.projectID)
            let model = try JSONDecoder().decode(ComponentRegistry.self,from:Data(contentsOf:runtime.registryURL))
            let data = try DataFlowOperatorRegistry.load(from:runtime.workingDirectoryURL.appendingPathComponent("runner/dataflow-operators.json"))
            let owner = try WorkflowAuthoringStore(library:library,projectID:source.projectID,modelRegistry:model,dataRegistry:data)
            let request = try capture(owner,runtime)
            XCTAssertGreaterThan(request.runtimeEvidence.workerFiles.count,3)
            XCTAssertNotNil(request.runtimeEvidence.workerFiles["runner/main.py"])
            XCTAssertNotNil(request.runtimeEvidence.workerFiles["runner/dataflow-operators.json"])
            XCTAssertTrue(try request.runtimeEvidence.matchesCurrent(runtime))
            XCTAssertEqual(request.runtimeEvidence.manifestSHA256 != nil,runtime.kind == .bundled)
        }
    }

    @MainActor func testRuntimeSymlinkAndOversizedWorkerAreRejected() throws {
        let (root,owner,runtime) = try setup(); defer { try? FileManager.default.removeItem(at:root) }
        let engine = runtime.workingDirectoryURL.appendingPathComponent("runner/engine.py")
        try FileManager.default.removeItem(at:engine)
        try FileManager.default.createSymbolicLink(at:engine,withDestinationURL:runtime.entryURL)
        XCTAssertThrowsError(try capture(owner,runtime))
        try FileManager.default.removeItem(at:engine)
        try Data(repeating:65,count:10_000_001).write(to:engine)
        XCTAssertThrowsError(try capture(owner,runtime))
    }
}
