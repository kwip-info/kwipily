import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowTrainedArtifactFlowTests:XCTestCase {
    func testExplicitSaveKeepsSharedActiveOperationUntilPublication()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        let deadline=Date().addingTimeInterval(60)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertNotNil(host.frozenBindings.reference(for:WorkflowExamples.trainAddress),host.statusMessage)
        let save=WorkflowNodeAddress(domain:.control,localID:"explicit-save")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:save,payload:.saveModel(config:["format":.string("checkpoint")]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:save,port:"model")),lease:owner.lease()))
        XCTAssertTrue(host.modelDestinations.set(destination:root.appendingPathComponent("explicit.pt"),for:save,lease:owner.lease()))
        let settingsBeforeSave=try workspace.projectSettingsSnapshot(projectID:owner.projectID)
        let settingsProposal=try workspace.previewProjectSettings(.historyCapture(.none),baseline:settingsBeforeSave)
        let busy=expectation(description:"Explicit save retains shared busy ownership"),saved=expectation(description:"Explicit save completed")
        var status:AnyCancellable?
        let observation=host.$modelSave.compactMap{$0}.sink { session in
            status=session.$statusMessage.filter{$0.hasPrefix("Verifying captured model")}.prefix(1).sink {_ in
                XCTAssertTrue(session.isBusy)
                XCTAssertTrue(host.hasActiveOperation,"Export must be owned even without automatic model consumers")
                do {_ = try workspace.applyProjectSettings(settingsProposal);XCTFail("Settings changed during an active operation")} catch {XCTAssertEqual((error as? ProjectSettingsFailure)?.code,"settings_busy")}
                XCTAssertEqual(owner.runHistoryPolicy,settingsBeforeSave.history)
                busy.fulfill()
            }
        }
        let publication=host.$modelSaveReceipts.filter{!$0.isEmpty}.prefix(1).sink {_ in saved.fulfill()}
        defer {observation.cancel();publication.cancel();status?.cancel()}
        let trainingIdentity=host.operationIdentity
        XCTAssertTrue(host.saveModel(address:save),host.statusMessage)
        let saveIdentity=host.operationIdentity
        XCTAssertNotEqual(saveIdentity,trainingIdentity,"An explicit Save must not borrow its producer job identity")
        XCTAssertFalse(host.saveModel(address:save),"A second explicit Save cannot enter during pending artifact resolution")
        XCTAssertEqual(host.operationIdentity,saveIdentity,"Refusing an active operation must preserve its ownership")
        await fulfillment(of:[busy,saved],timeout:30)
        XCTAssertEqual(host.operationIdentity,saveIdentity)
    }
    func testExplicitSaveAndCapturedTerminalNeverWriteSiblingOutputs()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        // Train with a connected View so required complete predictions are retained.
        let view=WorkflowNodeAddress(domain:.data,localID:"captured-view")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"predictions"),to:.init(address:view,port:"table")),lease:owner.lease()))
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        var deadline=Date().addingTimeInterval(60)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        let producer=try XCTUnwrap(host.currentExecution?.runID)
        XCTAssertNotNil(host.frozenBindings.reference(for:WorkflowExamples.trainAddress),host.statusMessage)
        let save=WorkflowNodeAddress(domain:.control,localID:"selected-save")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:save,payload:.saveModel(config:["format":.string("checkpoint")]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:save,port:"model")),lease:owner.lease()))
        XCTAssertTrue(host.modelDestinations.set(destination:root.appendingPathComponent("selected.pt"),for:save,lease:owner.lease()))
        let outputs=["selected-output","sibling-output"].map{WorkflowNodeAddress(domain:.data,localID:$0)}
        for address in outputs {
            XCTAssertTrue(host.canvas.apply(.insert(.init(address:address,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string(address.localID+".csv"),"separator":.string(","),"columns":.array([]),"overwrite":.bool(false)]))),lease:owner.lease()))
            XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"predictions"),to:.init(address:address,port:"table")),lease:owner.lease()))
            host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[address]),extending:false)
            host.outputDestinations.pickDirectory={root}
            XCTAssertTrue(host.outputDestinations.chooseDestination(for:address,lease:owner.lease()),host.outputDestinations.statusMessage)
        }
        XCTAssertTrue(host.saveModel(address:save),host.statusMessage)
        deadline=Date().addingTimeInterval(30)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        // A queued automatic preparation would hold hasActiveOperation through its task.
        try await Task.sleep(for:.milliseconds(200))
        XCTAssertNotNil(host.modelSaveReceiptsByAddress[save],host.statusMessage)
        for output in outputs {XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent(output.localID+".csv").path),"Explicit Save cannot publish unrelated tables")}
        XCTAssertTrue(host.prepareData(at:view),host.statusMessage)
        deadline=Date().addingTimeInterval(30)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertEqual(host.scopedPreparation?.phase,.completed,host.statusMessage)
        XCTAssertEqual(host.currentExecution?.runID,producer)
        for output in outputs {XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent(output.localID+".csv").path))}
        XCTAssertTrue(host.prepareData(at:outputs[0]),host.statusMessage)
        deadline=Date().addingTimeInterval(30)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertEqual(host.scopedPreparation?.phase,.completed,host.statusMessage)
        XCTAssertTrue(FileManager.default.fileExists(atPath:root.appendingPathComponent(outputs[0].localID+".csv").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent(outputs[1].localID+".csv").path))
        XCTAssertEqual(host.currentExecution?.runID,producer,"Captured preparation never retrains or replaces the producer")
        let setup=try XCTUnwrap(owner.document.nodes.first{$0.address==WorkflowExamples.trainAddress})
        guard case .train(var config)=setup.payload else {return XCTFail("Missing Train")}
        guard case .object(var run)=config["runSetup"] else {return XCTFail("Missing run setup")}
        run["epochs"] = .integer(4);config["runSetup"] = .object(run)
        XCTAssertTrue(host.canvas.apply(.configure(address:setup.address,config:config),lease:owner.lease()))
        XCTAssertThrowsError(try host.scopedDataPlan(at:outputs[1])){XCTAssertTrue($0 is WorkflowCapturedTableRequirement)}
        XCTAssertFalse(host.prepareData(at:outputs[1]))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent(outputs[1].localID+".csv").path))

    }
    func testInspectingPreparedTableAtCompletionDoesNotLoseTrainedModelPublication()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        let source=try XCTUnwrap(owner.document.nodes.first(where:DataFlowSourceOperators.isSource))
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[source.address]),extending:false)
        let read=expectation(description:"Source read reserved at completion"),published=expectation(description:"Model published after read joins")
        let observation=host.$outcomes.filter {!$0.isEmpty}.prefix(1).sink {_ in
            Task { @MainActor in
                let query=host.fullTableViewSession(for:source.address)
                XCTAssertNotNil(query)
                XCTAssertEqual(query?.load(),true)
                read.fulfill()
            }
        }
        let publication=host.$statusMessage.filter {$0.hasPrefix("Trained model ready") || $0.hasPrefix("Trained model unavailable")}.prefix(1).sink {_ in published.fulfill()}
        defer {observation.cancel();publication.cancel()}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        await fulfillment(of:[read,published],timeout:60)
        XCTAssertNotNil(host.frozenBindings.reference(for:WorkflowExamples.trainAddress),host.statusMessage)
    }
    func testTrainFeedsValidationAndExactCheckpointSaveInOrder()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        let save=WorkflowNodeAddress(domain:.control,localID:"save-checkpoint"),evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate-validation")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:save,payload:.saveModel(config:["format":.string("checkpoint")]))),lease:owner.lease()))
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:evaluate,payload:.evaluateTest(config:["settingsVersion":2,"partitionRole":.string("validation"),"device":.string("cpu")]))),lease:owner.lease()))
        for target in [save,evaluate] {
            XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:WorkflowExamples.trainAddress,port:"model"),to:.init(address:target,port:"model")),lease:owner.lease()))
        }
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:.init(domain:.data,localID:"fit"),port:"validation"),to:.init(address:evaluate,port:"test")),lease:owner.lease()))
        let destination=root.appendingPathComponent("model.pt")
        XCTAssertTrue(host.modelDestinations.set(destination:destination,for:save,lease:owner.lease()),host.modelDestinations.statusMessage)
        let saved=expectation(description:"Exact model saved after training"),evaluated=expectation(description:"Validation evaluated after training")
        let saveObservation=host.$modelSaveReceipts.filter {!$0.isEmpty}.prefix(1).sink {_ in saved.fulfill()}
        let evaluationObservation=host.$evaluationOutcomes.filter {!$0.isEmpty}.prefix(1).sink {_ in evaluated.fulfill()}
        defer {saveObservation.cancel();evaluationObservation.cancel()}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        let flowIdentity=host.operationIdentity
        await fulfillment(of:[saved,evaluated],timeout:90)
        XCTAssertEqual(host.operationIdentity,flowIdentity,"Automatic Evaluate/Save must preserve their owning Train job")
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.completed,outcome.error ?? host.statusMessage)
        XCTAssertEqual(host.evaluationOutcomes.first?.outcome.status,.completed,host.statusMessage)
        XCTAssertEqual(host.evaluation?.totalCount,16)
        XCTAssertEqual(try Data(contentsOf:destination),try Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/checkpoint.pt")))
        XCTAssertEqual(host.modelSaveReceipts.first?.reference.runID,outcome.runID.uuidString)
        let reference=try XCTUnwrap(host.frozenBindings.reference(for:WorkflowExamples.trainAddress))
        XCTAssertEqual(reference.runID,outcome.runID.uuidString)
        // Training and preparation-only flows must expose the same full captured tables.
        let partition=WorkflowNodeAddress(domain:.data,localID:"split")
        XCTAssertTrue(owner.select(.init(nodeAddresses:[partition]),lease:owner.lease()))
        let viewer=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:partition,port:"training")),"Completed Train must expose Partition's captured table, not hide it behind a preparation-only provider")
        XCTAssertTrue(viewer.load())
        let deadline=Date().addingTimeInterval(20)
        while viewer.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        XCTAssertNotNil(viewer.page,viewer.statusMessage)
        XCTAssertEqual(viewer.page?.totalRows,32)
        XCTAssertEqual(viewer.page?.rows.count,32)
        XCTAssertNotNil(host.capturedTableViewSession(for:.init(address:partition,port:"validation")))
        XCTAssertNil(host.capturedTableViewSession(for:.init(address:WorkflowExamples.sourceAddress,port:"table")),"An unselected source cannot acquire a table query")
        XCTAssertTrue(host.canvas.apply(.move(addresses:[partition],offset:.init(x:100,y:200)),lease:owner.lease()))
        XCTAssertTrue(host.capturedTableViewSession(for:.init(address:partition,port:"training")) === viewer)
        let edge=try XCTUnwrap(owner.document.edges.first {$0.from == .init(address:partition,port:"training")})
        XCTAssertTrue(owner.select(.init(edgeIDs:[edge.id]),lease:owner.lease()))
        XCTAssertTrue(host.capturedTableViewSession(for:edge.from) === viewer,"Connection inspection uses the same captured table")
        for (id,port,count) in [("observations","table",64),("cast","table",64),("split","test",16),("fit","training",32),("fit","validation",16),("training","table",32),("validation","table",16)] {
            let address=WorkflowNodeAddress(domain:.data,localID:id)
            XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
            let table=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:address,port:port)),"Missing \(id).\(port)")
            XCTAssertTrue(table.load())
            let end=Date().addingTimeInterval(20)
            while table.isBusy && Date()<end {try await Task.sleep(for:.milliseconds(20))}
            XCTAssertEqual(table.page?.totalRows,count,table.statusMessage)
            XCTAssertEqual(table.page?.rows.count,count,table.statusMessage)
        }
        XCTAssertTrue(owner.select(.init(nodeAddresses:[partition]),lease:owner.lease()))
        // A private source replacement invalidates the current Train output.
        let source=try XCTUnwrap(host.bindings.path(for:WorkflowExamples.sourceAddress)),copy=root.appendingPathComponent("replacement.csv")
        try FileManager.default.copyItem(at:URL(fileURLWithPath:source),to:copy)
        XCTAssertTrue(host.bindings.setPath(copy.path,for:WorkflowExamples.sourceAddress,lease:owner.lease(),expectedRevision:host.bindings.revision))
        XCTAssertNil(host.frozenBindings.reference(for:WorkflowExamples.trainAddress))
        XCTAssertNil(host.capturedTableViewSession(for:.init(address:partition,port:"training")))
        XCTAssertFalse(viewer.load(),"A retained table query cannot read after source-binding replacement")
        XCTAssertTrue(FileManager.default.fileExists(atPath:destination.path),"Invalidation never removes user exports")
    }
}
