import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowToolbarAdmissionTests:XCTestCase {
    private func idle(_ host:WorkflowWorkspaceSession)async throws {
        let deadline=Date().addingTimeInterval(60)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(40))}
        XCTAssertFalse(host.hasActiveOperation,host.statusMessage)
    }
    func testSelectedOutputRebindRefusesWithoutPredictingAndShowsCurrentGuidance()async throws {try await exerciseSelectedOutput(keepUnrelatedFit:false)}
    func testSelectedOutputUsesOnlyCurrentConnectedPredictionDespiteStaleSibling()async throws {try await exerciseSelectedOutput(keepUnrelatedFit:true)}
    private func exerciseSelectedOutput(keepUnrelatedFit:Bool)async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else{throw XCTSkip("Select qualified sklearn interpreter")}
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(PrebuiltModelsPractice.lessonID,device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession),owner=host.owner
        defer{host.cancel();if !workspace.isBusy{try? FileManager.default.removeItem(at:root)}}
        workspace.sklearnRuntimeCapability=try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        host.executionLaunch={_,args,dir,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,dir,line,exit)}
        XCTAssertTrue(host.run(train:PrebuiltModelsPractice.estimator));try await idle(host)
        XCTAssertEqual(host.backendFit?.phase,.completed)
        let save=WorkflowNodeAddress(domain:.control,localID:"save"),package=root.appendingPathComponent("model")
        XCTAssertTrue(host.modelDestinations.set(destination:package,for:save,lease:owner.lease()))
        XCTAssertTrue(host.saveModel(address:save));try await idle(host)
        let load=WorkflowNodeAddress(domain:.control,localID:"reload"),predict=WorkflowNodeAddress(domain:.control,localID:"predict"),output=WorkflowNodeAddress(domain:.data,localID:"export")
        var config=PrebuiltModelConfiguration.defaults("kmeans")
        config["featureColumns"] = .array([.string("usage"),.string("recency"),.string("breadth")]);config["operation"] = .string("predict")
        XCTAssertNotNil(owner.applyBatch([
             .delete(.init(nodeAddresses:keepUnrelatedFit ? []:[PrebuiltModelsPractice.estimator,save,.init(domain:.data,localID:"assignments"),.init(domain:.data,localID:"profiles")])),
            .insert(.init(address:load,payload:.frozenModel(config:LoadModelConfiguration(source:.documentPackage).values))),
            .insert(.init(address:predict,payload:.estimator(config:config))),
            .connect(from:.init(address:load,port:"model"),to:.init(address:predict,port:"model")),
            .connect(from:.init(address:.init(domain:.data,localID:"numeric"),port:"table"),to:.init(address:predict,port:"table"))
        ],lease:owner.lease()))
        try await host.modelLoader.importDocumentClassifierState(package,address:load,policy:workspace.executionPolicy(for:owner.projectID))
        XCTAssertTrue(host.run(train:predict));try await idle(host)
        let previous=try XCTUnwrap(host.backendFit)
        XCTAssertEqual(previous.phase,.completed)
        XCTAssertNotNil(owner.applyBatch([
            .insert(.init(address:output,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string("result.csv"),"separator":.string(","),"columns":.array([]),"overwrite":false]))),
            .connect(from:.init(address:predict,port:"predictions"),to:.init(address:output,port:"table"))
        ],lease:owner.lease()))
        host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[output]),extending:false)
        host.outputDestinations.pickDirectory={root}
        XCTAssertTrue(host.outputDestinations.chooseDestination(for:output,lease:owner.lease()))
        let source=try XCTUnwrap(host.bindings.path(for:PrebuiltModelsPractice.source)),next=root.appendingPathComponent("next-period.csv")
        try FileManager.default.copyItem(at:URL(fileURLWithPath:source),to:next)
        XCTAssertTrue(host.bindings.setPath(next.path,for:PrebuiltModelsPractice.source,lease:owner.lease(),expectedRevision:host.bindings.revision))
        if keepUnrelatedFit {
            // Retain a second, stale captured producer plus an ungranted sibling write.
            let denied=WorkflowNodeAddress(domain:.data,localID:"ungranted-sibling")
            XCTAssertNotNil(owner.applyBatch([
                .insert(.init(address:denied,payload:.dataOperator(operatorID:"kwip.data.outputDelimited",config:["filename":.string("forbidden.csv"),"separator":.string(","),"columns":.array([]),"overwrite":false]))),
                .connect(from:.init(address:PrebuiltModelsPractice.estimator,port:"predictions"),to:.init(address:denied,port:"table"))
            ],lease:owner.lease()))
            XCTAssertTrue(host.run(train:predict));try await idle(host)
            let current=try XCTUnwrap(host.backendFit)
            XCTAssertEqual(current.phase,.completed,current.status)
            XCTAssertFalse(host.predictionOutputIsCurrent(.init(address:PrebuiltModelsPractice.estimator,port:"predictions")))
            host.canvas.selectNode(try XCTUnwrap(host.canvas.projection.displayIDByAddress[output]),extending:false)
            workspace.run();try await idle(host)
            XCTAssertTrue(host.backendFit === current,"Selected output cannot rerun any producer")
            XCTAssertTrue(FileManager.default.fileExists(atPath:root.appendingPathComponent("result.csv").path),host.displayedStatusMessage)
            XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("forbidden.csv").path))
            XCTAssertEqual(host.scopedPreparation?.phase,.completed,host.displayedStatusMessage)
            return
        }
        workspace.run()
        XCTAssertFalse(host.hasActiveOperation)
        XCTAssertTrue(host.backendFit === previous,"Selected output must not implicitly rerun inference")
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("result.csv").path))
        XCTAssertEqual(host.displayedStatusMessage,host.statusMessage,"Latest admission refusal must win over retained Completed")
        XCTAssertTrue(host.displayedStatusMessage.contains("Prebuilt"),host.displayedStatusMessage)
        XCTAssertTrue(host.displayedStatusMessage.contains("Assign new rows"),host.displayedStatusMessage)
        XCTAssertEqual(previous.phase,.completed,"Keep completed evidence intact")
    }
}
