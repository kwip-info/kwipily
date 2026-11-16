import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class PrebuiltModelsIntegrationTests:XCTestCase {
    func testRealSegmentationTablesSaveReloadAndColdHistory()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else {throw XCTSkip("Select pinned interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let checkpoint=ProcessInfo.processInfo.environment["KWIPML_PREBUILT_CHECKPOINT_STORAGE"]
        let root=try checkpoint.map{URL(fileURLWithPath:$0)} ?? canonicalBackendTemporaryDirectory()
        defer{if checkpoint==nil {try? FileManager.default.removeItem(at:root)}}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(PrebuiltModelsPractice.lessonID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        workspace.sklearnRuntimeCapability=try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        var backendStatuses:[String]=[]
        host.executionLaunch={_,args,dir,line,exit in
            if host.backendFit?.phase == .training {backendStatuses.append(host.backendFit?.status ?? "")}
            return try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,dir,line,exit)
        }
        XCTAssertTrue(host.run(train:PrebuiltModelsPractice.estimator),host.statusMessage)
        let deadline=Date().addingTimeInterval(100)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let fit=try XCTUnwrap(host.backendFit)
        XCTAssertEqual(fit.phase,.completed,fit.status+"\n"+fit.messages.joined(separator:"\n"));guard fit.phase == .completed else{return}
        for (port,count) in [("predictions",80),("profiles",4)] {
            let e=WorkflowEndpoint(address:PrebuiltModelsPractice.estimator,port:port)
            XCTAssertTrue(host.predictionOutputIsCurrent(e));XCTAssertEqual(fit.predictionMetadata(for:e)?["rowCount"],.integer(Int64(count)))
            XCTAssertTrue(host.owner.select(.init(nodeAddresses:[PrebuiltModelsPractice.estimator]),lease:host.owner.lease()))
            let query=try XCTUnwrap(host.predictionTableViewSession(for:e));XCTAssertTrue(query.load(),query.statusMessage)
            while query.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(40))}
            XCTAssertEqual(query.page?.totalRows,count,query.statusMessage);query.invalidate()
        }
        let copied=root.appendingPathComponent("impossible-metrics-"+UUID().uuidString)
        let accepted=try await fit.verifiedOutput();try FileManager.default.copyItem(at:accepted,to:copied)
        var invalid=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:copied.appendingPathComponent("manifest.json")))
        if case .object(var metrics)=invalid["evaluation"] {metrics["noiseRows"]=100001;invalid["evaluation"] = .object(metrics)}
        try WorkflowArchiveCodec.encode(invalid).write(to:copied.appendingPathComponent("manifest.json"))
        let captured=try SklearnPreparedWorkerRequest.decode(try XCTUnwrap(host.backendHistory.entries.first).workerRequest)
        XCTAssertThrowsError(try SklearnWorkflowResult.verify(directory:copied,request:captured))
        let save=WorkflowNodeAddress(domain:.control,localID:"save"),destination=root.appendingPathComponent("prebuilt-export-"+UUID().uuidString)
        XCTAssertTrue(host.modelDestinations.set(destination:destination,for:save,lease:host.owner.lease()))
        XCTAssertTrue(host.saveModel(address:save),host.statusMessage)
        while host.isManagingTrainedArtifacts && Date()<deadline {try await Task.sleep(for:.milliseconds(40))}
        XCTAssertTrue(FileManager.default.fileExists(atPath:destination.appendingPathComponent("state.json").path),host.statusMessage)
        let load=WorkflowNodeAddress(domain:.control,localID:"reload"),predict=WorkflowNodeAddress(domain:.control,localID:"assign-new")
        var predictConfig=PrebuiltModelConfiguration.defaults("kmeans");predictConfig["featureColumns"] = .array([.string("usage"),.string("recency"),.string("breadth")]);predictConfig["operation"] = .string("predict")
        XCTAssertNotNil(host.owner.applyBatch([.insert(.init(address:load,name:"Reload frozen segments",x:20,y:700,payload:.frozenModel(config:LoadModelConfiguration(source:.documentPackage).values))),.insert(.init(address:predict,name:"Assign new prepared rows",x:700,y:700,payload:.estimator(config:predictConfig))),.connect(from:.init(address:load,port:"model"),to:.init(address:predict,port:"model")),.connect(from:.init(address:.init(domain:.data,localID:"numeric"),port:"table"),to:.init(address:predict,port:"table"))],lease:host.owner.lease()))
        try await host.modelLoader.importDocumentClassifierState(destination,address:load,policy:workspace.executionPolicy(for:host.owner.projectID))
        XCTAssertEqual(host.modelLoader.reference(load)?.metadata["task"],.string("prebuilt"))
        XCTAssertTrue(host.run(train:predict),host.statusMessage)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertEqual(backendStatuses,["Fitting sklearn estimator…","Predicting with frozen sklearn estimator…"],"Frozen inference must never be presented as a refit")
        let displayID=try XCTUnwrap(host.canvas.projection.displayIDByAddress[predict])
        let displayed=try XCTUnwrap(host.canvas.graph.nodes.first {$0.id==displayID})
        XCTAssertEqual(host.canvas.canvasSummary(displayed),"Predict with a frozen model")
        let reused=try XCTUnwrap(host.backendFit);XCTAssertEqual(reused.phase,.completed,reused.status+"\n"+reused.messages.joined(separator:"\n"))
        if case .object(let metrics)=reused.manifest?["evaluation"] {XCTAssertEqual(metrics["count"],80);XCTAssertNil(metrics["accuracy"])}else{XCTFail("Missing frozen inference metrics")}
        XCTAssertEqual(reused.manifest?["schemaVersion"],2)
        if case .object(let provenance)=reused.manifest?["provenance"] {
            XCTAssertEqual(provenance["inputSamples"],80)
            XCTAssertNil(provenance["trainingSamples"]);XCTAssertNil(provenance["validationSamples"])
        } else {XCTFail("Missing frozen input provenance")}
        let frozenOutput=try await reused.verifiedOutput()
        let frozenRequest=try SklearnPreparedWorkerRequest.decode(try XCTUnwrap(host.backendHistory.entries.first).workerRequest)
        let forged=root.appendingPathComponent("forged-input-roles")
        try FileManager.default.copyItem(at:frozenOutput,to:forged)
        var forgedManifest=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:forged.appendingPathComponent("manifest.json")))
        if case .object(var provenance)=forgedManifest["provenance"] {provenance["trainingSamples"]=80;forgedManifest["provenance"] = .object(provenance)}
        try WorkflowArchiveCodec.encode(forgedManifest).write(to:forged.appendingPathComponent("manifest.json"))
        XCTAssertThrowsError(try SklearnWorkflowResult.verify(directory:forged,request:frozenRequest))
        // A copied legacy-format record remains readable; never mutate retained History.
        var oldManifest=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:frozenOutput.appendingPathComponent("manifest.json")))
        var oldPopulation=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:frozenOutput.appendingPathComponent("population.json")))
        oldPopulation["trainingRowIDs"]=oldPopulation["inputRowIDs"];oldPopulation["validationRowIDs"]=oldPopulation["inputRowIDs"]
        let oldBytes=try WorkflowArchiveCodec.encode(oldPopulation),oldPin=WorkflowArchiveCodec.hash(oldBytes)
        try oldBytes.write(to:forged.appendingPathComponent("population.json"))
        oldManifest["schemaVersion"]=1
        if case .object(var provenance)=oldManifest["provenance"] {
            provenance.removeValue(forKey:"inputSamples");provenance["trainingSamples"]=80;provenance["validationSamples"]=80;provenance["populationSHA256"] = .string(oldPin)
            provenance["preparation"] = .object(["manifestSHA256":.string(frozenRequest.prepared.manifestSHA256),"graphDigest":.string(frozenRequest.prepared.graphDigest),"trainingNode":.string(frozenRequest.prepared.trainingNode),"trainingPort":.string(frozenRequest.prepared.trainingPort),"validationNode":.string(frozenRequest.prepared.validationNode),"validationPort":.string(frozenRequest.prepared.validationPort)])
            oldManifest["provenance"] = .object(provenance)
        }
        if case .array(let entries)=oldManifest["artifacts"] {oldManifest["artifacts"] = .array(entries.map {entry in
            guard case .object(var file)=entry,file["path"] == .string("population.json") else{return entry}
            file["bytes"] = .integer(Int64(oldBytes.count));file["sha256"] = .string(oldPin);return .object(file)
        })}
        try WorkflowArchiveCodec.encode(oldManifest).write(to:forged.appendingPathComponent("manifest.json"))
        XCTAssertNoThrow(try SklearnWorkflowResult.verify(directory:forged,request:frozenRequest))
        let restored=BackendFitHistoryStore(root:root,projectID:host.owner.projectID);await restored.reloadAndWait();XCTAssertEqual(restored.entries.count,2,restored.status)
        _=try await restored.verify(try XCTUnwrap(restored.entries.first))
        XCTAssertTrue(workspace.saveManagedProject(),workspace.statusMessage)
        if checkpoint != nil {print("PREBUILT_CHECKPOINT_PROJECT="+host.owner.projectID)}
    }
    func testRealSupervisedClassificationAndRegressionFamilies()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else{throw XCTSkip("Select pinned interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=try canonicalBackendTemporaryDirectory();defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadBackendWorkflowExample(backendID:"sklearn"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),address=WorkflowExamples.sklearnEstimatorAddress
        workspace.sklearnRuntimeCapability=try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        host.executionLaunch={_,args,dir,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,dir,line,exit)}
        for algorithm in ["logistic","randomForestClassifier","ridge","elasticNet","randomForestRegressor"] {
            var config=PrebuiltModelConfiguration.defaults(algorithm);config["featureColumns"] = .array([.string("signal"),.string("context")]);config["targetColumn"] = .string("label")
            XCTAssertNotNil(host.owner.apply(.configure(address:address,config:config),lease:host.owner.lease()))
            XCTAssertTrue(host.run(train:address),host.statusMessage)
            let deadline=Date().addingTimeInterval(60)
            while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
            let fit=try XCTUnwrap(host.backendFit)
            XCTAssertEqual(fit.phase,.completed,algorithm+" · "+fit.status+"\n"+fit.messages.joined(separator:"\n"))
            if case .object(let metrics)=fit.manifest?["evaluation"] {
                XCTAssertEqual(metrics["count"],16)
                if PrebuiltModelConfiguration.algorithms[algorithm]?.task=="regression" {XCTAssertNotNil(metrics["rmse"]);XCTAssertNil(metrics["accuracy"])}else{XCTAssertNotNil(metrics["accuracy"])}
            }else{XCTFail("Missing task-specific metrics")}
        }
    }

}
