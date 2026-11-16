import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class OCRClassificationWorkflowTests:XCTestCase {
    func testRealScannedDocumentsFitSaveReloadPredictAndHistory()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else {throw XCTSkip("Select the pinned sklearn interpreter.")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        guard let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_OCR_CLASSIFICATION"],let pack=ProcessInfo.processInfo.environment["KWIPML_TEST_OCR_PACK"] else {throw XCTSkip("Explicit OCR classifier fixture required.")}
        var document=try DocumentClassificationPracticeExample.document()
        for i in document.nodes.indices {
            switch document.nodes[i].payload {
            case .dataOperator(let id,var c):
                if document.nodes[i].address==DocumentClassificationPracticeExample.source {c["extensions"] = .array([.string("png")])}
                if document.nodes[i].address==DocumentClassificationPracticeExample.features {c["textColumn"] = .string("document_text")}
                document.nodes[i].payload = .dataOperator(operatorID:id,config:c)
            case .estimator(var c):c["featureColumns"] = .array([.string("document_text")]);document.nodes[i].payload = .estimator(config:c)
            default:break
            }
        }
        let ocrAddress=WorkflowNodeAddress(domain:.data,localID:"scan-ocr")
        document.nodes.append(.init(address:ocrAddress,name:"Read scan text",x:200,y:90,payload:.dataOperator(operatorID:OCRConfiguration.operatorID,config:try OCRConfiguration.defaults())))
        for i in document.edges.indices where document.edges[i].from.address==DocumentClassificationPracticeExample.source {document.edges[i].from = .init(address:ocrAddress,port:"documents")}
        document.edges.append(.init(id:"source-ocr",from:.init(address:DocumentClassificationPracticeExample.source,port:"table"),to:.init(address:ocrAddress,port:"table"),kind:.table))
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let copy=root.appendingPathComponent("scan-fixture");try FileManager.default.copyItem(at:URL(fileURLWithPath:fixture),to:copy)
        let session=try XCTUnwrap(workspace.workflowSession)
        for (address,leaf) in [(DocumentClassificationPracticeExample.source,"scans"),(DocumentClassificationPracticeExample.labels,"labels.csv"),(DocumentClassificationPracticeExample.unseen,"unseen.csv")] {
            XCTAssertTrue(session.bindings.setPath(copy.appendingPathComponent(leaf).path,for:address,lease:session.owner.lease(),expectedRevision:session.bindings.revision),session.bindings.error ?? "")
        }
        let packs=OCRPackStore.directory(workspace:workspace.workspaceDirectory)
        try FileManager.default.createDirectory(at:packs,withIntermediateDirectories:true,attributes:[.posixPermissions:0o700])
        for asset in OCRPackCatalog.assets {let bytes=try Data(contentsOf:URL(fileURLWithPath:pack).appendingPathComponent(asset.name.replacingOccurrences(of:".zip",with:".pth.zip")));let target=packs.appendingPathComponent(asset.name);try bytes.write(to:target,options:.withoutOverwriting);try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:target.path)}

        let host=try XCTUnwrap(workspace.workflowSession)
        workspace.sklearnRuntimeCapability=try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        host.executionLaunch = {_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
        XCTAssertTrue(host.run(train:DocumentClassificationPracticeExample.estimator),host.statusMessage)
        let deadline=Date().addingTimeInterval(240)
        while host.backendFit?.isRunning == true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let fit=try XCTUnwrap(host.backendFit)
        XCTAssertEqual(fit.phase,.completed,fit.status+"\n"+fit.messages.joined(separator:"\n"))
        guard fit.phase == .completed else {return}
        let fitEndpoint=WorkflowEndpoint(address:DocumentClassificationPracticeExample.estimator,port:"predictions")
        XCTAssertNotNil(fit.predictionArtifact(for:fitEndpoint))
        XCTAssertEqual(fit.predictionMetadata(for:fitEndpoint)?["rowCount"],9)
        XCTAssertEqual(fit.predictionMetadata(for:fitEndpoint)?["partition"],.string("validation"))
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[DocumentClassificationPracticeExample.estimator]),lease:host.owner.lease()))
        while (workspace.isBusy || host.owner.isBusy) && Date()<deadline {try await Task.sleep(for:.milliseconds(30))}
        let fitQuery=try XCTUnwrap(host.predictionTableViewSession(for:fitEndpoint))
        XCTAssertTrue(fitQuery.load(),fitQuery.statusMessage)
        while fitQuery.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(30))}
        XCTAssertEqual(fitQuery.page?.totalRows,9,fitQuery.statusMessage)
        XCTAssertEqual(fitQuery.page?.rows.count,9)
        fitQuery.invalidate()
        let acceptedFit=try await fit.verifiedOutput();XCTAssertTrue(FileManager.default.fileExists(atPath:acceptedFit.path))
        XCTAssertEqual(host.backendHistory.entries.count,1,host.backendHistory.status)
        let save=WorkflowNodeAddress(domain:.control,localID:"save"),destination=root.appendingPathComponent("exported-document-classifier")
        XCTAssertTrue(host.modelDestinations.set(destination:destination,for:save,lease:host.owner.lease()),host.modelDestinations.statusMessage)
        XCTAssertTrue(host.saveModel(address:save),host.statusMessage)
        while host.isManagingTrainedArtifacts && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertTrue(FileManager.default.fileExists(atPath:destination.appendingPathComponent("state.json").path),host.statusMessage)
        guard FileManager.default.fileExists(atPath:destination.appendingPathComponent("state.json").path) else {return}
        let load=WorkflowNodeAddress(domain:.control,localID:"load")
        try await host.modelLoader.importDocumentClassifierState(destination,address:load,policy:workspace.executionPolicy(for:host.owner.projectID))
        let reference=try XCTUnwrap(host.modelLoader.reference(load))
        XCTAssertEqual(reference.files.map(\.path),["state.json"],"Never import executable package files")
        // A loaded processor must not depend on the mutable original training folder.
        let originals=try XCTUnwrap(host.bindings.path(for:DocumentClassificationPracticeExample.source))
        try FileManager.default.moveItem(at:URL(fileURLWithPath:originals),to:root.appendingPathComponent("training-source-held-aside"))
        XCTAssertTrue(host.run(train:DocumentClassificationPracticeExample.predict),host.statusMessage)
        while host.backendFit?.isRunning == true && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let predicted=try XCTUnwrap(host.backendFit)
        XCTAssertEqual(predicted.phase,.completed,predicted.status+"\n"+predicted.messages.joined(separator:"\n"))
        guard predicted.phase == .completed else {return}
        let endpoint=WorkflowEndpoint(address:DocumentClassificationPracticeExample.predict,port:"predictions")
        XCTAssertNotNil(predicted.predictionArtifact(for:endpoint));XCTAssertTrue(host.predictionOutputIsCurrent(endpoint))
        let verified=try await predicted.verifiedOutput()
        let bytes=try Data(contentsOf:verified.appendingPathComponent("predictions.json"))
        let table=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        guard case .array(let rows)=table["rows"] else {return XCTFail("Missing complete typed output")}
        XCTAssertEqual(rows.count,3)
        let restored=BackendFitHistoryStore(root:root,projectID:host.owner.projectID);await restored.reloadAndWait()
        XCTAssertEqual(restored.entries.count,2,restored.status)
        XCTAssertEqual(Set(restored.entries.compactMap {$0.evaluation["operation"]?.stringValue}),Set(["fit","predict"]))
        // Completion schedules the downstream View preparation after releasing the fit.
        while (workspace.isBusy || host.owner.isBusy) && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertFalse(host.owner.isBusy,host.statusMessage);XCTAssertFalse(workspace.isBusy,host.statusMessage)
        let second=root.appendingPathComponent("resaved-after-prediction")
        XCTAssertTrue(host.modelDestinations.set(destination:second,for:save,lease:host.owner.lease()))
        XCTAssertTrue(host.saveModel(address:save),host.statusMessage)
        while host.isManagingTrainedArtifacts && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertTrue(FileManager.default.fileExists(atPath:second.appendingPathComponent("state.json").path),host.statusMessage)
        XCTAssertEqual(try Data(contentsOf:destination.appendingPathComponent("state.json")),try Data(contentsOf:second.appendingPathComponent("state.json")))
    }
}
