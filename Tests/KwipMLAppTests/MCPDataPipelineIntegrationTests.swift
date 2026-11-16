import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class MCPDataPipelineIntegrationTests:XCTestCase {
    private final class Secrets:MCPSecretStorage {
        var values:[String:String]=[:]
        func read(clientID:String)throws->String? {values[clientID]}
        func write(_ token:String,clientID:String)throws {values[clientID]=token}
        func remove(clientID:String)throws {values[clientID]=nil}
    }
    private func wait(_ condition:()->Bool)async throws {
        let deadline=Date().addingTimeInterval(30)
        while Date()<deadline {if condition(){return};try await Task.sleep(for:.milliseconds(20))}
        throw WorkflowArchiveError("Timed out waiting for actual captured table work.")
    }
    func testMCPRunsOneSnapshotAndPagesWithoutOriginalSourceThenHonorsRevocation()async throws {try await exercise(wideRow:false)}
    func testAnOversizedClientRowIsRefusedInsteadOfReturningANonadvancingEmptyPage()async throws {try await exercise(wideRow:true)}
    func testEstimatorJobReportsItsRealTrainingPhaseAndThenCompletes() async throws {try await estimatorExercise(includeOutput:false)}
    func testExplicitEstimatorStopsAtCapturedPredictionsThenExplicitOutputWritesWithoutRefitting() async throws {try await estimatorExercise(includeOutput:true)}
    private func estimatorExercise(includeOutput:Bool) async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],
              let python=ProcessInfo.processInfo.environment["KWIPML_SKLEARN_TEST_PYTHON"] else {
            throw XCTSkip("Needs the actual development runtime and pinned sklearn interpreter")
        }
        _=NSApplication.shared
        let root=try canonicalBackendTemporaryDirectory()
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        var document=try WorkflowTemplates.instantiate(id:"complaint-routing")
        let retained=Set(["csv","columns","split","features","classifier"] + (includeOutput ? ["predictions","validation_csv"]:[]))
        document.nodes.removeAll {!retained.contains($0.address.localID)}
        document.edges.removeAll {!retained.contains($0.from.address.localID) || !retained.contains($0.to.address.localID)}
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        let record=try library.saveSource(.workflow(document),id:document.projectID)
        let windows=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)})
        let window=try windows.open(record,present:false),workspace=window.session.workspace
        let host=try XCTUnwrap(workspace.workflowSession)
        let controller=MCPController(directory:root,secrets:Secrets(),startConfigured:false)
        var pendingExit:(()->Void)?
        defer {
            pendingExit?();host.cancel();controller.stop(persist:false)
            for window in windows.openControllers {_=windows.close(window)}
            if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}
        }
        await verifyRuntimeForIntegration(workspace)
        workspace.sklearnRuntimeCapability=try .decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected))
        let csv=root.appendingPathComponent("synthetic-documents.csv")
        var content="ComplaintID,ReceivedDate,Narrative,Product\n"
        for group in ["account","card","mortgage"] {
            for row in 0..<60 {content += "\(group)-\(row),2025-01-01,\(group) example request number \(row),\(group)\n"}
        }
        try Data(content.utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:.init(domain:.data,localID:"csv"),lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        // Run the real owned workers. Hold only the final backend callback so the
        // native training phase can be observed deterministically, without a timer race.
        host.executionLaunch={_,arguments,directory,onLine,onExit in
            let isBackend=host.currentBackendFit?.phase == .training
            return try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,{code,message in
                if isBackend {pendingExit={onExit(code,message)}} else {onExit(code,message)}
            })
        }
        let outputAddress=WorkflowNodeAddress(domain:.data,localID:"validation_csv")
        if includeOutput {
            XCTAssertTrue(host.owner.select(.init(nodeAddresses:[outputAddress]),lease:host.owner.lease()))
            host.outputDestinations.pickDirectory={root}
            XCTAssertTrue(host.outputDestinations.chooseDestination(for:outputAddress,lease:host.owner.lease()))
        }
        let actions:Set<MCPAction>=[.runStart,.jobGet]
        var settings=MCPSettings.factory;settings.enabled=true
        if includeOutput {settings.disabledComponents=[]}
        settings.clients=[.init(id:"test",name:"Test",actions:actions)]
        settings.projects[document.projectID] = .init(actions:actions,readCapturedValues:false)
        try controller.save(settings)
        let gateway=MCPWorkflowGateway(controller:controller,windows:windows)
        let started=try await gateway.dispatch(clientID:"test",action:MCPAction.runStart.rawValue,arguments:[
            "projectID":.string(document.projectID),"requestID":.string(UUID().uuidString),
            "terminal":.object(["domain":.string("control"),"localID":.string("classifier")])])
        let jobID=try XCTUnwrap(started["jobID"]?.stringValue)
        try await wait {pendingExit != nil || !host.hasActiveOperation}
        guard pendingExit != nil else {
            return XCTFail(host.statusMessage + " / " + (host.currentBackendFit?.status ?? "No backend session"))
        }
        XCTAssertEqual(host.currentBackendFit?.phase,.training)
        XCTAssertTrue(host.hasActiveOperation)
        let args:[String:JSONValue]=["projectID":.string(document.projectID),"jobID":.string(jobID)]
        let running=try await gateway.dispatch(clientID:"test",action:MCPAction.jobGet.rawValue,arguments:args)
        guard case .object(let value)=running["job"] else {return XCTFail("Missing current job")}
        XCTAssertEqual(value["state"],.string("running"))
        XCTAssertEqual(value["phase"],.string("training"),"Estimator fitting must not be described as table preparation")
        pendingExit?();pendingExit=nil
        try await wait {!host.hasActiveOperation}
        let finished=try await gateway.dispatch(clientID:"test",action:MCPAction.jobGet.rawValue,arguments:args)
        guard case .object(let final)=finished["job"] else {return XCTFail("Missing completed job")}
        XCTAssertEqual(final["state"],.string("completed"),host.statusMessage)
        XCTAssertEqual(final["phase"],.string("completed"))
        if includeOutput {
            let output=try XCTUnwrap(host.owner.document.nodes.first{$0.address==outputAddress})
            guard case .dataOperator(_,var config)=output.payload,let filename=config["filename"]?.stringValue else{return XCTFail("Missing output configuration")}
            let path=root.appendingPathComponent(filename)
            XCTAssertFalse(FileManager.default.fileExists(atPath:path.path),"An explicit estimator terminal must not write a connected output")
            guard !FileManager.default.fileExists(atPath:path.path) else{return}
            let fitted=host.currentBackendFit?.runID
            XCTAssertNotNil(host.currentBackendFit?.predictionArtifact(for:.init(address:.init(domain:.control,localID:"classifier"),port:"predictions")))
            let exported=try await gateway.dispatch(clientID:"test",action:MCPAction.runStart.rawValue,arguments:["projectID":.string(document.projectID),"requestID":.string(UUID().uuidString),"terminal":.object(["domain":.string("data"),"localID":.string("validation_csv")])])
            XCTAssertNotNil(exported["jobID"])
            // Publication may finish just before its owned teardown releases the
            // shared reservation. Editing/native Run requires fully idle admission.
            try await wait {!host.hasActiveOperation && !workspace.isBusy && !host.owner.isBusy}
            XCTAssertTrue(FileManager.default.fileExists(atPath:path.path),host.statusMessage)
            XCTAssertEqual(host.currentBackendFit?.runID,fitted,"Explicit data export must reuse captures without another fit")
            // Native Run Flow keeps its intentional descendant preparation.
            config["filename"] = .string("native-whole-flow.csv")
            XCTAssertNotNil(host.owner.apply(.configure(address:outputAddress,config:config),lease:host.owner.lease()))
            host.executionLaunch={_,arguments,directory,onLine,onExit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),arguments,directory,onLine,onExit)}
            XCTAssertTrue(host.run(useSelection:false),host.statusMessage)
            try await wait {!host.hasActiveOperation && !workspace.isBusy && !host.owner.isBusy}
            XCTAssertTrue(FileManager.default.fileExists(atPath:root.appendingPathComponent("native-whole-flow.csv").path),host.statusMessage)
            XCTAssertNotEqual(host.currentBackendFit?.runID,fitted)
        }
    }
    private func exercise(wideRow:Bool)async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Needs the actual isolated development runtime")}
        _=NSApplication.shared
        let root=try canonicalBackendTemporaryDirectory();try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),partition=WorkflowNodeAddress(domain:.data,localID:"partition"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        var document=WorkflowDocument(projectID:UUID().uuidString,name:"MCP snapshot contract",nodes:[
            .init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:CSVParsingConfiguration.newDefaults)),
            .init(address:partition,payload:.dataOperator(operatorID:"kwip.data.partition",config:["strategy":.string("random"),"seed":42,"validationPercent":9,"testPercent":1])),
            .init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[
                .init(from:.init(address:source,port:"table"),to:.init(address:partition,port:"table"),kind:.table),
                .init(from:.init(address:partition,port:"training"),to:.init(address:view,port:"table"),kind:.table)])
        if wideRow {
            document.nodes.removeAll {$0.address==partition}
            document.edges=[.init(from:.init(address:source,port:"table"),to:.init(address:view,port:"table"),kind:.table)]
        }
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects")),record=try library.saveSource(.workflow(document),id:document.projectID)
        let windows=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)})
        let window=try windows.open(record,present:false),workspace=window.session.workspace,host=try XCTUnwrap(workspace.workflowSession)
        let controller=MCPController(directory:root,secrets:Secrets(),startConfigured:false)
        defer {host.cancel();controller.stop(persist:false);for window in windows.openControllers {_=windows.close(window)};if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        await verifyRuntimeForIntegration(workspace)
        let csv=root.appendingPathComponent("documents.csv"),padding=String(repeating:"x",count:2200)
        let content:String
        if wideRow {
            let header=(0..<128).map{"column_"+String($0)}.joined(separator:",")+"\n"
            let row=Array(repeating:padding,count:128).joined(separator:",")+"\n"
            content=header+Array(repeating:row,count:6).joined()
        } else {content="text,padding\n"+Array(repeating:"hello world,\(padding)\n",count:700).joined()}
        try Data(content.utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let actions:Set<MCPAction>=[.runStart,.jobGet,.resultsRead]
        var settings=MCPSettings.factory;settings.enabled=true;settings.clients=[.init(id:"test",name:"Test",actions:actions)];settings.projects[document.projectID] = .init(actions:actions,readCapturedValues:true)
        try controller.save(settings)
        let gateway=MCPWorkflowGateway(controller:controller,windows:windows)
        func call(_ action:MCPAction,_ extra:[String:JSONValue]=[:])async throws->[String:JSONValue] {
            var arguments=extra;arguments["projectID"] = .string(document.projectID)
            return try await gateway.dispatch(clientID:"test",action:action.rawValue,arguments:arguments)
        }
        let started=try await call(.runStart,["requestID":.string(UUID().uuidString)])
        let job=try XCTUnwrap(started["jobID"]?.stringValue)
        try await wait {!workspace.isBusy && [WorkflowExecutionPhase.completed,.failed,.cancelled].contains(host.currentDataPreparation?.phase ?? .idle)}
        guard host.currentDataPreparation?.phase == .completed else {return XCTFail(host.statusMessage+" / "+(host.currentDataPreparation?.statusMessage ?? "No preparation"))}
        let status=try await call(.jobGet,["jobID":.string(job)])
        guard case .object(let jobValue)=status["job"] else {return XCTFail("No shared job status")}
        XCTAssertEqual(jobValue["state"],.string("completed"))
        // A captured View is independent of the now unavailable original CSV.
        try FileManager.default.moveItem(at:csv,to:csv.appendingPathExtension("away"))
        let address=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(view))
        let query=try await call(.resultsRead,["address":address,"port":.string("table"),"includeValues":true,"page":true])
        let queryID=try XCTUnwrap(query["queryID"]?.stringValue)
        if wideRow {
            let deadline=Date().addingTimeInterval(30)
            repeat {
                do {
                    let page=try await call(.resultsRead,["address":address,"includeValues":true,"queryID":.string(queryID)])
                    if page["state"] == .string("completed") {XCTFail("A row above the client payload budget must not produce a completed empty page with unchanged offset");return}
                } catch let refusal as MCPSettings.Refusal {XCTAssertEqual(refusal.code,"table_row_exceeds_client_budget");return}
                try await Task.sleep(for:.milliseconds(20))
            } while Date()<deadline
            return XCTFail("Wide-row query did not finish")
        }
        var page:[String:JSONValue]=[:];let deadline=Date().addingTimeInterval(30)
        repeat {page=try await call(.resultsRead,["address":address,"includeValues":true,"queryID":.string(queryID)]);if page["state"] == .string("completed"){break};try await Task.sleep(for:.milliseconds(20))} while Date()<deadline
        XCTAssertEqual(page["state"],.string("completed"));XCTAssertEqual(page["totalRows"],630)
        guard case .array(let rows)=page["rows"] else{return XCTFail("No actual page rows")}
        XCTAssertFalse(rows.isEmpty);XCTAssertLessThan(rows.count,100)
        XCTAssertEqual(page["nextOffset"],.integer(Int64(rows.count)));XCTAssertEqual(page["truncated"],true)
        settings.projects[document.projectID]?.readCapturedValues=false;try controller.save(settings)
        do {_=try await call(.resultsRead,["address":address,"includeValues":true,"queryID":.string(queryID)]);XCTFail("Revoked values were returned")} catch {}
    }
}
