import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class MCPWorkflowGatewayTests:XCTestCase {
    final class Secrets:MCPSecretStorage {
        var values:[String:String]=[:]
        func read(clientID:String)throws->String?{values[clientID]}
        func write(_ token:String,clientID:String)throws{values[clientID]=token}
        func remove(clientID:String)throws{values[clientID]=nil}
    }
    struct Fixture {let controller:MCPController;let gateway:MCPWorkflowGateway;let windows:ProjectWindowRegistry;let owner:WorkflowAuthoringStore;let client:String}
    func fixture(document:WorkflowDocument?=nil)throws->Fixture {
        _=NSApplication.shared
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent("mcp-gateway-"+UUID().uuidString)
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let windows=ProjectWindowRegistry(presentsWindows:false,makeWorkspace:{WorkspaceStore(rootOverride:root,storageOverride:directory,detectCapabilities:false)})
        let window:ProjectWindowController
        if let document {let record=try windows.restorationWorkspace.projectLibrary.saveSource(.workflow(document));window=try windows.open(record,present:false)}
        else {window=try windows.newFlow(present:false)}
        let owner=try XCTUnwrap(window.session.workspace.workflowSession?.owner)
        let controller=MCPController(directory:directory,secrets:Secrets(),startConfigured:false)
        var settings=MCPSettings.factory;settings.enabled=true;settings.clients=[.init(id:"test",name:"Test",actions:Set(MCPAction.allCases))];settings.projects[owner.projectID] = .init(actions:Set(MCPAction.allCases),readCapturedValues:true)
        try controller.save(settings)
        addTeardownBlock {await MainActor.run {for window in windows.openControllers {_=windows.close(window)};controller.stop(persist:false);try? FileManager.default.removeItem(at:directory)}}
        return .init(controller:controller,gateway:.init(controller:controller,windows:windows),windows:windows,owner:owner,client:"test")
    }
    func call(_ fixture:Fixture,_ action:MCPAction,_ extra:[String:JSONValue]=[:])async throws->[String:JSONValue] {
        var args=extra;if action.requiresProject{args["projectID"] = .string(fixture.owner.projectID)}
        if [.projectCreate,.patchApply,.runStart].contains(action),args["requestID"]==nil{args["requestID"] = .string(UUID().uuidString)}
        return try await fixture.gateway.dispatch(clientID:fixture.client,action:action.rawValue,arguments:args)
    }
    func testCatalogAcceptsCanonicalDataOperatorWithoutChangingCreationIdentityOrPermissions()async throws {
        let f=try fixture(),operatorID="kwip.data.sourceCSV"
        let response=try await call(f,.catalogGet,["componentID":.string(operatorID)])
        guard case .array(let components)=response["components"],case .object(let item)=components.first else {return XCTFail("Missing catalog entry")}
        let creation=try XCTUnwrap(item["componentID"]?.stringValue)
        XCTAssertNotEqual(creation,operatorID)
        XCTAssertEqual(item["lookupIDs"],.array([.string(creation),.string(operatorID)]))
        let exact=try await call(f,.catalogGet,["componentID":.string(creation)])
        XCTAssertEqual(response,exact)
        var settings=f.controller.settings;settings.clients[0].disabledComponents.insert(operatorID);try f.controller.save(settings)
        for key in [operatorID,creation] {
            do {_=try await call(f,.catalogGet,["componentID":.string(key)]);XCTFail("Disabled component exposed")}
            catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"component_unavailable")}
        }
        XCTAssertTrue(f.owner.document.nodes.isEmpty)
    }
    func testTemplateIDsAreDiscoverableThroughConciseCatalog() async throws {
        let f=try fixture()
        let response=try await call(f,.catalogList,["readPath":.array([.string("templates")]),"readLimit":1])
        guard case .object(let page)=response["metadataRead"],
              case .array(let entries)=page["entries"],case .object(let entry)=entries.first,
              case .object(let template)=entry["value"] else {return XCTFail("Missing discoverable template metadata")}
        XCTAssertEqual(template["id"],.string(try XCTUnwrap(WorkflowTemplates.catalog().first).id))
        XCTAssertEqual(page["total"],.integer(Int64(try WorkflowTemplates.catalog().count)))
        XCTAssertNotNil(page["readRevision"])
        XCTAssertTrue(f.owner.document.nodes.isEmpty)
    }
    func testTemplateCreationRespectsComponentPolicyAndDoesNotReplaceCanvas() async throws {
        let f=try fixture()
        var policy=f.controller.settings
        policy.disabledComponents=[]
        policy.clients[0].actions.insert(.projectCreate)
        try f.controller.save(policy)
        let original=f.owner.document
        let created=try await call(f,.projectCreate,["templateID":.string("sales-exceptions"),"requestID":.string(UUID().uuidString)])
        let id=try XCTUnwrap(created["projectID"]?.stringValue)
        XCTAssertNotEqual(id,original.projectID)
        XCTAssertEqual(f.owner.document,original)
        let library=ProjectLibrary(directory:f.controller.directory.appendingPathComponent("projects"))
        guard case .workflow(let source)=try library.loadSource(id:id) else {return XCTFail("Expected workflow")}
        XCTAssertEqual(source.templateOrigin?.id,"sales-exceptions")
        XCTAssertFalse(source.nodes.isEmpty)
        policy=f.controller.settings;policy.disabledComponents.insert("kwip.data.outputDelimited");try f.controller.save(policy)
        let before=try library.listSources().count
        do {
            _=try await call(f,.projectCreate,["templateID":.string("sales-exceptions"),"requestID":.string(UUID().uuidString)])
            XCTFail("Disabled output must refuse before writing a project")
        } catch {}
        XCTAssertEqual(try library.listSources().count,before)
    }
    func testJoinProjectionPairRefusesPreviewBeforeExecutionWithoutMutation()async throws {
        let address=WorkflowNodeAddress(domain:.data,localID:"join")
        let config:[String:JSONValue] = ["leftKeys":.array(["region","date"].map(JSONValue.string)),"rightKeys":.array(["region","date"].map(JSONValue.string)),"mode":.string("left"),"nullPolicy":.string("neverMatch"),"cardinality":.string("manyToOne"),"leftPrefix":.string("left_"),"rightPrefix":.string("right_"),"maxRows":1000]
        let f=try fixture(document:WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,payload:.dataOperator(operatorID:"kwip.data.join",config:config))],edges:[]))
        var invalid=config;invalid["newColumns"] = .string("keep")
        let before=f.owner.document
        let result=try await call(f,.patchPreview,["revision":.integer(Int64(before.revision)),"operations":.array([.object(["operation":.string("configure"),"address":.object(["domain":.string("data"),"localID":.string("join")]),"configuration":.object(invalid)])])])
        XCTAssertEqual(result["code"],.string("invalid_configuration"));XCTAssertEqual(result["diagnosticCode"],.string("join.projectionPair"));XCTAssertNil(result["proposalID"]);XCTAssertEqual(f.owner.document,before)
    }
    func testAPIDiscoveryAuthorsHolidayMappingAndRefusesSecretsWithoutEcho()async throws {
        let f=try fixture()
        let catalog=try await call(f,.catalogGet,["componentID":.string("kwip.data.sourceCSV")])
        guard case .array(let entries)=catalog["components"],case .object(let entry)=entries.first,
              case .array(let variants)=entry["configurationVariants"],case .object(let variant)=variants.first,
              case .object(let example)=variant["exampleConfiguration"] else {return XCTFail("Missing discoverable API configuration variant")}
        let component=try XCTUnwrap(entry["componentID"]),before=f.owner.document
        func preview(_ config:[String:JSONValue])async throws->[String:JSONValue] {
            try await call(f,.patchPreview,["revision":.integer(Int64(before.revision)),"operations":.array([
                .object(["operation":.string("insert"),"componentID":component,"nodeID":.string("holiday-api"),"x":0,"y":0]),
                .object(["operation":.string("configure"),"address":.object(["domain":.string("data"),"localID":.string("holiday-api")]),"configuration":.object(config)])])])
        }
        let accepted=try await preview(example);XCTAssertNotNil(accepted["proposalID"])
        var mixed=example;mixed["separator"] = .string(",")
        let refused=try await preview(mixed);XCTAssertEqual(refused["code"],.string("invalid_configuration"));XCTAssertNil(refused["proposalID"])
        XCTAssertEqual(refused["diagnosticCode"],.string("api.inputEnvelope"))
        guard case .object(var options)=example["options"] else {return XCTFail("Missing example options")}
        let secret="PRIVATE-SENTINEL-DO-NOT-ECHO";options["headers"] = .object(["Authorization":.string(secret)])
        var bad=example;bad["options"] = .object(options)
        let hidden=try await preview(bad);XCTAssertEqual(hidden["diagnosticCode"],.string("api.requestFields"));XCTAssertNil(hidden["proposalID"])
        XCTAssertFalse(String(decoding:try JSONEncoder().encode(hidden),as:UTF8.self).contains(secret))
        XCTAssertEqual(f.owner.document,before);XCTAssertFalse(f.owner.isBusy)
    }
    func testProjectLifecycleCleanCloseNeverAutosavesNativeUnsavedState()async throws {
        let f=try fixture(),id=f.owner.projectID,root=f.controller.directory
        let window=try XCTUnwrap(f.windows.openControllers.first)
        window.session.workspace.hasUnsavedChanges=true
        let snapshot=try ProjectLifecycleAuthority.snapshot(projectID:id,root:root,windows:f.windows)
        XCTAssertFalse(snapshot.dirty,"Native unsaved state is independent of explicit version differences")
        XCTAssertTrue(snapshot.nativeUnsavedChanges)
        let bytes=try Data(contentsOf:CanvasVersionJournal(root:root,projectID:id).fileURL)
        XCTAssertThrowsError(try ProjectLifecycleAuthority.close(projectID:id,revision:snapshot.lifecycleRevision,root:root,windows:f.windows))
        XCTAssertEqual(f.windows.projectCount,1)
        XCTAssertTrue(window.session.workspace.hasUnsavedChanges,"Clean-close must not invoke native autosave")
        XCTAssertEqual(try Data(contentsOf:CanvasVersionJournal(root:root,projectID:id).fileURL),bytes)
        window.session.workspace.hasUnsavedChanges=false
    }
    func testProjectLifecycleRefusesStaleBusyDraftsAndCorruptJournal()async throws {
        let f=try fixture(),id=f.owner.projectID,root=f.controller.directory
        let initial=try ProjectLifecycleAuthority.snapshot(projectID:id,root:root,windows:f.windows)
        XCTAssertTrue(f.owner.rename("Native changed",lease:f.owner.lease()))
        XCTAssertThrowsError(try ProjectLifecycleAuthority.rename(projectID:id,name:"Lost update",revision:initial.lifecycleRevision,root:root,windows:f.windows))
        let latest=try ProjectLifecycleAuthority.snapshot(projectID:id,root:root,windows:f.windows)
        XCTAssertThrowsError(try ProjectLifecycleAuthority.rename(projectID:id,name:String(repeating:"x",count:81),revision:latest.lifecycleRevision,root:root,windows:f.windows))
        f.owner.setExecutionBusy(true)
        let busy=try ProjectLifecycleAuthority.snapshot(projectID:id,root:root,windows:f.windows)
        XCTAssertTrue(busy.busy)
        XCTAssertThrowsError(try ProjectLifecycleAuthority.close(projectID:id,revision:busy.lifecycleRevision,root:root,windows:f.windows))
        do {_=try await call(f,.projectSave);XCTFail("Busy save succeeded")}catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"save_refused")}
        f.owner.setExecutionBusy(false)
        let address=WorkflowNodeAddress(domain:.data,localID:"view")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:address,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:f.owner.lease()))
        _=f.owner.select(.init(nodeAddresses:[address]),lease:f.owner.lease())
        let edit=UUID()
        XCTAssertTrue(f.owner.setInspectorEditPending(contextID:edit,address:address,key:"test",pending:true,lease:f.owner.lease()))
        let draft=try ProjectLifecycleAuthority.snapshot(projectID:id,root:root,windows:f.windows)
        XCTAssertTrue(draft.pendingInspectorEdits)
        XCTAssertThrowsError(try ProjectLifecycleAuthority.rename(projectID:id,name:"Discarded draft",revision:draft.lifecycleRevision,root:root,windows:f.windows))
        f.owner.clearDataEdits(contextID:edit)
        let journal=try CanvasVersionJournal(root:root,projectID:id),bytes=try Data(contentsOf:journal.fileURL)
        defer {try? bytes.write(to:journal.fileURL)}
        try Data("private-corruption-test".utf8).write(to:journal.fileURL)
        do {_=try await f.gateway.dispatch(clientID:f.client,action:"project_status",arguments:["projectID":.string(id)]);XCTFail("Corrupt journal accepted")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"project_lifecycle_unavailable");XCTAssertFalse(error.localizedDescription.contains("private-corruption-test"))}
        XCTAssertEqual(f.windows.projectCount,1)
    }
    func testProjectLifecycleIsScopedCleanAndReportsSavedVersion()async throws {
        let f=try fixture(),id=f.owner.projectID,args:[String:JSONValue]=["projectID":.string(f.owner.projectID)]
        let first=try await f.gateway.dispatch(clientID:f.client,action:"project_status",arguments:args)
        XCTAssertEqual(first["open"],true);XCTAssertEqual(first["dirty"],false)
        var rename=args;rename["lifecycleRevision"]=first["lifecycleRevision"];rename["name"] = .string("Renamed safely");rename["requestID"] = .string(UUID().uuidString)
        let renamed=try await f.gateway.dispatch(clientID:f.client,action:"project_rename",arguments:rename)
        XCTAssertEqual(renamed["dirty"],true);XCTAssertEqual(f.owner.document.name,"Renamed safely")
        var close=args;close["lifecycleRevision"]=renamed["lifecycleRevision"];close["requestID"] = .string(UUID().uuidString)
        do {_=try await f.gateway.dispatch(clientID:f.client,action:"project_close",arguments:close);XCTFail("Dirty project closed")}catch {}
        XCTAssertEqual(f.windows.projectCount,1)
        let saved=try await call(f,.projectSave)
        guard case .object(let version)=saved["savedVersion"] else {return XCTFail("Missing exact saved journal version")}
        let journal=try CanvasVersionJournal(root:f.controller.directory,projectID:id)
        XCTAssertEqual(version["id"],.string(try XCTUnwrap(journal.versions.first).id))
        XCTAssertEqual(version["hash"],.string(try XCTUnwrap(journal.versions.first).hash))
        let clean=try await f.gateway.dispatch(clientID:f.client,action:"project_status",arguments:args)
        XCTAssertEqual(clean["dirty"],false)
        close["lifecycleRevision"]=clean["lifecycleRevision"];close["requestID"] = .string(UUID().uuidString)
        let closed=try await f.gateway.dispatch(clientID:f.client,action:"project_close",arguments:close)
        XCTAssertEqual(closed["open"],false);XCTAssertEqual(f.windows.projectCount,0)
        let after=try await f.gateway.dispatch(clientID:f.client,action:"project_status",arguments:args)
        XCTAssertEqual(after["open"],false);XCTAssertEqual(f.windows.projectCount,0,"Read must not open the project")
        XCTAssertFalse(String(decoding:try JSONEncoder().encode(after),as:UTF8.self).contains(f.controller.directory.path))
        var policy=f.controller.settings;policy.projects[id] = .createdProject;try f.controller.save(policy)
        do {_=try await f.gateway.dispatch(clientID:f.client,action:"project_status",arguments:args);XCTFail("Legacy grant broadened")}catch {}
    }
    func testSupervisedTrainDiscoveryAndConfigurationRoundtrip()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["workflow.train"])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let value)=$0{return value["selfPlay"] != nil};return false}) else {return XCTFail("Train missing")}
        XCTAssertEqual(row["supervisedTraining"],try WorkflowTrainConfigurationDiscovery.contract())
        let example=try WorkflowTrainConfigurationDiscovery.supervisedExample()
        let insert:JSONValue = .object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("train")])
        let preview=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([insert])])
        _=try await call(f,.patchApply,["proposalID":preview["proposalID"]!])
        let address:JSONValue = .object(["domain":.string("control"),"localID":.string("train")])
        func configure(_ value:[String:JSONValue])async throws->[String:JSONValue] {
            try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("configure"),"address":address,"configuration":.object(value)])])])
        }
        let proposed=try await configure(example)
        _=try await call(f,.patchApply,["proposalID":proposed["proposalID"]!])
        guard case .train(let actual)=f.owner.document.nodes.first(where:{$0.address.localID=="train"})?.payload else {return XCTFail("Train not retained")}
        XCTAssertEqual(try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(actual)),try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(example)))
        let inspected=try await call(f,.workflowInspect)
        let inspection=String(decoding:try JSONEncoder().encode(inspected),as:UTF8.self)
        XCTAssertTrue(inspection.contains("loss_mse"));XCTAssertTrue(inspection.contains("target_2"));XCTAssertTrue(inspection.contains("cpu"))
        let before=f.owner.document
        var invalid=example;invalid["runSetup"] = .object(["device":.string("cuda"),"epochs":3])
        do {_=try await configure(invalid);XCTFail("Unsupported device admitted")}catch {}
        invalid=example;invalid["runSetup"] = .object(["epoch":3])
        do {_=try await configure(invalid);XCTFail("Typo silently ignored")}catch {}
        invalid=example;invalid["dataMapping"] = .object(["featureColums":.array([.string("x")]),"targetColumn":.string("y")])
        do {_=try await configure(invalid);XCTFail("Mapping typo admitted")}catch {}
        invalid=example;invalid["runSetup"] = .object(["epochs":0])
        do {_=try await configure(invalid);XCTFail("Epoch clamped")}catch {}
        invalid=example;invalid["objective"] = .object(["schemaVersion":1,"componentID":.string("loss_mse"),"configurationVersion":1,"config":.object(["reduccion":.string("sum")])])
        do {_=try await configure(invalid);XCTFail("Loss typo ignored")}catch {}
        invalid=example;invalid["dataMapping"] = .object(["lossTargetMapping":.object(["schemaVersion":1,"componentID":.string("loss_mse"),"inputColumns":.array([.string("x1"),.string("x2")]),"roles":.array([.object(["role":.string("target"),"columns":.array([.string("y1"),.string("y2")]),"sampleShape":.array([3]),"dtype":.string("float32")])]),"tripletSourceGroups":.array([])])])
        do {_=try await configure(invalid);XCTFail("Target shape mismatch admitted")}catch {}
        XCTAssertEqual(f.owner.document,before)
        let text=String(decoding:try JSONEncoder().encode(row["supervisedTraining"]),as:UTF8.self)
        XCTAssertFalse(text.contains(f.controller.directory.path));XCTAssertFalse(text.contains("Bearer"))
    }
    func testCreatedProjectRequiresSeparateSettingsConsent()async throws {
        let f=try fixture(),prior=f.controller.settings.projects
        let created=try await call(f,.projectCreate,["name":.string("Scoped new project")])
        let id=try XCTUnwrap(created["projectID"]?.stringValue)
        let grant=try XCTUnwrap(f.controller.settings.projects[id])
        let expected:Set<MCPAction>=[.projectOpen,.projectSave,.workflowInspect,.workflowValidate,.patchPreview,.patchApply,.patchUndo,.runStart,.runCancel,.jobGet,.resultsRead]
        XCTAssertEqual(grant.actions,expected)
        XCTAssertFalse(grant.readCapturedValues);XCTAssertFalse(grant.readHistory)
        for (key,value) in prior {XCTAssertEqual(f.controller.settings.projects[key],value)}
        for action:MCPAction in [.settingsGet,.settingsPreview,.settingsApply] {
            do {_=try await f.gateway.dispatch(clientID:f.client,action:action.rawValue,arguments:["projectID":.string(id)]);XCTFail("New project granted settings implicitly")}
            catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
        }
        _=try await f.gateway.dispatch(clientID:f.client,action:"workflow_inspect",arguments:["projectID":.string(id)])
        _=try await f.gateway.dispatch(clientID:f.client,action:"project_save",arguments:["projectID":.string(id)])
        var policy=f.controller.settings;policy.projects[id]?.actions.insert(.settingsGet);try f.controller.save(policy)
        _=try await f.gateway.dispatch(clientID:f.client,action:"settings_get",arguments:["projectID":.string(id)])
    }
    func testSaveTerminalUsesNativeRefusalRatherThanUnsupportedTerminal()async throws {
        let f=try fixture(),address=WorkflowNodeAddress(domain:.control,localID:"save")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:address,payload:.saveModel(config:[:]))),lease:f.owner.lease()))
        do { _=try await call(f,.runStart,["terminal":.object(["domain":.string("control"),"localID":.string("save")])]);XCTFail("Blocked Save admitted") }
        catch let error as MCPSettings.Refusal { XCTAssertEqual(error.code,"not_permitted") }
        var settings=f.controller.settings;settings.disabledComponents.remove("workflow.saveModel");try f.controller.save(settings)
        let missing=try await call(f,.resultsRead,["address":.object(["domain":.string("control"),"localID":.string("save")])])
        XCTAssertEqual(missing["available"],false);XCTAssertNil(missing["exports"])
        do { _=try await call(f,.runStart,["terminal":.object(["domain":.string("control"),"localID":.string("save")])]);XCTFail("Missing model admitted") }
        catch let error as MCPSettings.Refusal { XCTAssertEqual(error.code,"run_refused") }
        XCTAssertFalse(f.owner.isBusy)
    }
    func testSaveTerminalValidationUsesSharedAuthoringPlanWithoutRequiringAFit()async throws {
        let f=try fixture(),save=WorkflowNodeAddress(domain:.control,localID:"save")
        let doc=try PrebuiltModelsPractice.document(projectID:f.owner.projectID)
        var settings=f.controller.settings;settings.disabledComponents.remove("workflow.saveModel");try f.controller.save(settings)
        XCTAssertNotNil(f.owner.applyBatch(doc.nodes.map(WorkflowCommand.insert)+doc.edges.map{WorkflowCommand.connect(from:$0.from,to:$0.to)},lease:f.owner.lease()))
        let endpoint=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(save))
        let before=f.owner.document
        let result:[String:JSONValue]
        do {result=try await call(f,.workflowValidate,["terminal":endpoint])}
        catch {return XCTFail("Save authoring validation refused: \(error)")}
        XCTAssertEqual(result["valid"],true)
        XCTAssertEqual(result["scope"],.string("Save model authoring plan; verified source artifact, private destination access and runtime admission remain execution prerequisites"))
        XCTAssertEqual(f.owner.document,before);XCTAssertFalse(f.owner.isBusy)
        let receipt=try await call(f,.resultsRead,["address":endpoint])
        XCTAssertNil(receipt["exports"],"Authoring validation cannot manufacture an export receipt")
    }
    func testSaveTerminalValidationRejectsMissingAndUnsupportedModelProviders()async throws {
        for unsupported in [false,true] {
            let f=try fixture(),save=WorkflowNodeAddress(domain:.control,localID:"save")
            var settings=f.controller.settings;settings.disabledComponents.remove("workflow.saveModel");try f.controller.save(settings)
            var commands:[WorkflowCommand]=[.insert(.init(address:save,payload:.saveModel(config:[:]))) ]
            if unsupported {
                let source=WorkflowNodeAddress(domain:.control,localID:"fit-only-analysis")
                commands += [.insert(.init(address:source,payload:.estimator(config:PrebuiltModelConfiguration.defaults("dbscan")))),.connect(from:.init(address:source,port:"model"),to:.init(address:save,port:"model"))]
            }
            if unsupported {
                XCTAssertNil(f.owner.applyBatch(commands,lease:f.owner.lease()),"Authoring must reject DBSCAN's nonexistent model output")
                XCTAssertNotNil(f.owner.apply(.insert(.init(address:save,payload:.saveModel(config:[:]))),lease:f.owner.lease()))
            } else {XCTAssertNotNil(f.owner.applyBatch(commands,lease:f.owner.lease()))}
            let endpoint=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(save))
            let refusal=try await call(f,.workflowValidate,["terminal":endpoint])
            XCTAssertEqual(refusal["isError"],.bool(true));XCTAssertEqual(refusal["reason"],.string("requiredInput"));XCTAssertFalse(f.owner.isBusy)
            XCTAssertNil(refusal["jobID"])
        }
    }
    private func evaluationValidationDocument()throws->WorkflowDocument {
        var doc=try WorkflowExamples.preparedGroupSplitClassifier()
        doc.nodes.removeAll {$0.address.domain != .data}
        let retained=Set(doc.nodes.map(\.address));doc.edges.removeAll {!retained.contains($0.from.address) || !retained.contains($0.to.address)}
        let frozen=WorkflowNodeAddress(domain:.control,localID:"frozen"),evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        doc.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:[:]))]
        doc.edges += [.init(from:.init(address:frozen,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),.init(from:.init(address:.init(domain:.data,localID:"split"),port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
        return doc
    }
    func testSpecializedTerminalValidationIsPureAndUsesSharedPlans()async throws {
        let cases:[(WorkflowDocument,WorkflowNodeAddress,String)]=[
            (try evaluationValidationDocument(),.init(domain:.control,localID:"evaluate"),"evaluation authoring plan"),
            (try GameEpisodePractice.document(),GameEpisodePractice.session,"game session authoring plan"),
            (try SelfPlayPractice.document(),SelfPlayPractice.train,"self-play authoring plan"),
            (try TextGenerationPracticeExample.document(),TextGenerationPracticeExample.address,"generation configuration and model-provider binding only")]
        for (document,address,scope) in cases {
            let f=try fixture(document:document)
            var settings=f.controller.settings;settings.disabledComponents.remove("workflow.saveModel");try f.controller.save(settings)
            let host=try XCTUnwrap(f.windows.openControllers.first?.session.workspace.workflowSession)
            let original=f.owner.document,identity=host.operationIdentity,bindings=host.frozenBindings.revision
            host.executionLaunch={_,_,_,_,_ in XCTFail("Validation must never launch a worker");throw CancellationError()}
            let endpoint=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(address))
            do {
                let result=try await call(f,.workflowValidate,["terminal":endpoint])
                XCTAssertEqual(result["valid"],true);XCTAssertTrue(result["scope"]?.stringValue?.hasPrefix(scope)==true)
                XCTAssertNil(result["jobID"])
            } catch {XCTFail("\(scope) refused: \(error)")}
            XCTAssertEqual(f.owner.document,original);XCTAssertEqual(host.operationIdentity,identity);XCTAssertEqual(host.frozenBindings.revision,bindings)
            XCTAssertFalse(host.hasActiveOperation);XCTAssertNil(host.backendFit)
            let results=try await call(f,.resultsRead,["address":endpoint]);XCTAssertNil(results["exports"])
        }
    }
    func testSpecializedValidationRejectsMissingRequiredBindings()async throws {
        let cases:[(WorkflowDocument,WorkflowNodeAddress,String)]=[
            (try evaluationValidationDocument(),.init(domain:.control,localID:"evaluate"),"test"),
            (try GameEpisodePractice.document(),GameEpisodePractice.session,"definition"),
            (try SelfPlayPractice.document(),SelfPlayPractice.train,"definition"),
            (try TextGenerationPracticeExample.document(),TextGenerationPracticeExample.address,"model")]
        for (source,address,port) in cases {
            var document=source;document.edges.removeAll {$0.to.address==address && $0.to.port==port}
            let f=try fixture(document:document)
            var settings=f.controller.settings;settings.disabledComponents.remove("workflow.saveModel");try f.controller.save(settings)
            let before=f.owner.document,endpoint=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(address))
            let refusal=try await call(f,.workflowValidate,["terminal":endpoint])
            XCTAssertEqual(refusal["isError"],.bool(true));XCTAssertEqual(refusal["valid"],.bool(false));XCTAssertNil(refusal["jobID"])
            XCTAssertEqual(f.owner.document,before);XCTAssertFalse(f.owner.isBusy)
        }
    }
    func testExplicitDataValidationRefusesMissingInputs()async throws {
        let f=try fixture(),address=WorkflowNodeAddress(domain:.data,localID:"view")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:address,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:f.owner.lease()))
        let endpoint=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(address))
        let refusal=try await call(f,.workflowValidate,["terminal":endpoint])
        XCTAssertEqual(refusal["code"],.string("workflow_validation_failed"));XCTAssertEqual(refusal["valid"],.bool(false));XCTAssertNil(refusal["jobID"])
    }
    func testValidationConfigurationAndLineageFailuresAreSafeAndPure()async throws {
        for reason in ["configuration","lineage"] {
            var document=try WorkflowExamples.preparedGroupSplitClassifier()
            if reason=="configuration" {
                let index=try XCTUnwrap(document.nodes.firstIndex(where:DataFlowSourceOperators.isSource))
                guard case .dataOperator(let id,var config)=document.nodes[index].payload else {return XCTFail("Missing source")}
                config["separator"] = .object(["secret":.string("/private/customer.csv")])
                document.nodes[index].payload = .dataOperator(operatorID:id,config:config)
            } else {
                let index=try XCTUnwrap(document.edges.firstIndex{$0.to.address.localID=="fit" && $0.to.port=="training"})
                document.edges[index].from.port="validation"
            }
            let f=try fixture(document:document),host=try XCTUnwrap(f.windows.openControllers.first?.session.workspace.workflowSession)
            let original=f.owner.document,identity=host.operationIdentity
            host.executionLaunch={_,_,_,_,_ in XCTFail("Authoring validation launched worker");throw CancellationError()}
            let endpoint=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(WorkflowExamples.trainAddress))
            let result=try await call(f,.workflowValidate,["terminal":endpoint])
            XCTAssertEqual(result["reason"],.string(reason));XCTAssertEqual(result["valid"],.bool(false));XCTAssertNotNil(result["component"])
            XCTAssertNil(result["jobID"]);XCTAssertEqual(f.owner.document,original);XCTAssertEqual(host.operationIdentity,identity);XCTAssertFalse(host.hasActiveOperation)
            XCTAssertFalse(String(decoding:try JSONEncoder().encode(result),as:UTF8.self).contains("/private/customer.csv"))
        }
    }
    func testCapturedTerminalPrerequisiteIsActionableAndDoesNotCreateJob()async throws {
        let train=WorkflowNodeAddress(domain:.control,localID:"train"),view=WorkflowNodeAddress(domain:.data,localID:"view")
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:train,payload:.train(config:[:])),.init(address:view,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))],edges:[.init(id:"predictions",from:.init(address:train,port:"predictions"),to:.init(address:view,port:"table"),kind:.table)])
        let f=try fixture(document:document),terminal=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(view))
        for action in [MCPAction.workflowValidate,.runStart] {
            let result=try await call(f,action,["terminal":terminal])
            XCTAssertEqual(result["code"],.string("captured_table_required"));XCTAssertEqual(result["isError"],.bool(true));XCTAssertNil(result["jobID"])
        }
        let omitted=try await call(f,.runStart)
        XCTAssertEqual(omitted["code"],.string("terminal_required"));XCTAssertNil(omitted["jobID"])
        XCTAssertEqual(f.owner.document,document)
        var settings=f.controller.settings;settings.clients[0].disabledComponents.insert("workflow.train");try f.controller.save(settings)
        do {_=try await call(f,.workflowValidate,["terminal":terminal]);XCTFail("Disabled producer must be checked before capture diagnostics")} catch {}
    }
    func testScopedJobFailureExposesSafeActionableFormulaDiagnostic()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Explicit worker required")}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),formula=WorkflowNodeAddress(domain:.data,localID:"formula")
        let operation:JSONValue = .object(["id":.string("op"),"expression":.string("[customer] + \"private suffix\""),"outputs":.array([.object(["name":.string("combined"),"dtype":.string("string"),"write":.string("new")])]),"nullPolicy":.string("preserve")])
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),.init(address:formula,payload:.dataOperator(operatorID:"kwip.data.formula",config:["operations":.array([operation])]))],edges:[.init(id:"input",from:.init(address:source,port:"table"),to:.init(address:formula,port:"table"),kind:.table)])
        let f=try fixture(document:document),workspace=try XCTUnwrap(f.windows.openControllers.first?.session.workspace),host=try XCTUnwrap(workspace.workflowSession)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(host.bindings.setPath(root.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)}
        let terminal=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(formula))
        let accepted=try await call(f,.runStart,["terminal":terminal]),job=try XCTUnwrap(accepted["jobID"])
        let deadline=Date().addingTimeInterval(30)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        let status=try await call(f,.jobGet,["jobID":job])
        guard case .object(let result)=status["job"],case .object(let diagnostic)=result["diagnostic"] else {return XCTFail("Missing diagnostic: \(status) / \(host.statusMessage)")}
        XCTAssertEqual(result["state"],.string("failed"));XCTAssertEqual(diagnostic["code"],.string("expression_numeric_required"))
        let wire=String(decoding:try JSONEncoder().encode(status),as:UTF8.self)
        XCTAssertFalse(wire.contains("private suffix"));XCTAssertFalse(wire.contains(root.path));XCTAssertFalse(wire.contains("data02-observations.csv"))
        XCTAssertNil(host.currentExecution);XCTAssertTrue(host.scopedPreparation?.savedOutputs.isEmpty == true)
    }
    func testExplicitSourceMCPCompletesOwnedJobWithoutDownstreamOutput()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Explicit worker required")}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let source=WorkflowNodeAddress(domain:.data,localID:"source")
        let document=WorkflowDocument(projectID:UUID().uuidString,name:"Scoped MCP",nodes:[.init(address:source,payload:.dataOperator(operatorID:"kwip.data.sourceCSV",config:["separator":.string(",")])),.init(address:.init(domain:.control,localID:"unrelated-train"),payload:.train(config:[:]))],edges:[])
        let f=try fixture(document:document),workspace=try XCTUnwrap(f.windows.openControllers.first?.session.workspace),host=try XCTUnwrap(workspace.workflowSession)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(host.bindings.setPath(root.appendingPathComponent("runner/dataflow_examples/data02-observations.csv").path,for:source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        host.executionLaunch={_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)}
        let terminal=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(source))
        let validation=try await call(f,.workflowValidate,["terminal":terminal]);XCTAssertEqual(validation["valid"],true)
        XCTAssertNil(host.currentDataPreparation)
        let accepted=try await call(f,.runStart,["terminal":terminal]),job=try XCTUnwrap(accepted["jobID"])
        let deadline=Date().addingTimeInterval(60)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        let status=try await call(f,.jobGet,["jobID":job])
        guard case .object(let jobStatus)=status["job"] else {return XCTFail("Missing job envelope")}
        XCTAssertEqual(jobStatus["state"],.string("completed"),"\(status) / \(host.statusMessage)")
        XCTAssertNil(host.currentExecution);XCTAssertNil(host.currentBackendFit)
        XCTAssertEqual(host.owner.document,document)
    }
    func testFullCellMCPUsesPhysicalRowsUnicodeChunksAndRevocableCurrentEvidence()async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Explicit development worker required")}
        let f=try fixture(),store=f.windows.restorationWorkspace
        let address=WorkflowNodeAddress(domain:.data,localID:"view")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:address,payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:f.owner.lease()))
        let original=f.owner.document
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=try canonicalBackendTemporaryDirectory();try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let long=String(repeating:"🔎e\u{301} words\n",count:5000)
        let values=root.appendingPathComponent("values.json"),index=root.appendingPathComponent("view.sqlite3")
        try JSONSerialization.data(withJSONObject:[[0,"short"],[1,long]]).write(to:values)
        let process=Process();process.executableURL=URL(fileURLWithPath:python);process.currentDirectoryURL=repository
        process.arguments=["-c","import json,sys;from pathlib import Path;from types import SimpleNamespace;from runner.table_view import build_index;rows=json.loads(Path(sys.argv[1]).read_text());build_index(SimpleNamespace(columns=[dict(name='id',dtype='int64',nullable=False),dict(name='text',dtype='string',nullable=False)],rows=rows,row_ids=['a','b']),Path(sys.argv[2]),source_sha256='a'*64)",values.path,index.path]
        try process.run();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0)
        await verifyRuntimeForIntegration(store)
        let runtime=try XCTUnwrap(store.runtime),pin=WorkflowArchiveCodec.hash(try Data(contentsOf:index))
        var held=false,current=true
        let session=TableViewQuerySession(artifact:.init(indexURL:index,indexSHA256:pin,sourceSHA256:String(repeating:"a",count:64)),workspace:root,projectID:f.owner.projectID,runtime:runtime,reservation:.init(acquire:{if held{return nil};held=true;return UUID()},release:{_ in held=false}),accepts:{current})
        defer {session.invalidate()}
        XCTAssertTrue(session.load(sort:.init(column:0,descending:true)))
        let deadline=Date().addingTimeInterval(45)
        while session.isBusy && Date()<deadline {try await Task.sleep(for:.milliseconds(20))}
        let queryID=f.gateway.retainTableQuery(client:f.client,project:f.owner.projectID,address:address,session:session)
        let base:[String:JSONValue]=["address":.object(["domain":.string("data"),"localID":.string("view")]),"queryID":.string(queryID),"includeValues":true]
        let paged=try await call(f,.resultsRead,base)
        XCTAssertEqual(paged["positions"],.array([1,0]));XCTAssertEqual(paged["truncated"],true)
        func cell(_ start:Int)async throws->[String:JSONValue] {
            var args=base;args["cellRow"]=1;args["cellColumn"]=1;args["cellStart"] = .integer(Int64(start))
            var response=try await call(f,.resultsRead,args)
            guard let token=response["cellReadID"] else {XCTFail("Missing cell handle");return response}
            var poll=base;poll["cellReadID"]=token
            while response["state"] == .string("loading") && Date()<deadline {try await Task.sleep(for:.milliseconds(20));response=try await call(f,.resultsRead,poll)}
            return response
        }
        let first=try await cell(0)
        XCTAssertEqual(first["text"],.string(String(String.UnicodeScalarView(long.unicodeScalars.prefix(4096)))))
        XCTAssertEqual(first["nextOffset"],4096);XCTAssertEqual(first["completeCell"],false);XCTAssertEqual(first["units"],.string("unicodeScalars"))
        let second=try await cell(4096)
        XCTAssertEqual(second["text"],.string(String(String.UnicodeScalarView(long.unicodeScalars.dropFirst(4096).prefix(4096)))))
        let final=try await cell(long.unicodeScalars.count-4096);XCTAssertEqual(final["hasMore"],false);XCTAssertEqual(final["completeCell"],false)
        XCTAssertFalse(String(data:try JSONEncoder().encode(first),encoding:.utf8)!.contains(root.path))
        var invalid=base;invalid["cellRow"]=99;invalid["cellColumn"]=1
        do {_=try await call(f,.resultsRead,invalid);XCTFail("Unreturned physical position accepted")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"invalid_cell_request")}
        for start:JSONValue in [-1,.integer(Int64(TableViewQuery.maximumCellScalars+1)),.bool(true)] {
            var args=base;args["cellRow"]=1;args["cellColumn"]=1;args["cellStart"]=start
            do {_=try await call(f,.resultsRead,args);XCTFail("Invalid scalar offset accepted")}
            catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"invalid_cell_request")}
        }
        let unavailable=try await cell(long.unicodeScalars.count+1)
        XCTAssertEqual(unavailable["state"],.string("unavailable"));XCTAssertNil(unavailable["text"])
        XCTAssertFalse(String(data:try JSONEncoder().encode(unavailable),encoding:.utf8)!.contains(root.path))
        var shortArgs=base;shortArgs["cellRow"]=0;shortArgs["cellColumn"]=1
        var short=try await call(f,.resultsRead,shortArgs)
        var shortPoll=base;shortPoll["cellReadID"]=short["cellReadID"]
        while short["state"] == .string("loading") && Date()<deadline {try await Task.sleep(for:.milliseconds(20));short=try await call(f,.resultsRead,shortPoll)}
        XCTAssertEqual(short["text"],.string("short"));XCTAssertEqual(short["completeCell"],true);XCTAssertEqual(short["hasMore"],false)
        var poll=base;poll["cellReadID"]=short["cellReadID"]
        var otherPolicy=f.controller.settings;otherPolicy.clients.append(.init(id:"other",name:"Other",actions:Set(MCPAction.allCases)));try f.controller.save(otherPolicy)
        var foreign=poll;foreign["projectID"] = .string(f.owner.projectID)
        do {_=try await f.gateway.dispatch(clientID:"other",action:"results_read",arguments:foreign);XCTFail("Other client reused cell handle")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"query_unavailable")}
        var policy=f.controller.settings;policy.projects[f.owner.projectID]?.readCapturedValues=false;try f.controller.save(policy)
        do {_=try await call(f,.resultsRead,poll);XCTFail("Revoked values read")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
        policy.projects[f.owner.projectID]?.readCapturedValues=true;try f.controller.save(policy)
        current=false
        do {_=try await call(f,.resultsRead,poll);XCTFail("Stale completed cell read")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"query_unavailable")}
        XCTAssertEqual(f.owner.document,original)
    }
    private func applicationCall(_ f:Fixture,_ name:String,_ arguments:[String:JSONValue]=[:],client:String?=nil)async throws->[String:JSONValue] {
        var args=arguments
        if name=="application_settings_apply",args["requestID"]==nil {args["requestID"] = .string(UUID().uuidString)}
        return try await f.gateway.dispatch(clientID:client ?? f.client,action:name,arguments:args)
    }
    private func applicationPreview(_ f:Fixture,group:String,values:[String:JSONValue]?=nil,reset:Bool=false)async throws->[String:JSONValue] {
        let snapshot=try await applicationCall(f,"application_settings_get")
        var args:[String:JSONValue] = ["settingsRevision":try XCTUnwrap(snapshot["settingsRevision"]),"group":.string(group),"reset":.bool(reset)]
        if let values {args["values"] = .object(values)}
        return try await applicationCall(f,"application_settings_preview",args)
    }
    func testApplicationSettingsNotificationObservationNeverProjectsNativeErrors()throws {
        let expected=MCPWorkflowGateway.applicationFeedbackObservation("Notifications are off.")
        for raw in ["System notifications enabled.","Delivery failed: /Users/private/token=secret","Permission granted"] {
            let result=MCPWorkflowGateway.applicationFeedbackObservation(raw)
            XCTAssertEqual(result,expected)
            XCTAssertFalse(String(data:try JSONEncoder().encode(result),encoding:.utf8)!.contains(raw))
            guard case .object(let value)=result else {return XCTFail("Missing typed observation")}
            XCTAssertEqual(value["authorization"],.string("unknown"));XCTAssertEqual(value["permissionRequested"],false)
        }
    }
    func testApplicationCacheConstraintsAndApplyUseCapturedWorkload() async throws {
        let f = try fixture(), store = f.windows.restorationWorkspace
        let policy = try ExecutionWorkloadPolicy(preset: .balanced)
        let before = try store.applicationSettingsSnapshot()
        _ = try store.applyApplicationSettings(store.previewApplicationSettings(.execution(try .init(workloadPolicy: policy)), baseline: before))
        let inventory = try await applicationCall(f, "application_settings_get")
        guard case .object(let constraints) = inventory["constraints"] else { return XCTFail("Missing constraints") }
        XCTAssertEqual(constraints["preparationCacheLimitBytes"], .array([1, .integer(policy.limit(.retainedBytes))]))
        let preview = try await applicationPreview(f, group: "storage", values: ["managedStorageLimitBytes": .null, "preparationCacheLimitBytes": 300_000_000])
        _ = try await applicationCall(f, "application_settings_apply", ["proposalID": try XCTUnwrap(preview["proposalID"])])
        XCTAssertEqual(try store.appPreferenceStore.load().execution.preparationCacheLimitBytes, 300_000_000)
        XCTAssertEqual(try store.appPreferenceStore.load().execution.workloadPolicy, policy)
    }
    func testApplicationSettingsInventoryPreviewApplyAndDurableRetry()async throws {
        let f=try fixture(),store=f.windows.restorationWorkspace,before=try store.appPreferenceStore.load(),graph=f.owner.document
        let inventory=try await applicationCall(f,"application_settings_get")
        guard case .array(let groups)=inventory["supportedGroups"] else {return XCTFail("Missing application settings inventory")}
        XCTAssertEqual(Set(groups.compactMap(\.stringValue)),Set(["execution","storage","trainingCapture","derivativeCapture","appearance","canvas","feedback"]))
        let preview=try await applicationPreview(f,group:"canvas",values:["zoomSensitivity":0.75,"restoreAtLaunch":true])
        XCTAssertEqual(try store.appPreferenceStore.load(),before)
        let args:[String:JSONValue] = ["proposalID":try XCTUnwrap(preview["proposalID"]),"requestID":.string(UUID().uuidString)]
        let applied=try await applicationCall(f,"application_settings_apply",args)
        XCTAssertEqual(applied["applied"],true);XCTAssertEqual(try store.appPreferenceStore.load().effectiveCanvasZoomSensitivity,0.75)
        let retry=try await applicationCall(f,"application_settings_apply",args);XCTAssertEqual(retry,applied)
        let recreated=MCPWorkflowGateway(controller:f.controller,windows:f.windows)
        let durable=try await recreated.dispatch(clientID:f.client,action:"application_settings_apply",arguments:args);XCTAssertEqual(durable,try JSONDecoder().decode([String:JSONValue].self,from:JSONEncoder().encode(applied)))
        XCTAssertEqual(f.owner.document,graph)
        let wire=String(data:try JSONEncoder().encode(applied),encoding:.utf8)!
        XCTAssertFalse(wire.contains(store.applicationDirectory.path));XCTAssertFalse(wire.contains("fileURL"))
    }
    func testApplicationSettingsEveryTypedGroupAndReplayRevocation()async throws {
        let f=try fixture(),store=f.windows.restorationWorkspace,graph=f.owner.document
        var initialPolicy=f.controller.settings;initialPolicy.projects[f.owner.projectID]?.readCapturedValues=false;try f.controller.save(initialPolicy)
        let groups:[(String,[String:JSONValue])]=[
            ("execution",["maximumRunSeconds":120,"cpuThreads":2,"maximumResidentBytes":.null]),
            ("storage",["managedStorageLimitBytes":.null,"preparationCacheLimitBytes":1000000]),
            ("trainingCapture",["activationValues":false,"predictionRows":true,"performanceMeasurements":true,"tensorHeatmaps":false,"heatmapFrameLimit":0]),
            ("derivativeCapture",["values":true]),
            ("appearance",["mode":.string("light"),"palette":.string("teal"),"transparency":.string("opaque")]),
            ("feedback",["showStatusIcon":false,"notifyCompleted":false,"notifyFailed":false])]
        var lastArgs:[String:JSONValue]=[:]
        for (group,values) in groups {
            let preview=try await applicationPreview(f,group:group,values:values)
            lastArgs=["proposalID":try XCTUnwrap(preview["proposalID"]),"requestID":.string(UUID().uuidString)]
            let applied=try await applicationCall(f,"application_settings_apply",lastArgs)
            guard case .object(let saved)=applied["savedGroups"] else {return XCTFail("Missing typed groups")}
            XCTAssertEqual(saved[group],.object(values))
        }
        let before=try store.appPreferenceStore.load()
        var settings=f.controller.settings;settings.clients[0].actions=Set(settings.clients[0].actions.filter{$0.rawValue != "application_settings_preview"});try f.controller.save(settings)
        do {_=try await applicationCall(f,"application_settings_apply",lastArgs);XCTFail("Revoked preview grant replayed durable receipt")}
        catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"not_permitted")}
        XCTAssertEqual(try store.appPreferenceStore.load(),before);XCTAssertEqual(f.owner.document,graph)
        XCTAssertTrue(f.controller.settings.projects[f.owner.projectID]?.readCapturedValues == false)
    }
    func testApplicationSettingsRejectCrossClientMixedKeysAndConflictingRequest()async throws {
        let f=try fixture(),store=f.windows.restorationWorkspace
        var policy=f.controller.settings;policy.clients.append(.init(id:"other",name:"Other",actions:Set(MCPAction.allCases)));try f.controller.save(policy)
        let preview=try await applicationPreview(f,group:"appearance",values:["mode":.string("system"),"palette":.string("rose"),"transparency":.string("subtle")])
        do {_=try await applicationCall(f,"application_settings_apply",["proposalID":try XCTUnwrap(preview["proposalID"])],client:"other");XCTFail("Cross-client preview consumed")}
        catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"application_settings_proposal_unavailable")}
        let args:[String:JSONValue]=["proposalID":try XCTUnwrap(preview["proposalID"]),"requestID":.string(UUID().uuidString)]
        _=try await applicationCall(f,"application_settings_apply",args)
        var conflict=args;conflict["proposalID"] = .string(UUID().uuidString)
        do {_=try await applicationCall(f,"application_settings_apply",conflict);XCTFail("Conflicting UUID admitted")}
        catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"request_id_conflict")}
        let before=try store.appPreferenceStore.load()
        for (group,values) in [("appearance",["mode":JSONValue.string("rainbow"),"palette":.string("rose"),"transparency":.string("opaque")]),("execution",["maximumRunSeconds":120,"cpuThreads":2,"maximumResidentBytes":.bool(true)]),("feedback",["showStatusIcon":true,"notifyCompleted":false,"notifyFailed":false,"requestPermission":true])] {
            do {_=try await applicationPreview(f,group:group,values:values);XCTFail("Invalid typed payload admitted")}
            catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"application_settings_invalid_values")}
            XCTAssertEqual(try store.appPreferenceStore.load(),before)
        }
        let discovery=try await f.gateway.dispatch(clientID:f.client,action:"tools_list",arguments:[:])
        guard case .array(let tools)=discovery["tools"],case .object(let tool)=tools.first(where:{if case .object(let t)=$0{return t["name"] == .string("application_settings_preview")};return false}),case .object(let schema)=tool["inputSchema"],case .object(let fields)=schema["properties"] else {return XCTFail("Missing application schema")}
        XCTAssertNil(fields["projectID"]);XCTAssertNotNil(fields["values"])
    }
    func testApplicationSettingsRequireIndependentGrantsAndRejectProjectScope()async throws {
        let f=try fixture()
        var settings=f.controller.settings
        settings.clients[0].actions=Set(settings.clients[0].actions.filter{!$0.rawValue.hasPrefix("application_settings_")})
        try f.controller.save(MCPSettings.decode(settings.encoded()))
        for name in ["application_settings_get","application_settings_preview","application_settings_apply"] {
            do {_=try await applicationCall(f,name);XCTFail("Legacy/project grants conferred global access")}
            catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"not_permitted")}
        }
        settings.clients[0].actions=Set(MCPAction.allCases);settings.projects=[:];try f.controller.save(settings)
        _=try await applicationCall(f,"application_settings_get")
        do {_=try await applicationCall(f,"application_settings_get",["projectID":.string(f.owner.projectID)]);XCTFail("Application action accepted misleading project scope")}
        catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"invalid_arguments")}
    }
    func testApplicationSettingsStaleResetInvalidAndRevocationRefusePartialWrites()async throws {
        let f=try fixture(),store=f.windows.restorationWorkspace
        let preview=try await applicationPreview(f,group:"canvas",values:["zoomSensitivity":0.5,"restoreAtLaunch":true])
        var changed=try store.appPreferenceStore.load();changed.appearance.mode = .light;try store.appPreferenceStore.save(changed)
        do {_=try await applicationCall(f,"application_settings_apply",["proposalID":try XCTUnwrap(preview["proposalID"])]);XCTFail("Stale application proposal applied")}
        catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"application_settings_stale")}
        XCTAssertEqual(try store.appPreferenceStore.load(),changed)
        for values:[String:JSONValue] in [["zoomSensitivity":0.5,"restoreAtLaunch":true,"path":.string("secret")],["zoomSensitivity":0,"restoreAtLaunch":true],["zoomSensitivity":1,"restoreAtLaunch":.string("yes")]] {
            do {_=try await applicationPreview(f,group:"canvas",values:values);XCTFail("Invalid group admitted")}
            catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"application_settings_invalid_values")}
            XCTAssertEqual(try store.appPreferenceStore.load(),changed)
        }
        let reset=try await applicationPreview(f,group:"canvas",reset:true)
        let applied=try await applicationCall(f,"application_settings_apply",["proposalID":try XCTUnwrap(reset["proposalID"]),"requestID":.string(UUID().uuidString)])
        XCTAssertEqual(applied["applied"],true);XCTAssertEqual(try store.appPreferenceStore.load().appearance,changed.appearance)
        let latest=try await applicationPreview(f,group:"canvas",values:["zoomSensitivity":0.5,"restoreAtLaunch":false])
        var settings=f.controller.settings;settings.clients[0].actions=Set(settings.clients[0].actions.filter{$0.rawValue != "application_settings_apply"});try f.controller.save(settings)
        do {_=try await applicationCall(f,"application_settings_apply",["proposalID":try XCTUnwrap(latest["proposalID"])]);XCTFail("Revoked global write admitted")}
        catch let failure as MCPSettings.Refusal {XCTAssertEqual(failure.code,"not_permitted")}
    }
    private func settingsCall(_ f:Fixture,_ action:String,_ extra:[String:JSONValue]=[:],client:String?=nil)async throws->[String:JSONValue] {
        var args=extra;args["projectID"] = .string(f.owner.projectID)
        if action=="settings_apply",args["requestID"]==nil {args["requestID"] = .string(UUID().uuidString)}
        return try await f.gateway.dispatch(clientID:client ?? f.client,action:action,arguments:args)
    }
    private func settingsPreview(_ f:Fixture,group:String="execution",values:[String:JSONValue]?=nil,reset:Bool=false)async throws->[String:JSONValue] {
        let current=try await settingsCall(f,"settings_get")
        var args:[String:JSONValue]=["settingsRevision":try XCTUnwrap(current["settingsRevision"]),"group":.string(group),"reset":.bool(reset)]
        if !reset {args["values"] = .object(values ?? ["maximumRunSeconds":.null,"cpuThreads":3,"memory":.object(["mode":.string("inherit")])])}
        return try await settingsCall(f,"settings_preview",args)
    }
    func testSettingsPreviewApplyAndIdempotentRetryAreSeparateFromGraphAndValueAccess()async throws {
        let f=try fixture(),store=try XCTUnwrap(f.windows.openControllers.first?.session.workspace)
        var policy=f.controller.settings;policy.projects[f.owner.projectID]?.readCapturedValues=false;try f.controller.save(policy)
        let before=try store.projectSettingsSnapshot(projectID:f.owner.projectID),graph=f.owner.document
        let preview=try await settingsPreview(f)
        XCTAssertEqual(try store.projectSettingsSnapshot(projectID:f.owner.projectID).revision,before.revision)
        let request=UUID().uuidString,args:[String:JSONValue]=["proposalID":try XCTUnwrap(preview["proposalID"]),"requestID":.string(request)]
        let applied=try await settingsCall(f,"settings_apply",args)
        XCTAssertEqual(applied["applied"],true)
        let repeated=try await settingsCall(f,"settings_apply",args);XCTAssertEqual(repeated,applied)
        let after=try store.projectSettingsSnapshot(projectID:f.owner.projectID)
        XCTAssertEqual(after.overrides.cpuThreads,3);XCTAssertNotEqual(after.revision,before.revision);XCTAssertEqual(f.owner.document,graph)
        XCTAssertFalse(f.controller.settings.projects[f.owner.projectID]?.readCapturedValues ?? true)
        let fresh=MCPWorkflowGateway(controller:f.controller,windows:f.windows)
        var replay=args;replay["projectID"] = .string(f.owner.projectID)
        let restored=try await fresh.dispatch(clientID:f.client,action:"settings_apply",arguments:replay);XCTAssertEqual(restored,applied,"Retry survives gateway recreation")
    }
    func testSettingsCaptureGroupsAreTypedAndCannotGrantCapturedReads()async throws {
        let f=try fixture(),store=try XCTUnwrap(f.windows.openControllers.first?.session.workspace)
        var policy=f.controller.settings;policy.projects[f.owner.projectID]?.readCapturedValues=false;try f.controller.save(policy)
        for (group,values) in [("trainingCapture",["activationValues":JSONValue.bool(false),"predictionRows":false,"performanceMeasurements":false,"tensorHeatmaps":false,"heatmapFrameLimit":0]),("derivativeCapture",["values":JSONValue.bool(true)])] {
            let proposal=try await settingsPreview(f,group:group,values:values)
            _=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(proposal["proposalID"])])
        }
        let state=try store.projectSettingsSnapshot(projectID:f.owner.projectID)
        XCTAssertEqual(state.overrides.sampleCapture?.activationValues,false);XCTAssertEqual(state.overrides.derivativeCapture?.values,true)
        XCTAssertThrowsError(try f.controller.settings.authorize(clientID:f.client,projectID:f.owner.projectID,action:.resultsRead,requiresValues:true))
        let before=state.revision
        do {_=try await settingsPreview(f,group:"trainingCapture",values:["activationValues":true,"predictionRows":true,"performanceMeasurements":false,"tensorHeatmaps":false,"heatmapFrameLimit":2]);XCTFail("Frames without heatmaps admitted")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"settings_invalid_values")}
        XCTAssertEqual(try store.projectSettingsSnapshot(projectID:f.owner.projectID).revision,before)
    }
    func testSettingsBusyApplyAndConflictingRequestRefuseWithoutMutation()async throws {
        let f=try fixture(),store=try XCTUnwrap(f.windows.openControllers.first?.session.workspace)
        let preview=try await settingsPreview(f),before=try store.projectSettingsSnapshot(projectID:f.owner.projectID)
        store.isInspectingData=true
        do {_=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(preview["proposalID"])]);XCTFail("Busy settings changed")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"settings_busy")}
        store.isInspectingData=false
        XCTAssertEqual(try store.projectSettingsSnapshot(projectID:f.owner.projectID).revision,before.revision)
        let request=UUID().uuidString
        _=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(preview["proposalID"]),"requestID":.string(request)])
        do {_=try await settingsCall(f,"settings_apply",["proposalID":.string(UUID().uuidString),"requestID":.string(request)]);XCTFail("Conflicting request accepted")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"request_id_conflict")}
        var policy=f.controller.settings;policy.clients[0].actions=Set(MCPAction.allCases.filter{$0.rawValue != "settings_preview"});try f.controller.save(policy)
        do {_=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(preview["proposalID"]),"requestID":.string(request)]);XCTFail("Revoked preview grant recovered apply receipt")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
    }
    func testSettingsLegacyGrantsAndProjectClientIntersectionDenyNewActions()async throws {
        let f=try fixture()
        var policy=f.controller.settings
        let old=Set(MCPAction.allCases.filter{!$0.rawValue.hasPrefix("settings_")})
        policy.clients[0].actions=old;policy.projects[f.owner.projectID]?.actions=old
        let restored=try MCPSettings.decode(policy.encoded());try f.controller.save(restored)
        for action in ["settings_get","settings_preview","settings_apply"] {
            do {_=try await settingsCall(f,action);XCTFail("Legacy client gained settings")}
            catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
        }
        policy.clients[0].actions=Set(MCPAction.allCases);try f.controller.save(policy)
        do {_=try await settingsCall(f,"settings_get");XCTFail("Client grant bypassed project")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
    }
    func testSettingsProposalsRefuseStaleGlobalOrProjectStateWithoutGraphChange()async throws {
        for global in [false,true] {
            let f=try fixture(),store=try XCTUnwrap(f.windows.openControllers.first?.session.workspace)
            let proposal=try await settingsPreview(f),graph=f.owner.document
            if global {XCTAssertTrue(store.setExecutionPreferences(try .init(cpuThreads:2)))}
            else {let baseline=try store.projectSettingsSnapshot(projectID:f.owner.projectID);_=try store.applyProjectSettings(store.previewProjectSettings(.derivativeCapture(.init(values:true)),baseline:baseline))}
            let current=try store.projectSettingsSnapshot(projectID:f.owner.projectID)
            do {_=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(proposal["proposalID"])]);XCTFail("Stale proposal applied")}
            catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"settings_stale")}
            XCTAssertEqual(try store.projectSettingsSnapshot(projectID:f.owner.projectID).revision,current.revision);XCTAssertEqual(f.owner.document,graph)
        }
    }
    func testSettingsApplyReauthorizesAndBindsProposalToClient()async throws {
        let f=try fixture(),proposal=try await settingsPreview(f)
        var policy=f.controller.settings;policy.clients.append(.init(id:"other",name:"Other",actions:Set(MCPAction.allCases)));try f.controller.save(policy)
        do {_=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(proposal["proposalID"])],client:"other");XCTFail("Cross-client proposal applied")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"settings_proposal_unavailable")}
        policy.projects[f.owner.projectID]?.actions=Set(MCPAction.allCases.filter{$0.rawValue != "settings_apply"});try f.controller.save(policy)
        do {_=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(proposal["proposalID"])]);XCTFail("Revoked grant applied")}
        catch let error as MCPSettings.Refusal {XCTAssertEqual(error.code,"not_permitted")}
    }
    func testSettingsStrictFieldsAndResetNeverPartiallyWriteOtherGroups()async throws {
        let f=try fixture(),store=try XCTUnwrap(f.windows.openControllers.first?.session.workspace)
        let initial=try store.projectSettingsSnapshot(projectID:f.owner.projectID)
        for values:[String:JSONValue] in [["cpuThreads":3,"path":.string("/private")],["maximumRunSeconds":1,"cpuThreads":999,"memory":.object(["mode":.string("inherit")])],
            ["maximumRunSeconds":1,"cpuThreads":2,"memory":.object(["mode":.string("inherit"),"maximumResidentBytes":134217728])],
            ["maximumRunSeconds":1,"cpuThreads":2,"memory":.object(["mode":.string("limit"),"maximumResidentBytes":true])],
            ["maximumRunSeconds":1,"cpuThreads":2,"memory":.object(["mode":.string("limit"),"maximumResidentBytes":134217728,"path":.string("/private")])]] {
            do {_=try await settingsPreview(f,values:values);XCTFail("Invalid settings admitted")}catch{}
            XCTAssertEqual(try store.projectSettingsSnapshot(projectID:f.owner.projectID).revision,initial.revision)
        }
        let history=try await settingsPreview(f,group:"historyCapture",values:["metricsAndDiagnostics":false,"componentOutputs":false])
        _=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(history["proposalID"])])
        let resource=try await settingsPreview(f);_=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(resource["proposalID"])])
        let reset=try await settingsPreview(f,reset:true);_=try await settingsCall(f,"settings_apply",["proposalID":try XCTUnwrap(reset["proposalID"])])
        let after=try store.projectSettingsSnapshot(projectID:f.owner.projectID);XCTAssertNil(after.overrides.cpuThreads);XCTAssertEqual(after.history,.none)
        let read=try await settingsCall(f,"settings_get")
        guard case .object(let groups)=read["groups"],case .object(let training)=groups["trainingCapture"],case .object(let requested)=training["resolvedPreference"],case .object(let optional)=training["optionalCaptureAfterHistoryPolicy"] else {return XCTFail("Missing capture policy distinction")}
        XCTAssertEqual(requested["activationValues"],true);XCTAssertEqual(optional["activationValues"],false);XCTAssertEqual(optional["predictionRows"],false)
        XCTAssertNil(training["effective"],"Unadmitted preferences must not claim exact runtime capture")
    }
    func testSaveAndReceiptDiscoveryExplainsExplicitExport()async throws {
        let f=try fixture(),response=try await f.gateway.dispatch(clientID:f.client,action:"tools_list",arguments:[:])
        guard case .array(let tools)=response["tools"] else{return XCTFail("Missing tools")}
        for (name,phrase) in [("run_start","never retrains"),("results_read","exports.model")] {
            guard case .object(let tool)=tools.first(where:{if case .object(let v)=$0{return v["name"] == .string(name)};return false}) else{return XCTFail(name)}
            XCTAssertTrue(tool["description"]?.stringValue?.contains(phrase)==true)
        }
    }
    func testExportReceiptProjectionNeverDisclosesPathsOrClaimsFreshVerification()throws {
        let result=MCPWorkflowGateway.tableWriteReceipt(["nodeID":.string("output"),"status":.string("saved"),"path":.string("/private/customer.csv"),"rowCount":741,"bytes":99,"sha256":.string(String(repeating:"a",count:64)),"completeSelectedTable":true,"privateExtra":.string("secret")])
        XCTAssertEqual(result["rowCount"],741);XCTAssertNil(result["path"]);XCTAssertNil(result["privateExtra"])
        XCTAssertEqual(result["evidence"],.string("retained native write receipt; not a fresh file verification"))
    }
    func testInspectionReportsConfiguredKMeansPortsInsteadOfPaletteDefaults()async throws {
        let f=try fixture(),catalog=try await call(f,.catalogList)
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let r)=$0{return r["familyID"] == .string("workflow.logisticRegression")};return false}) else{return XCTFail("Missing Prebuilt models")}
        let address:JSONValue = .object(["domain":.string("control"),"localID":.string("cluster")])
        let config:JSONValue = .object(["task":.string("prebuilt"),"backendID":.string("sklearn"),"algorithm":.string("kmeans"),"operation":.string("fit"),"featureColumns":.array(["x","y"].map(JSONValue.string)),"targetColumn":.string(""),"settings":.object(["clusters":4,"seed":42,"initializations":10,"maxIterations":200,"tolerance":0.0001])])
        let proposal=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([
            .object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("cluster")]),
            .object(["operation":.string("configure"),"address":address,"configuration":config])])])
        _=try await call(f,.patchApply,["proposalID":proposal["proposalID"]!])
        let inspected=try await call(f,.workflowInspect)
        guard case .array(let nodes)=inspected["nodePorts"],case .object(let ports)=nodes.first,
              case .array(let outputs)=ports["outputs"] else{return XCTFail("Configured ports unavailable")}
        let ids=outputs.compactMap{if case .object(let p)=$0{return p["id"]?.stringValue};return nil}
        XCTAssertTrue(ids.contains("profiles"));XCTAssertFalse(ids.contains("confusion"))
        XCTAssertEqual(ports["address"],address)
    }
    func testUnknownValidationDescriptionsStayPrivate() {
        XCTAssertNil(MCPWorkflowGateway.safeConfigurationDiagnostic(.init("/private/token=secret")))
        XCTAssertNil(MCPWorkflowGateway.safeConfigurationDiagnostic(.init("Choose a supported File Input unknownFiles. /private/token=secret")))
    }
    func testOCRConfigurationPreviewReturnsOnlySafeFieldGuidance()async throws {
        let f=try fixture()
        let source=WorkflowNodeAddress(domain:.data,localID:"source"),media=WorkflowNodeAddress(domain:.data,localID:"media")
        let sourceConfig=FileInputConfiguration.newDefaults,mediaConfig=try PrepareMediaConfiguration.defaults()
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:source,payload:.dataOperator(operatorID:DataFlowSourceOperators.fileInput,config:sourceConfig))),lease:f.owner.lease()))
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:media,payload:.dataOperator(operatorID:PrepareMediaConfiguration.operatorID,config:mediaConfig))),lease:f.owner.lease()))
        let before=f.owner.document
        var badSource=sourceConfig;badSource["unknownFiles"] = .string("/private/do-not-echo-token")
        var badMedia=mediaConfig;badMedia["matte"] = .array([255,255,255])
        for (address,config,field) in [(source,badSource,"unknownFiles"),(media,badMedia,"matte")] {
            let result=try await call(f,.patchPreview,["revision":.integer(Int64(before.revision)),"operations":.array([.object(["operation":.string("configure"),"address":.object(["domain":.string("data"),"localID":.string(address.localID)]),"configuration":.object(config)])])])
            XCTAssertEqual(result["code"],.string("invalid_configuration"))
            XCTAssertEqual(result["isError"],true)
            guard case .array(let fields)=(result["fields"] ?? result["candidateFields"]) else {return XCTFail("Missing fixed field guidance")}
            XCTAssertTrue(fields.contains(.string(field)))
            XCTAssertNil(result["proposalID"])
            XCTAssertFalse(String(data:try JSONEncoder().encode(result),encoding:.utf8)!.contains("do-not-echo"))
            XCTAssertEqual(f.owner.document,before)
            XCTAssertFalse(f.owner.isBusy)
        }
    }
    func testMalformedPatchReturnsSafeIndexedGuidanceWithoutMutation()async throws {
        let f=try fixture(),before=f.owner.document
        let result:[String:JSONValue]
        do {
            _=try await call(f,.patchPreview,["revision":.integer(Int64(before.revision)),"operations":.array([.object(["op":.string("insert"),"componentID":.string("private-path-must-not-echo")])])])
            return XCTFail("Malformed patch admitted")
        } catch let error as MCPWorkflowGateway.PatchDecodingFailure { result=error.response }
        XCTAssertEqual(result["isError"],.bool(true))
        XCTAssertEqual(result["operationIndex"],0)
        XCTAssertTrue(result["message"]?.stringValue?.contains("'operation'") == true)
        XCTAssertFalse(String(data:try JSONEncoder().encode(result),encoding:.utf8)!.contains("private-path-must-not-echo"))
        XCTAssertEqual(f.owner.document,before)
    }
    func testCatalogRevisionIsStableAcrossIdenticalDiscovery()async throws {
        let f=try fixture(),first=try await call(f,.catalogList)
        for _ in 0..<4 {
            let next=try await call(f,.catalogList)
            XCTAssertEqual(first["components"],next["components"])
            XCTAssertEqual(first["catalogRevision"],next["catalogRevision"])
        }
    }
    func testPatchDiscoveryNamesTheOperationDiscriminator()async throws {
        let f=try fixture()
        let result=try await f.gateway.dispatch(clientID:f.client,action:"tools_list",arguments:[:])
        guard case .array(let tools)=result["tools"],
              case .object(let tool)=tools.first(where:{if case .object(let t)=$0{return t["name"] == .string("patch_preview")};return false}),
              case .object(let schema)=tool["inputSchema"],case .object(let properties)=schema["properties"],
              case .object(let operations)=properties["operations"],case .object(let items)=operations["items"],
              case .object(let fields)=items["properties"],case .array(let required)=items["required"],
              case .object(let discriminator)=fields["operation"],case .array(let kinds)=discriminator["enum"]
        else{return XCTFail("Patch schema must expose directly typed operation fields")}
        XCTAssertTrue(required.contains(.string("operation")))
        XCTAssertNil(items["oneOf"])
        XCTAssertNil(fields["op"])
        XCTAssertGreaterThan(kinds.count,10)
        guard case .object(let configuration)=fields["configuration"],let guidance=configuration["description"]?.stringValue else {return XCTFail("Missing replacement semantics")}
        XCTAssertTrue(guidance.contains("FULL REPLACEMENT"))
        XCTAssertTrue(guidance.contains("root nodes and container children"))
        XCTAssertTrue(guidance.contains("workflow_inspect"))
        XCTAssertNotNil(fields["from"])
        let catalog=try await catalogWithDetails(f,families:["workflow.train"])
        XCTAssertEqual(catalog["patchOperationSchema"],.object(items))
    }
    func testFineTuningDiscoveryUsesTheSameExpandedSchemaAndFiniteRetention()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["workflow.train"])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let value)=$0{return value["familyID"] == .string("workflow.train")};return false}),case .object(let task)=row["fineTuning"]else{return XCTFail("Fine-tuning absent from shared Train discovery")}
        XCTAssertEqual(task["schema"],FineTuningConfiguration.configurationSchema)
        XCTAssertEqual(row["lessonID"],.string("train-data-access"))
        XCTAssertEqual(row["relatedLessonIDs"],.array(ComponentLessonDestination.relatedEntryIDs(type:"train").map(JSONValue.string)))
        guard case .object(let schema)=task["schema"],case .array(let branches)=schema["oneOf"],branches.count==2,
              case .object(let causal)=branches[0],case .object(let fields)=causal["properties"],
              case .object(let retention)=fields["checkpointCount"]else{return XCTFail("Missing finite causal retention")}
        XCTAssertEqual(branches[1],TypedDecisionConfiguration.trainingSchema)
        guard case .object(let typed)=branches[1],case .object(let typedFields)=typed["properties"] else{return XCTFail("Missing typed-decision fields")}
        for key in ["stateColumns","targetColumns","groupColumn","calibrationFraction","targetError","mode"] {XCTAssertNotNil(typedFields[key],key)}
        XCTAssertEqual(retention["maximum"],10);XCTAssertEqual(retention["default"],0)
        for key in ["exampleFormat","promptColumn","responseColumn","lossCoverage","startMode","resumeRecord","activationCheckpointing"]{XCTAssertNotNil(fields[key])}
    }
    func testEditableMoEDiscoveryAndPlacementUseCanonicalGraphs()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["mixture_experts"])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let r)=$0{return r["routedExperts"] != nil};return false}) else{return XCTFail("MoE absent from shared registry discovery")}
        let proposal=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("mixture")])])])
        XCTAssertEqual(row["lessonID"],.string("mixture-experts"))
        _=try await call(f,.patchApply,["proposalID":proposal["proposalID"]!])
        guard case .modelPrimitive("mixture_experts",2,let values)=f.owner.document.nodes.first?.payload else{return XCTFail("Incorrect MoE placement")}
        XCTAssertEqual(try MixtureExpertsConfiguration(values:values,registry:f.owner.modelRegistry).parameterCount,180)
        let source=f.owner.document,root=WorkflowNodeAddress(domain:.model,localID:"mixture")
        let encoder=JSONEncoder()
        let rootJSON=try JSONDecoder().decode(JSONValue.self,from:encoder.encode(root))
        let edit:[String:JSONValue]=["operation":.string("configure"),"container":rootJSON,"childID":.string("expert_a/hidden"),"configuration":.object(["in_features":2,"out_features":8,"bias":false])]
        let child=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(edit)])])
        _=try await call(f,.patchApply,["proposalID":child["proposalID"]!])
        guard case .modelPrimitive(_,_,let changed)=f.owner.document.nodes.first?.payload else{return XCTFail()}
        XCTAssertEqual(try MixtureExpertsConfiguration(values:changed,registry:f.owner.modelRegistry).experts[0].graph.nodes.first{$0.id=="hidden"}?.config["bias"],false)
        _=try await call(f,.patchUndo)
        XCTAssertTrue(f.owner.document.matchesCapturedConstruction(source))
        var settings=f.controller.settings;settings.disabledComponents.insert("kwip.linear");try f.controller.save(settings)
        do{_=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(edit)])]);XCTFail("Container must not bypass blocked layer families")}
        catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"not_permitted")}
    }
    func testPrebuiltModelsDiscoveryIsOneCanonicalTool()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["workflow.logisticRegression"])
        guard case .array(let entries)=catalog["components"] else{return XCTFail("Missing catalog")}
        let models=entries.compactMap {if case .object(let row)=$0,row["prebuiltModels"] != nil{return row};return nil}
        XCTAssertEqual(models.count,1)
        XCTAssertEqual(models.first?["title"],.string("Prebuilt models"))
        XCTAssertEqual(models.first?["prebuiltModels"],PrebuiltModelConfiguration.catalogJSON)
        XCTAssertEqual(models.first?["lessonID"],.string("prebuilt-models"))
        XCTAssertFalse(entries.contains {if case .object(let row)=$0{return row["title"] == .string("Keras classifier")};return false})
    }
    func testPrebuiltFamilyRestrictionAlsoCoversKeras()async throws {
        let f=try fixture(),a=WorkflowNodeAddress(domain:.control,localID:"keras")
        let config:[String:JSONValue]=["backendID":.string("tensorflow"),"featureColumns":.array([.string("a"),.string("b")]),"targetColumn":.string("y"),"settings":.object(TensorFlowEstimatorConfiguration.defaultSettings)]
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:a,payload:.estimator(config:config))),lease:f.owner.lease()))
        for blocked in ["workflow.logisticRegression","workflow.kerasClassifier"] {
            var settings=f.controller.settings;settings.disabledComponents=[blocked];try f.controller.save(settings)
            do {_=try await call(f,.workflowInspect);XCTFail("Unified tool must preserve blocked families")}catch {XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"not_permitted")}
        }
    }
    func testPrebuiltExplicitTerminalValidationUsesEstimatorCompiler()async throws {
        let f=try fixture()
        let doc=try PrebuiltModelsPractice.document(projectID:f.owner.projectID)
        // This complete synthetic lesson contains Save model; preserve the production
        // default denial and explicitly grant that family only in this fixture.
        var settings=f.controller.settings;settings.disabledComponents.remove("workflow.saveModel");try f.controller.save(settings)
        let commands=doc.nodes.map(WorkflowCommand.insert)+doc.edges.map{WorkflowCommand.connect(from:$0.from,to:$0.to)}
        XCTAssertNotNil(f.owner.applyBatch(commands,lease:f.owner.lease()))
        let endpoint=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(PrebuiltModelsPractice.estimator))
        let result=try await call(f,.workflowValidate,["terminal":endpoint]);XCTAssertEqual(result["valid"],true)
    }
    func testDocumentTaskDiscoveryUsesCanonicalDefaults()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["workflow.logisticRegression"])
        guard case .array(let entries)=catalog["components"] else {return XCTFail("Missing catalog")}
        let descriptions=entries.compactMap {if case .object(let row)=$0,case .object(let task)=row["documentClassification"] {return task};return nil}
        XCTAssertFalse(descriptions.isEmpty)
        for task in descriptions {
            XCTAssertEqual(task["defaultConfiguration"],.object(DocumentClassificationConfiguration.defaults))
            XCTAssertEqual(task["featureSettings"],.object(DocumentClassificationConfiguration.featureDefaults))
            XCTAssertEqual(task["operations"],.array(["fit","predict","evaluate"].map(JSONValue.string)))
        }
    }
    func testAudioClassificationDiscoveryAndPlacementUseSharedRegistry()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:[AudioClassificationConfiguration.operatorID])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let entry)=$0 {return entry["familyID"] == .string(AudioClassificationConfiguration.operatorID)};return false}) else {return XCTFail("Audio classification absent from discovery")}
        let definition=try XCTUnwrap(f.owner.dataRegistry.operatorDefinition(id:AudioClassificationConfiguration.operatorID))
        XCTAssertEqual(row["configurationSchema"],definition.configSchema)
        let proposal=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("audio-classification")])])])
        _=try await call(f,.patchApply,["proposalID":proposal["proposalID"]!])
        guard case .dataOperator(let id,let config)=f.owner.document.nodes.first?.payload else {return XCTFail("Incorrect audio classification placement")}
        XCTAssertEqual(id,AudioClassificationConfiguration.operatorID);XCTAssertEqual(config,definition.defaultConfig)
    }
    func testOCRDiscoveryAndPlacementUseTheSharedRegistry()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:[OCRConfiguration.operatorID])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let entry)=$0 {return entry["familyID"] == .string(OCRConfiguration.operatorID)};return false}) else {return XCTFail("OCR missing from canonical discovery")}
        let definition=try XCTUnwrap(f.owner.dataRegistry.operatorDefinition(id:OCRConfiguration.operatorID))
        XCTAssertEqual(row["configurationSchema"],definition.configSchema)
        let proposal=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("ocr")])])])
        _=try await call(f,.patchApply,["proposalID":proposal["proposalID"]!])
        guard case .dataOperator(let id,let config)=f.owner.document.nodes.first?.payload else {return XCTFail("Incorrect OCR placement")}
        XCTAssertEqual(id,OCRConfiguration.operatorID)
        XCTAssertEqual(config,try OCRConfiguration.defaults())
    }
    func testTrainDiscoveryExposesSharedInlineSelectionPolicy()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["workflow.train"])
        guard case .array(let entries)=catalog["components"],case .object(let train)=entries.first(where:{if case .object(let row)=$0 {return row["familyID"] == .string("workflow.train")};return false}),case .object(let controls)=train["inlineRunControls"] else {return XCTFail("Train controls missing from discovery")}
        XCTAssertEqual(controls["configurationPath"],.string("runSetup.trainingControl"))
        XCTAssertEqual(controls["schema"],TrainingControlPolicy.configurationSchema)
        let policy=try JSONDecoder().decode(TrainingControlPolicy.self,from:JSONEncoder().encode(controls["default"]))
        XCTAssertEqual(policy,.factory)
    }
    func testAtomicPatchAndUndoRefusesInterveningManualEdit()async throws {
        let f=try fixture(),before=f.owner.document
        let catalog=try await call(f,.catalogList)
        guard case .array(let entries)=catalog["components"],let view=entries.first(where:{if case .object(let row)=$0{return row["familyID"] == .string("kwip.data.view")};return false}),case .object(let row)=view else{return XCTFail("View absent from canonical catalog")}
        let inserted:[String:JSONValue]=["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("view"),"x":.integer(200),"y":.integer(80)]
        let preview=try await call(f,.patchPreview,["revision":.integer(Int64(before.revision)),"operations":.array([.object(inserted)])])
        XCTAssertEqual(f.owner.document,before,"Preview must never change persistence or selection")
        _=try await call(f,.patchApply,["proposalID":preview["proposalID"]!])
        XCTAssertEqual(f.owner.document.nodes.count,1);XCTAssertTrue(f.owner.canUndo)
        _=try await call(f,.patchUndo);XCTAssertTrue(f.owner.document.nodes.isEmpty)
        let second=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(inserted)])])
        _=try await call(f,.patchApply,["proposalID":second["proposalID"]!])
        XCTAssertTrue(f.owner.rename("Owner changed project",lease:f.owner.lease()))
        do{_=try await call(f,.patchUndo);XCTFail("Must not undo intervening owner edit")}catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"undo_unavailable")}
        XCTAssertEqual(f.owner.document.name,"Owner changed project")
    }
    func testInvalidPlacementDoesNotMutateAndRevocationBlocksInspection()async throws {
        let f=try fixture(),before=f.owner.document
        do {
            _=try await call(f,.patchPreview,["revision":.integer(Int64(before.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":.string("not-a-tool")])])])
            XCTFail("Unknown component admitted")
        }catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"component_unavailable")}
        XCTAssertEqual(f.owner.document,before)
        var policy=f.controller.settings;policy.projects.removeAll();try f.controller.save(policy)
        do{_=try await call(f,.workflowInspect);XCTFail("Revoked project exposed")}catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"not_permitted")}
    }
    func testIdenticalMutationRetrySurvivesGatewayRecreation()async throws {
        let f=try fixture(),catalog=try await call(f,.catalogList)
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let r)=$0{return r["familyID"] == .string("kwip.data.view")};return false})else{return XCTFail("Missing View")}
        let preview=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("view")])])])
        let request=UUID().uuidString,args:[String:JSONValue]=["projectID":.string(f.owner.projectID),"proposalID":preview["proposalID"]!,"requestID":.string(request)]
        let first=try await f.gateway.dispatch(clientID:f.client,action:"patch_apply",arguments:args)
        let recreated=MCPWorkflowGateway(controller:f.controller,windows:f.windows)
        let retry=try await recreated.dispatch(clientID:f.client,action:"patch_apply",arguments:args)
        XCTAssertEqual(first,retry);XCTAssertEqual(f.owner.document.nodes.count,1)
        var collision=args;collision["proposalID"] = .string(UUID().uuidString)
        do{_=try await recreated.dispatch(clientID:f.client,action:"patch_apply",arguments:collision);XCTFail("Reused key with changed request admitted")}catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"request_id_conflict")}
    }
    func testDisabledFamilyCannotBeBypassedByPrimitiveVariant()async throws {
        let f=try fixture(),catalog=try await call(f,.catalogList)
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let r)=$0{return r["familyID"] == .string("kwip.activation")};return false})else{return XCTFail("Missing Activation")}
        let preview=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("activation")])])])
        _=try await call(f,.patchApply,["proposalID":preview["proposalID"]!])
        var policy=f.controller.settings;policy.disabledComponents.insert("kwip.activation");try f.controller.save(policy)
        do{_=try await call(f,.workflowInspect);XCTFail("Existing primitive bypassed blocked family")}catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"not_permitted")}
    }
    func testProjectIsolationAndDisabledAction()async throws {
        let f=try fixture(),other=try f.windows.newFlow(present:false)
        let id=try XCTUnwrap(other.session.workspace.activeProjectID)
        do{_=try await f.gateway.dispatch(clientID:f.client,action:"workflow_inspect",arguments:["projectID":.string(id)]);XCTFail("Unshared project leaked")}catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"not_permitted")}
        var policy=f.controller.settings;policy.disabledActions.insert(.patchPreview);try f.controller.save(policy)
        do{_=try await call(f,.patchPreview);XCTFail("Disabled action admitted")}catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"not_permitted")}
    }
    func testNoOpMCPPatchCannotClaimUndoOfManualEdit()async throws {
        let f=try fixture(),address=WorkflowNodeAddress(domain:.data,localID:"manual-view")
        XCTAssertNotNil(f.owner.apply(.insert(.init(address:address,name:"My view",payload:.dataOperator(operatorID:"kwip.data.view",config:[:]))),lease:f.owner.lease()))
        let before=f.owner.document
        let preview=try await call(f,.patchPreview,["revision":.integer(Int64(before.revision)),"operations":.array([.object(["operation":.string("rename"),"address":.object(["domain":.string("data"),"localID":.string(address.localID)]),"name":.string("My view")])])])
        _=try await call(f,.patchApply,["proposalID":preview["proposalID"]!])
        do{_=try await call(f,.patchUndo);XCTFail("A no-op cannot borrow the previous manual edit's Undo authority")}catch{XCTAssertEqual((error as? MCPSettings.Refusal)?.code,"undo_unavailable")}
        XCTAssertEqual(f.owner.document,before)
    }

}

extension MCPWorkflowGatewayTests {
    func testMediaDiscoveryUsesSharedRegistrySchemaLessonAndInputAlternatives()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:[PrepareMediaConfiguration.operatorID,"workflow.modelDefinition","input","output"])
        guard case .array(let entries)=catalog["components"] else {return XCTFail("Missing current native catalog")}
        let media=try XCTUnwrap(entries.compactMap {value->[String:JSONValue]? in guard case .object(let row)=value,row["familyID"] == .string(PrepareMediaConfiguration.operatorID) else {return nil};return row}.first)
        XCTAssertEqual(media["lessonID"],.string(PrepareMediaPracticeExample.exampleID))
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        XCTAssertEqual(media["configurationSchema"],registry.operatorDefinition(id:PrepareMediaConfiguration.operatorID)?.configSchema)
        let definitions=entries.compactMap {value->[JSONValue]? in guard case .object(let row)=value,case .array(let inputs)?=row["modelInputDefinitions"] else {return nil};return inputs}.flatMap{$0}
        XCTAssertTrue(definitions.contains {value in guard case .object(let definition)=value,definition["id"] == .string("input"),case .object(let alternatives)?=definition["configurationAlternatives"] else {return false};return alternatives["tensor"] != nil})
    }
}
extension MCPWorkflowGatewayTests {
    func testGameRuleDiscoveryAndPlacementUseCanonicalDefinition()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:[GameRuleConfiguration.operatorID])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let e)=$0{return e["familyID"] == .string(GameRuleConfiguration.operatorID)};return false}) else{return XCTFail("Game definition missing from shared discovery")}
        let definition=try XCTUnwrap(f.owner.dataRegistry.operatorDefinition(id:GameRuleConfiguration.operatorID))
        XCTAssertEqual(row["configurationSchema"],definition.configSchema)
        let proposal=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("rules")])])])
        _=try await call(f,.patchApply,["proposalID":proposal["proposalID"]!])
        guard case .dataOperator(let id,let config)=f.owner.document.nodes.first?.payload else{return XCTFail("Incorrect game resource placement")}
        XCTAssertEqual(id,GameRuleConfiguration.operatorID)
        XCTAssertEqual(config,definition.defaultConfig)
        XCTAssertEqual(config["definition"],.object(try GameRuleConfiguration.example("take-away")))
    }
}

extension MCPWorkflowGatewayTests {
    func testGameSessionDiscoveryAndPlacementUseSharedConfiguration()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["workflow.gameSession"])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let e)=$0{return e["familyID"] == .string("workflow.gameSession")};return false}) else{return XCTFail("Game session missing from shared discovery")}
        XCTAssertEqual(row["configurationSchema"],GameSessionConfiguration.configurationSchema)
        let proposal=try await call(f,.patchPreview,["revision":.integer(Int64(f.owner.document.revision)),"operations":.array([.object(["operation":.string("insert"),"componentID":row["componentID"]!,"nodeID":.string("session")])])])
        _=try await call(f,.patchApply,["proposalID":proposal["proposalID"]!])
        guard case .gameSession(let config)=f.owner.document.nodes.first?.payload else{return XCTFail("Incorrect session placement")}
        XCTAssertEqual(try GameSessionConfiguration(config).values,try GameSessionConfiguration([:]).values)
        XCTAssertEqual(f.owner.document.schemaVersion,22)
    }
}

extension MCPWorkflowGatewayTests {
    func testSelfPlayDiscoveryUsesSharedTaskConfiguration()async throws {
        let f=try fixture(),catalog=try await catalogWithDetails(f,families:["workflow.train"])
        guard case .array(let entries)=catalog["components"],case .object(let row)=entries.first(where:{if case .object(let e)=$0{return e["familyID"] == .string("workflow.train")};return false}),case .object(let task)=row["selfPlay"]else{return XCTFail("Self-play missing from Train discovery")}
        XCTAssertEqual(task["schema"],SelfPlayConfiguration.configurationSchema)
        XCTAssertEqual(task["defaultConfiguration"],.object(["selfPlay":.object(SelfPlayConfiguration.defaults)]))
        XCTAssertEqual(task["lessonID"],.string(SelfPlayPractice.lessonID))
    }
}


extension MCPWorkflowGatewayTests {
    func testCampaignHistorySuspensionRechecksCurrentAuthorityBeforeAdmission() async throws {
        for change in ["client", "project", "component", "revision", "lease", "busy", "draft"] {
            let f=try fixture(), terminal=WorkflowNodeAddress(domain:.control,localID:"train")
            XCTAssertNotNil(f.owner.apply(.insert(.init(address:terminal,payload:.train(config:[:]))),lease:f.owner.lease()))
            XCTAssertNotNil(f.owner.apply(.configure(address:terminal,config:["selfPlay":.object(SelfPlayConfiguration.defaults)]),lease:f.owner.lease()), f.owner.error ?? "")
            var loaded=false
            let gateway=MCPWorkflowGateway(controller:f.controller,windows:f.windows,campaignHistory:{ _, _ in
                await Task.yield()
                loaded=true
                var policy=f.controller.settings
                switch change {
                case "client": policy.clients.removeAll(); try f.controller.save(policy)
                case "project": policy.projects.removeAll(); try f.controller.save(policy)
                case "component": policy.disabledComponents.insert("workflow.train"); try f.controller.save(policy)
                case "revision": XCTAssertTrue(f.owner.rename("Changed during History",lease:f.owner.lease()))
                case "lease": f.owner.invalidate()
                case "busy": f.owner.setExecutionBusy(true)
                default: XCTAssertTrue(f.owner.setInspectorEditPending(contextID:UUID(),address:terminal,key:"configuration",pending:true,lease:f.owner.lease()))
                }
                // Deliberately no source: authorization/state must be rechecked before
                // even selecting a History record or attempting the native side effect.
                return []
            })
            let address=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(terminal))
            do {
                _=try await gateway.dispatch(clientID:f.client,action:MCPAction.runStart.rawValue,arguments:["projectID":.string(f.owner.projectID),"requestID":.string(UUID().uuidString),"terminal":address,"campaignAction":.string("resume"),"sourceRunID":.string(UUID().uuidString)])
                XCTFail("Changed authority admitted: \(change)")
            } catch {
                let expected=["client","project","component"].contains(change) ? "not_permitted" : ["busy","draft"].contains(change) ? "busy_or_pending_edits" : "stale_run_request"
                XCTAssertEqual((error as? MCPSettings.Refusal)?.code,expected,change)
            }
            XCTAssertTrue(loaded,change)
            XCTAssertFalse(f.windows.openControllers.contains { $0.session.workspace.workflowSession?.hasActiveOperation == true })
            f.owner.setExecutionBusy(false)
        }
    }
}
