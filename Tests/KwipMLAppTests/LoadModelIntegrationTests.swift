import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LoadModelIntegrationTests:XCTestCase {
    func testActualImportTokensExportAndReopenKeepModelIdentity()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_MODEL_FOLDER"] else {throw XCTSkip("Requires the explicit local tiny-model fixture and isolated runtime")}
        let root=try canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),address=WorkflowNodeAddress(domain:.control,localID:"imported-model")
        defer {host.modelLoader.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:address,payload:.frozenModel(config:[:]))),lease:host.owner.lease()))
        let context=try WorkflowControlNodeContext(owner:host.owner,address:address)
        XCTAssertTrue(context.authority.configure(LoadModelConfiguration(source:.localFolder,maximumTokens:512).values,expected:[:]),context.statusMessage)
        let runtime=try XCTUnwrap(workspace.runtime),policy=try workspace.executionPolicy(for:host.owner.projectID)
        try await host.modelLoader.importFolder(URL(fileURLWithPath:fixture),address:address,runtime:runtime,policy:policy)
        let reference=try XCTUnwrap(host.modelLoader.reference(address),host.modelLoader.status)
        XCTAssertEqual(reference.metadata["architecture"],.string("gpt2"))
        XCTAssertFalse(reference.files.isEmpty)
        let op=ModelAssetOperation()
        let packet=try await op.run(action:"tokenize",settings:["texts":.array([.string("hello world")]),"maximumTokens":16,"truncation":.string("reject"),"padding":.string("longest"),"paddingSide":.string("right")],source:reference.directory(root:root),files:reference.files,projectID:host.owner.projectID,root:root,accountingRoot:root,runtime:runtime,policy:policy,isCurrent:{true},onProgress:{_ in})
        let tokens=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:packet.directory.appendingPathComponent("output/tokens.json")))
        XCTAssertEqual(tokens["input_ids"],.array([.array([3,4])]))
        let output=try await ModelAssetOperation().run(action:"export",settings:["shardBytes":1024],source:reference.directory(root:root),files:reference.files,projectID:host.owner.projectID,root:root,accountingRoot:root,runtime:runtime,policy:policy,isCurrent:{true},onProgress:{_ in})
        XCTAssertTrue(output.files.contains{$0.path=="output/model/kwip-manifest.json"})
        let generationConfig:[String:JSONValue]=["promptMode":.string("one"),"prompt":.string("hello world"),"promptColumn":.string("prompt"),"textColumn":.string("text"),"idsColumn":.string("generated_ids"),"countColumn":.string("new_tokens"),"stopColumn":.string("stop_reason"),"settings":.object(["maxNewTokens":3,"maxTotalTokens":3,"stopAtEOS":false])]
        let generated=try await ModelAssetOperation().run(action:"generate",settings:["configuration":.object(generationConfig),"maximumTokens":16,"device":.string("cpu")],source:reference.directory(root:root),files:reference.files,projectID:host.owner.projectID,root:root,accountingRoot:root,runtime:runtime,policy:policy,isCurrent:{true},onProgress:{_ in})
        let table=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:generated.directory.appendingPathComponent("output/table.json")))
        guard case .array(let rows)=table["rows"],case .array(let first)=rows.first else {return XCTFail("Expected a captured generation table")}
        XCTAssertEqual(rows.count,1);XCTAssertEqual(first[3],3)
        let reloaded=try ModelLoaderStore(owner:host.owner,root:root,accountingRoot:root)
        XCTAssertEqual(reloaded.reference(address),reference,"Private snapshot must survive without reopening the original model folder")
        host.modelLoader.preview(address,text:"hello",runtime:runtime,policy:policy)
        XCTAssertTrue(workspace.isBusy,"A source/model task reserves the workspace without blocking canvas selection")
        XCTAssertFalse(host.run(train:WorkflowExamples.trainAddress),"Do not start competing training during a model task")
        host.cancel()
        for _ in 0..<200 {if !host.modelLoader.isBusy {break};try await Task.sleep(for:.milliseconds(10))}
        XCTAssertFalse(host.modelLoader.isBusy);XCTAssertFalse(workspace.isBusy)
        var changed=reference.files
        let file=changed[0];changed[0] = .init(path:file.path,bytes:file.bytes,sha256:String(repeating:"0",count:64))
        do {_=try await ModelAssetOperation().run(action:"inspect",settings:[:],source:reference.directory(root:root),files:changed,projectID:host.owner.projectID,root:root,accountingRoot:root,runtime:runtime,policy:policy,isCurrent:{true},onProgress:{_ in});XCTFail("Changed source pins accepted")} catch {}
        let operationRoot=root.appendingPathComponent("workflow-operations/"+host.owner.projectID)
        for child in try FileManager.default.contentsOfDirectory(at:operationRoot,includingPropertiesForKeys:nil) {
            XCTAssertFalse(FileManager.default.fileExists(atPath:child.appendingPathComponent("source").path),"Terminal model jobs must remove owned source copies")
        }
        let before=reference.identity
        XCTAssertTrue(context.rename("Renamed model"))
        XCTAssertEqual(host.modelLoader.reference(address)?.identity,before,"Presentation edits must not invalidate model data")
        XCTAssertTrue(context.authority.configure(LoadModelConfiguration(source:.localFolder,maximumTokens:16).values,expected:LoadModelConfiguration(source:.localFolder,maximumTokens:512).values))
        XCTAssertEqual(host.modelLoader.reference(address)?.identity,before,"Token limits are live configuration, not model identity")
        // Exercise the real canvas route, including its workspace reservation;
        // direct ModelAssetOperation generation above cannot detect admission bugs.
        let generator=WorkflowNodeAddress(domain:.control,localID:"generate-imported")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:generator,payload:.generateText(config:["prompt":.string("hello world"),"maxNewTokens":3,"maxTotalTokens":3,"device":.string("cpu")]))),lease:host.owner.lease()))
        XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:address,port:"model"),to:.init(address:generator,port:"model")),lease:host.owner.lease()))
        var priorGenerationID:String?
        for cold in [false,true] {
            if cold {XCTAssertTrue(workspace.beginWorkflowProject(id:host.owner.projectID,forceReload:true),workspace.statusMessage)}
            let current=try XCTUnwrap(workspace.workflowSession)
            for _ in 0..<300 where workspace.isBusy || current.owner.isBusy {try await Task.sleep(for:.milliseconds(10))}
            XCTAssertTrue(current.generateText(address:generator),current.statusMessage)
            XCTAssertTrue(workspace.isBusy,"Accepted generation must reserve synchronously before a result reader or another action can claim the workspace")
            for _ in 0..<1500 {if !current.modelLoader.isBusy {break};try await Task.sleep(for:.milliseconds(20))}
            let completed=try XCTUnwrap(current.modelLoader.generationRecord,current.modelLoader.status)
            XCTAssertNotEqual(completed.operationID,priorGenerationID,"Reopening must permit a new generation, not reuse the retained output")
            priorGenerationID=completed.operationID
            XCTAssertFalse(workspace.isBusy,current.modelLoader.status)
            XCTAssertFalse(current.owner.isBusy)
        }
        let current=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(current.generateText(address:generator),current.statusMessage)
        current.cancel()
        for _ in 0..<1500 {if !current.modelLoader.isBusy {break};try await Task.sleep(for:.milliseconds(20))}
        XCTAssertFalse(workspace.isBusy,"Cancellation must release the exact generation reservation")
        XCTAssertFalse(current.owner.isBusy)
    }
    func testNewLoaderSchemaCanStillTrainAndReadExistingHistory()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires isolated runtime")}
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace);XCTAssertTrue(workspace.loadPreparedWorkflowExample(device:"cpu"))
        let host=try XCTUnwrap(workspace.workflowSession),address=WorkflowNodeAddress(domain:.control,localID:"loader")
        XCTAssertTrue(host.canvas.apply(.insert(.init(address:address,payload:.frozenModel(config:LoadModelConfiguration(source:.localFolder).values))),lease:host.owner.lease()))
        XCTAssertEqual(host.owner.document.schemaVersion,20)
        let done=expectation(description:"Existing classifier completes with new loader schema")
        let observer=host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink {_ in done.fulfill()}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        await fulfillment(of:[done],timeout:60);observer.cancel()
        let outcome=try XCTUnwrap(host.outcomes.first);XCTAssertEqual(outcome.status,.completed,outcome.error ?? "")
        let saved=try WorkflowFrozenModelSnapshot.select(.init(projectID:host.owner.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent))),root:root)
        XCTAssertEqual(saved.projectID,host.owner.projectID)
    }
    func testFileSnapshotRejectsSymlinkAndChangedExpectedBytes()throws {
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        for name in ["config.json","tokenizer.json","model.safetensors"] {try Data((name.hasSuffix(".json") ? "{}":"test").utf8).write(to:root.appendingPathComponent(name))}
        let files=try ModelAssetFiles.inventory(root)
        let destination=root.appendingPathComponent("snapshot");try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:false)
        try Data("changed".utf8).write(to:root.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try ModelAssetFiles.copy(files,from:root,to:destination))
        let link=root.appendingPathComponent("tokenizer.json");try FileManager.default.removeItem(at:link)
        try FileManager.default.createSymbolicLink(at:link,withDestinationURL:root.appendingPathComponent("config.json"))
        XCTAssertThrowsError(try ModelAssetFiles.inventory(root))
    }
}
