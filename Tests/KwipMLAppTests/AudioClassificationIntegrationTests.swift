import XCTest
import AppKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class AudioClassificationIntegrationTests:XCTestCase {
    func testRealCapturedModelWholeFlowAndColdHistory()async throws {
        let env=ProcessInfo.processInfo.environment
        guard let development=env["KWIPML_TEST_DEVELOPMENT_ROOT"],let model=env["KWIPML_AST_FIXTURE"],let audio=env["KWIPML_AUDIO_CLASSIFICATION_WAV"] else {throw XCTSkip("Select the explicit offline AST and PCM WAV fixture.")}
        _=NSApplication.shared
        let checkpoint=env["KWIPML_AST_CHECKPOINT_STORAGE"]
        let root=try checkpoint.map{URL(fileURLWithPath:$0)} ?? canonicalBackendTemporaryDirectory()
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(AudioClassificationPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer{host.cancel();if checkpoint==nil && !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        let address=AudioClassificationPracticeExample.model
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[address]),lease:host.owner.lease()))
        let context=try WorkflowControlNodeContext(owner:host.owner,address:address)
        XCTAssertTrue(context.authority.configure(LoadModelConfiguration(source:.localFolder).values,expected:context.configuration),context.statusMessage)
        let fixture=root.appendingPathComponent("example-sources/"+host.owner.projectID+"/classification-uat.wav")
        try FileManager.default.createDirectory(at:fixture.deletingLastPathComponent(),withIntermediateDirectories:true)
        try FileManager.default.copyItem(at:URL(fileURLWithPath:audio),to:fixture)
        XCTAssertTrue(host.bindings.setPath(fixture.path,for:AudioClassificationPracticeExample.source,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let runtime=try XCTUnwrap(workspace.runtime),policy=try workspace.executionPolicy(for:host.owner.projectID)
        try await host.modelLoader.importFolder(URL(fileURLWithPath:model),address:address,runtime:runtime,policy:policy)
        let reference=try XCTUnwrap(host.modelLoader.reference(address),host.modelLoader.status)
        XCTAssertEqual(reference.metadata["task"],.string("audioClassification"))
        XCTAssertFalse(reference.files.contains{$0.path=="tokenizer.json"})
        XCTAssertTrue(host.owner.select(.init(),lease:host.owner.lease()))
        XCTAssertTrue(host.run(),host.statusMessage)
        let deadline=Date().addingTimeInterval(90)
        while host.hasActiveOperation && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        let session=try XCTUnwrap(host.dataPreparation,host.statusMessage)
        XCTAssertEqual(session.phase,.completed,session.statusMessage+"\n"+session.messages.joined(separator:"\n"))
        XCTAssertFalse(workspace.isBusy)
        await host.dataPreparationHistory.reloadAndWait()
        let entry=try XCTUnwrap(host.dataPreparationHistory.entries.first,host.dataPreparationHistory.status)
        let outputs=try await host.dataPreparationHistory.outputs(for:entry)
        XCTAssertNotNil(outputs[AudioClassificationPracticeExample.classifier])
        let captured=try XCTUnwrap(session.snapshot(for:AudioClassificationPracticeExample.classifier))
        XCTAssertEqual(captured.dataCompleteness,"Complete data")
        XCTAssertTrue(workspace.saveManagedProject(),workspace.statusMessage)
        let id=host.owner.projectID
        XCTAssertTrue(workspace.beginWorkflowProject(id:id,forceReload:true),workspace.statusMessage)
        let reopened=try XCTUnwrap(workspace.workflowSession)
        XCTAssertEqual(reopened.modelLoader.reference(address)?.identity,reference.identity)
        await reopened.dataPreparationHistory.reloadAndWait()
        _=try await reopened.dataPreparationHistory.verify(try XCTUnwrap(reopened.dataPreparationHistory.entries.first))
        if checkpoint != nil {print("AUDIO_CLASSIFICATION_CHECKPOINT_PROJECT="+id)}
    }
}
