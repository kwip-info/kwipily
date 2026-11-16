import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RotaryPositionIntegrationTests:XCTestCase {
    private func wait(_ done:()->Bool)async throws {
        for _ in 0..<1500 {if done() {return};try await Task.sleep(nanoseconds:20_000_000)}
        throw WorkflowArchiveError("Timed out waiting for the rotary token model.")
    }
    func testManagedCausalTrainGenerateAndStandaloneExport()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Requires actual local runtime")}
        let root=try canonicalBackendTemporaryDirectory(),workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.attentionRuntimeCapabilities.contains("rotary_position"))
        XCTAssertTrue(workspace.loadComponentWorkflowExample("rotary_position",device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer {host.cancel();if !workspace.isBusy {try? FileManager.default.removeItem(at:root)}}
        XCTAssertTrue(host.run(train:WorkflowExamples.trainAddress),host.statusMessage)
        try await wait {
            (host.frozenBindings.reference(for:WorkflowExamples.trainAddress) != nil || host.statusMessage.hasPrefix("Trained model unavailable"))
                && !workspace.isBusy && !host.owner.isBusy && !host.hasActiveOperation
        }
        _ = try XCTUnwrap(host.frozenBindings.reference(for:WorkflowExamples.trainAddress),host.statusMessage)
        let trainedCount = host.outcomes.count
        // Publishing a frozen reference is not the idle boundary: automatic
        // model consumers and their final data sweep still own the operation.
        // The manual Generate action uses the same idle precondition as the UI.
        XCTAssertTrue(host.generateText(address:TextGenerationPracticeExample.address),host.statusMessage)
        try await wait {!workspace.isBusy && !host.hasActiveOperation}
        let record=try XCTUnwrap(host.generation?.record,host.generation?.statusMessage ?? host.statusMessage)
        XCTAssertEqual(record.status,"completed")
        XCTAssertEqual(host.outcomes.count,trainedCount,"Generate must reuse the completed model without retraining")
        let verified=try record.verify(root:root)
        guard case .array(let rows)=verified.result["rows"],case .object(let row)=rows.first else {return XCTFail("Missing generated table")}
        XCTAssertEqual(row["new_tokens"],.integer(16))
        let destination=root.appendingPathComponent("rotary-replay"),pending=try WorkflowGenerationExport.stage(record,root:root,to:destination)
        defer {pending.close()};try pending.commit()
        let model=try String(contentsOf:destination.appendingPathComponent("model.py"),encoding:.utf8)
        XCTAssertTrue(model.contains("execute_rotary_position"));XCTAssertTrue(model.contains("rotary_position"))
        let process=Process(),stdout=Pipe(),stderr=Pipe()
        process.executableURL=URL(fileURLWithPath:development).appendingPathComponent(".venv/bin/python")
        process.arguments=[destination.appendingPathComponent("generate.py").path]
        process.standardOutput=stdout;process.standardError=stderr
        try process.run();let bytes=stdout.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,String(data:stderr.fileHandleForReading.readDataToEndOfFile(),encoding:.utf8) ?? "Replay failed")
        XCTAssertEqual(try JSONDecoder().decode(JSONValue.self,from:bytes),.object(verified.result))

    }
    func testInlineDraftAndLessonArtworkRegistration()throws {
        var draft=try AttentionInspectorDraft(componentID:"rotary_position",values:[:])
        draft.values["rotation"] = .string("custom");draft.setText("3",key:"rotary_dim")
        XCTAssertThrowsError(try draft.configuration());draft.setText("2",key:"rotary_dim")
        draft.setText("1e-4",key:"position_scale");draft.setText("-3",key:"sequence_axis")
        XCTAssertEqual(try draft.configuration()["sequence_axis"],.integer(-3))
        XCTAssertEqual(try draft.configuration()["position_scale"],.number(0.0001))
        XCTAssertEqual(ComponentArtworkCatalog.assets["rotary_position"],"rotary_position-v2")
        XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:"rotary_position"))
    }
}
