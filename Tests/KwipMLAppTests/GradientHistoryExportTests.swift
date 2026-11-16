import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

/// Genuine prepared training and cold History, separate from synthetic archive admission.
@MainActor final class GradientHistoryExportTests:XCTestCase {
    func testActualSelectedDerivativesKeepIndependentPermissionColdHistoryAndPortableHelpers()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the actual development worker for gradient History/export")}
        let repository=URL(fileURLWithPath:development),root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        XCTAssertTrue(workspace.gradientControlRuntimeCapabilities.contains("gradient_policy"));XCTAssertTrue(workspace.gradientControlRuntimeCapabilities.contains("trainability"))
        XCTAssertTrue(workspace.setDerivativeCapturePreferences(.init(values:true)))
        let document=try GradientWorkflowExamples.document(task:.selectedInfluence,device:"cpu")
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:workspace.registry,dataRegistry:data)
        let host=try WorkflowWorkspaceSession(workspace:workspace,owner:owner)
        workspace.activeProjectID=document.projectID;workspace.workflowSession=host
        defer{host.retire();workspace.workflowSession=nil}
        let csv=root.appendingPathComponent("private-observations.csv");try Data(try GradientWorkflowExamples.sourceCSV().utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:WorkflowExamples.sourceAddress,lease:owner.lease(),expectedRevision:0))
        let finished=expectation(description:"Actual selected-derivative training reaches terminal")
        let subscription=host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in finished.fulfill()};defer{subscription.cancel()}
        guard host.run(train:WorkflowExamples.trainAddress) else {return XCTFail(host.statusMessage)}
        // A later global edit must not revoke an already accepted run's permission.
        XCTAssertTrue(workspace.setDerivativeCapturePreferences(.factory))
        await fulfillment(of:[finished],timeout:65)
        let outcome=try XCTUnwrap(host.outcomes.first)
        guard outcome.status == .completed else {return XCTFail(outcome.error ?? host.statusMessage)}
        XCTAssertNil(outcome.recordError)
        XCTAssertEqual(host.execution?.derivativeResults?.requests.count,1)
        XCTAssertEqual(host.execution?.derivativeResults?.requests.first?.targets.count,2)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let archived)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Actual gradient result must verify from captured source, permission and artifacts")}
        XCTAssertEqual(archived.capture.derivativeCapture,.init(values:true));XCTAssertEqual(archived.capture.captureFormatVersion,8)
        let policy=try XCTUnwrap(archived.plan.gradientPolicyRecipe)
        XCTAssertEqual(policy.source,WorkflowExamples.trainAddress)
        XCTAssertEqual(policy.configuration,try GradientPolicyConfiguration(values:GradientLearningTask.selectedInfluence.gradientValues))
        XCTAssertTrue(archived.plan.inlineTrainingRecipeRoles.contains("gradientPolicy"))
        XCTAssertNil(archived.plan.trainabilityRecipe,"Selected influence does not change any layer's trainability.")
        XCTAssertEqual(archived.derivativeResults?.requests.count,1)
        XCTAssertEqual(archived.derivativeResults?.requests.first?.targets.count,2)
        for file in ["gradient_policy.py","derivative-results.json"] {XCTAssertTrue(archived.outcome.artifacts.contains{$0.path == "training/artifacts/"+file})}
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:root);defer{package.remove()}
        for name in ["gradient_policy.py"] {
            let pin=try XCTUnwrap(archived.outcome.artifacts.first{$0.path == "training/artifacts/"+name})
            XCTAssertEqual(package.files.first{$0.path == name}?.sha256,pin.sha256)
        }
        XCTAssertFalse(archived.outcome.artifacts.contains {$0.path == "training/artifacts/trainability_recipe.py"})
        XCTAssertFalse(package.files.contains {$0.path == "trainability_recipe.py" || $0.path == "derivative-results.json"},"Portable inference excludes unselected policy helpers and private derivative arrays.")
        // Captured derivative arrays belong to the inspection artifact, not executable policy helpers.
        let helper=try String(contentsOf:package.directory.appendingPathComponent("gradient_policy.py"),encoding:.utf8)
        XCTAssertFalse(helper.contains("rowIDsSHA256"));XCTAssertFalse(helper.contains("derivative-results.json"))
        // Execute the exported model, not just the helper admission seam.
        let prediction=Process(),output=Pipe()
        prediction.executableURL=repository.appendingPathComponent(".venv/bin/python")
        prediction.currentDirectoryURL=package.directory
        prediction.arguments=[package.directory.appendingPathComponent("inference.py").path,"--example"]
        prediction.standardOutput=output;prediction.standardError=output
        try prediction.run()
        let deadline=Date().addingTimeInterval(15)
        while prediction.isRunning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        if prediction.isRunning {prediction.terminate();XCTFail("Portable prediction exceeded its bounded deadline")}
        prediction.waitUntilExit()
        let predicted=output.fileHandleForReading.readDataToEndOfFile()
        XCTAssertEqual(prediction.terminationStatus,0,String(decoding:predicted,as:UTF8.self))
        if prediction.terminationStatus==0 {
            let result=try JSONDecoder().decode([String:JSONValue].self,from:predicted)
            guard case .array(let rows)=result["modelOutput"] else {return XCTFail("Portable objective model must return its actual output array")}
            XCTAssertEqual(rows.count,2)
            for raw in rows {guard case .array(let row)=raw else {return XCTFail("Missing portable output row")};XCTAssertEqual(row.count,1);XCTAssertTrue(row.allSatisfy{$0.doubleValue?.isFinite == true})}
        }
        let artifact=outcome.directory.appendingPathComponent("training/artifacts/derivative-results.json"),original=try Data(contentsOf:artifact)
        try Data("{}".utf8).write(to:artifact)
        guard case .unavailable=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Changed derivative evidence must invalidate cold History")}
        XCTAssertThrowsError(try WorkflowHistoryReader.stageTrainedExport(reference,root:root))
        try original.write(to:artifact)
    }
}
