import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

/// Genuine managed capture from completed native History, followed by isolated no-source reload.
@MainActor final class WorkflowProgramExportIntegrationTests:XCTestCase {
    func testActualTrainedHistoryStaticAndBoundedProgramsReloadAfterOriginalRemoval()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable the actual development runtime")}
        guard let pointer=realpath(FileManager.default.temporaryDirectory.path,nil) else {throw WorkflowFileLeaseError.unsafePath}
        let root=URL(fileURLWithPath:String(cString:pointer)).appendingPathComponent(UUID().uuidString);free(pointer)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let repository=URL(fileURLWithPath:development)
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        let runtime=try XCTUnwrap(workspace.runtime),version=try XCTUnwrap(workspace.runtimeContractVersion)
        XCTAssertTrue(workspace.programExportRuntimeCapability.isAvailable)
        let document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        _ = try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:workspace.registry,dataRegistry:.load(from:repository.appendingPathComponent("runner/dataflow-operators.json")))
        let budget=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:60),projectMaximumRunSeconds:nil,resources:.init(cpuThreads:2,maximumResidentBytes:1_073_741_824))
        let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,executionBudget:budget,sampleCapture:.factory,moduleCompositionEvidence:workspace.moduleCompositionDiscoveryEvidence)
        let id=UUID(),directory=root.appendingPathComponent("workflow-runs/\(document.projectID)/\(id.uuidString)")
        let ended=expectation(description:"Actual native branched training completes")
        var outcome:WorkflowExecutionOutcome?
        let run=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:try .capture(repository.appendingPathComponent("Sources/KwipMLApp/Resources/workflow-observations.csv"))],directory:directory,reservation:.init(acquire:{UUID()},release:{_ in}),onTerminal:{outcome=$0;ended.fulfill()})
        guard run.start() else {return XCTFail(run.statusMessage)}
        defer {if owner.isBusy {run.cancel()}}
        await fulfillment(of:[ended],timeout:65)
        XCTAssertEqual(run.phase,.completed,run.statusMessage+" / "+(outcome?.error ?? ""))
        XCTAssertNil(outcome?.recordError)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:id)
        guard case .verified(let saved)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Genuine completed archive must verify")}
        let content=try XCTUnwrap(saved.modelContent),preview=try XCTUnwrap(content.predictionPreview)
        XCTAssertGreaterThanOrEqual(preview.rows.count,7)
        let identity=WorkflowProgramExportIdentity(runID:saved.outcome.runID,requestIdentity:saved.outcome.requestIdentity,modelManifestSHA256:try XCTUnwrap(saved.outcome.modelManifestSHA256))
        let checkpointURL=directory.appendingPathComponent("training/artifacts/checkpoint.pt")
        let checkpointBefore=try Data(contentsOf:checkpointURL),manifestURL=directory.appendingPathComponent("training/artifacts/manifest.json"),manifestBefore=try Data(contentsOf:manifestURL)
        let transport=WorkflowProgramExportManagedTransport(expectedTorchVersion:version,accountingRoot:workspace.applicationDirectory,storageLimitBytes:1_000_000_000)
        let writer=WorkflowProgramExportWriter(transport:transport,source:WorkflowTrainedExportWriter(accountingRoot:workspace.applicationDirectory,storageLimitBytes:1_000_000_000))
        for mode in ["static","bounded"] {
            var fields=ProgramExportConfiguration.defaults
            fields["batch_mode"] = .string(mode)
            if mode == "bounded" {fields["maximum_batch_size"]=7}
            let package=try await writer.stage(reference:reference,root:root,runtime:runtime,expected:identity,configuration:ProgramExportConfiguration(values:fields),executionBudget:budget)
            XCTAssertEqual(package.identity,identity);XCTAssertEqual(package.torchVersion,version)
            XCTAssertEqual(package.inputSampleShape,[2]);XCTAssertEqual(package.outputSampleShape,[2])
            try package.publish(to:root.appendingPathComponent(mode));package.remove()
            XCTAssertEqual(try Data(contentsOf:checkpointURL),checkpointBefore)
            XCTAssertEqual(try Data(contentsOf:manifestURL),manifestBefore)
        }
        var names:[String:String]=[:]
        for (expanded,origin) in saved.plan.model.origins where ["branch-a","branch-b","classifier"].contains(origin.localNodeID) {names[origin.localNodeID]=expanded}
        XCTAssertEqual(Set(names.keys),Set(["branch-a","branch-b","classifier"]))
        let rows:[JSONValue]=try preview.rows.prefix(7).map {row in .array(try row.inputs.map {.number(try XCTUnwrap(Double($0)))})}
        let evidence:JSONValue = .object(["names":.object(names.mapValues(JSONValue.string)),"rows":.array(rows),"classes":.array(preview.rows.prefix(7).map{.integer(Int64($0.predicted))})])
        try WorkflowArchiveCodec.encode(evidence).write(to:root.appendingPathComponent("rows.json"))
        try checkpointBefore.write(to:root.appendingPathComponent("parent.pt"))
        let expectedScript=root.appendingPathComponent("measure-parent.py")
        try #"""
        import json,sys
        from pathlib import Path
        import torch
        import torch.nn.functional as F
        root=Path(__file__).parent
        evidence=json.loads((root/'rows.json').read_text())
        state=torch.load(root/'parent.pt',map_location='cpu',weights_only=True)['modelState']
        x=torch.tensor(evidence['rows'],dtype=torch.float32)
        def affine(role,x):
            prefix='layers.'+evidence['names'][role]
            return F.linear(x,state[prefix+'.weight'],state[prefix+'.bias'])
        y=affine('classifier',torch.cat([affine('branch-a',x).clamp_min(0),affine('branch-b',x).tanh()],dim=-1))
        assert y.argmax(1).tolist()==evidence['classes']
        (root/'expected.json').write_text(json.dumps(y.tolist()))
        """#.write(to:expectedScript,atomically:true,encoding:.utf8)
        try await python(runtime,script:expectedScript,root:root)
        try FileManager.default.removeItem(at:root.appendingPathComponent("parent.pt"))
        try FileManager.default.removeItem(at:root.appendingPathComponent("workflow-runs"))
        let verifyScript=root.appendingPathComponent("verify-relocated.py")
        try #"""
        import importlib.util,json,sys
        from pathlib import Path
        import torch
        root=Path(__file__).parent
        assert not (root/'parent.pt').exists() and not (root/'workflow-runs').exists()
        assert not any((Path(p)/'runner'/'program_export.py').exists() for p in sys.path if p)
        x=torch.tensor(json.loads((root/'rows.json').read_text())['rows'],dtype=torch.float32)
        expected=torch.tensor(json.loads((root/'expected.json').read_text()),dtype=torch.float32)
        for mode in ('static','bounded'):
            folder=root/mode
            assert not (folder/'model.py').exists() and not (folder/'checkpoint.pt').exists()
            spec=importlib.util.spec_from_file_location('isolated_'+mode,folder/'inference.py')
            helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
            predictor=helper.load_model()
            sizes=(2,) if mode=='static' else (2,3,7)
            for size in sizes:
                actual=predictor.predict(x[:size])
                assert list(actual.shape)==[size,2]
                torch.testing.assert_close(actual,expected[:size],rtol=1e-5,atol=1e-6)
            for size in ((1,3) if mode=='static' else (1,8)):
                try:predictor.predict(torch.zeros(size,2))
                except (ValueError,RuntimeError,AssertionError):pass
                else:raise AssertionError('Out-of-contract batch was accepted')
        print('native-managed-relocated-parity-ok')
        """#.write(to:verifyScript,atomically:true,encoding:.utf8)
        try await python(runtime,script:verifyScript,root:root)
    }
    private func python(_ runtime:RuntimeLocation,script:URL,root:URL)async throws {
        let done=expectation(description:script.lastPathComponent)
        var code:Int32?,lines:[String]=[]
        let process=try WorkflowExecutionSession.launchWorker(runtime.executableURL,["-I",script.path],root,{lines.append($0)},{value,error in code=value;if !error.isEmpty {lines.append(error)};done.fulfill()})
        defer {process.cancel()}
        await fulfillment(of:[done],timeout:45)
        XCTAssertEqual(code,0,lines.joined(separator:"\n"))
        if code != 0 {throw WorkflowArchiveError("Isolated portable verification failed: "+lines.joined(separator:"\n"))}
    }
}
