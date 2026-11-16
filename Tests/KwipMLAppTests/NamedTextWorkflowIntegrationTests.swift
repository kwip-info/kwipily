import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class NamedTextWorkflowIntegrationTests:XCTestCase {
    func testRealTokenTrainingColdHistoryAndPortableExport()async throws {try await exercise(explicitObjective:false)}
    func testRealNamedIntegerObjectiveHistoryAndPortableExport()async throws {try await exercise(explicitObjective:true)}
    func testRealSequenceObjectiveHistoryAndPortableExport()async throws {try await exercise(explicitObjective:true,sequence:true)}
    func testSequenceCaptureOffRetainsReplayAndPortableModel()async throws {try await exercise(explicitObjective:true,sequence:true,capture:false)}
    private func exercise(explicitObjective:Bool,sequence:Bool=false,capture:Bool=true)async throws {
        guard let python=ProcessInfo.processInfo.environment["KWIPML_NAMED_INPUT_TEST_PYTHON"] ?? ProcessInfo.processInfo.environment["KWIPML_BACKEND_TEST_PYTHON"] else {throw XCTSkip("Select real isolated runner interpreter")}
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        var document=try sequence ? SequenceTrainingPracticeExample.document():TextEncodingPracticeExample.trainingDocument()
        if explicitObjective && !sequence {
            let objective=try LossConfiguration(componentID:"loss_cross_entropy",values:[:]).recipeValue
            let index=try XCTUnwrap(document.nodes.firstIndex(where:{$0.address==WorkflowExamples.trainAddress}))
            guard case .train(var config)=document.nodes[index].payload,case .object(var mapping)=config["dataMapping"],case .object(let plan)=mapping["namedInputPlan"] else {return XCTFail("Named mapping missing")}
            config["objective"]=objective
            let target=LossTargetMapping(componentID:"loss_cross_entropy",inputColumns:["tokens"],roles:[.init(role:"target",columns:["label"],sampleShape:[],dtype:.int64)])
            mapping=["namedInputPlan":.object(plan),"lossTargetMapping":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(target))]
            config["dataMapping"] = .object(mapping);document.nodes[index].payload = .train(config:config)
        }
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        XCTAssertTrue(host.bindings.setPath(repository.appendingPathComponent("Sources/KwipMLApp/Resources/text-training-observations.csv").path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        if !capture {try host.owner.setRunHistoryPolicy(.none)}
        host.executionLaunch={_,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(URL(fileURLWithPath:python),args,directory,line,exit)}
        guard host.run() else {return XCTFail(host.statusMessage)}
        let deadline=Date().addingTimeInterval(90)
        while (host.outcomes.isEmpty || host.owner.isBusy) && Date()<deadline {try await Task.sleep(for:.milliseconds(50))}
        if host.owner.isBusy {host.cancel()}
        let session=try XCTUnwrap(host.execution)
        XCTAssertEqual(session.phase,.completed,session.statusMessage)
        XCTAssertTrue(host.owner.select(.init(nodeAddresses:[TextEncodingPracticeExample.address]),lease:host.owner.lease()))
        if capture {
        let viewer=try XCTUnwrap(host.capturedTableViewSession(for:.init(address:TextEncodingPracticeExample.address,port:"training")))
        let vocabulary=try XCTUnwrap(viewer.tokenVocabulary?.resolve(column:"tokens"))
        XCTAssertEqual(vocabulary,["<unk>","<pad>","a","b"])
        XCTAssertNil(try viewer.tokenVocabulary?.resolve(column:"label"))
        }
        let outcome=try XCTUnwrap(host.outcomes.first)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let history)=WorkflowHistoryReader.load(reference,root:workspace.workspaceDirectory) else {return XCTFail("Cold named History must verify")}
        XCTAssertEqual(history.plan.namedInputPlan?.inputs.first?.name,"tokens")
        XCTAssertFalse(session.liveTrainingProgress.isEmpty)
        XCTAssertEqual(session.liveTrainingProgress.last?.kind,"validation")
        XCTAssertEqual(history.modelContent?.trainingProgress != nil,capture)
        XCTAssertEqual(outcome.artifactHashes[TrainingProgressArchive.path] != nil,capture)
        XCTAssertEqual(FileManager.default.fileExists(atPath:outcome.directory.appendingPathComponent(TrainingProgressArchive.path).path),capture)
        if let archive=history.modelContent?.trainingProgress {
            try archive.validate(request:history.capture.requestIdentity,worker:try XCTUnwrap(history.outcome.workerRunID),metrics:try XCTUnwrap(history.modelContent?.metrics))
        }
        if sequence {
            XCTAssertEqual(history.plan.namedInputPlan?.schemaVersion,2)
            XCTAssertEqual(history.plan.requiredPreparationNodes,["text","windows"])
        }
        let package=try WorkflowHistoryReader.stageTrainedExport(reference,root:workspace.workspaceDirectory)
        defer {package.remove()}
        XCTAssertTrue(package.files.contains {$0.path=="tokenizers.json"})
        XCTAssertTrue(package.files.contains {$0.path=="named_inputs.py"})
        let process=Process();process.executableURL=URL(fileURLWithPath:python);process.arguments=[package.directory.appendingPathComponent("inference.py").path,"--example"];process.currentDirectoryURL=package.directory
        let pipe=Pipe();process.standardOutput=pipe;process.standardError=pipe
        try process.run();let output=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0,String(decoding:output,as:UTF8.self))
        if sequence && capture {
            workspace.refreshDraftCode();await workspace.codeGenerationTask?.value
            let limit=Date().addingTimeInterval(20)
            while workspace.isGeneratingCode && Date()<limit {try await Task.sleep(for:.milliseconds(20))}
            XCTAssertNil(workspace.draftCodeError)
            XCTAssertTrue(workspace.currentCodeReady)
            let runtime=try XCTUnwrap(workspace.runtime),evidence=try XCTUnwrap(workspace.codeRuntimeEvidence)
            let contents=try WorkflowCodePackage.contents(files:workspace.draftCodeFiles,runtime:runtime,expectedEvidence:evidence)
            let code=root.appendingPathComponent("Code package")
            try WorkflowCodePackage.write(contents,to:code)
            let bindings=root.appendingPathComponent("bindings.json")
            try JSONSerialization.data(withJSONObject:["sources":["observations":repository.appendingPathComponent("Sources/KwipMLApp/Resources/text-training-observations.csv").path]]).write(to:bindings)
            let replay=root.appendingPathComponent("Replay"),log=root.appendingPathComponent("replay.log")
            FileManager.default.createFile(atPath:log.path,contents:nil)
            let status=try await Task.detached {()->Int32 in
                let p=Process(),output=try FileHandle(forWritingTo:log);defer {try? output.close()}
                p.executableURL=URL(fileURLWithPath:python);p.currentDirectoryURL=code
                p.arguments=[code.appendingPathComponent("workflow.py").path,"--bindings",bindings.path,"--output",replay.path]
                p.standardOutput=output;p.standardError=output;try p.run();p.waitUntilExit();return p.terminationStatus
            }.value
            let replayLog=try String(contentsOf:log,encoding:.utf8)
            XCTAssertEqual(status,0,replayLog)
            let actual=try JSONDecoder().decode(JSONValue.self,from:Data(contentsOf:replay.appendingPathComponent("training-000/metrics.json")))
            let saved=try JSONDecoder().decode(JSONValue.self,from:Data(contentsOf:outcome.directory.appendingPathComponent("training/artifacts/metrics.json")))
            XCTAssertEqual(actual,saved)
        }
        if explicitObjective {return}
        let selected=try WorkflowFrozenModelSnapshot.select(reference,root:workspace.workspaceDirectory)
        var evaluationDocument=host.owner.document
        let saved=WorkflowNodeAddress(domain:.control,localID:"saved"),evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate")
        evaluationDocument.nodes += [.init(address:saved,payload:.frozenModel(config:[:])),.init(address:evaluate,payload:.evaluateTest(config:["device":.string("cpu"),"batchSize":3]))]
        evaluationDocument.edges += [.init(from:.init(address:saved,port:"model"),to:.init(address:evaluate,port:"model"),kind:.frozenModel),.init(from:.init(address:TextEncodingPracticeExample.address,port:"test"),to:.init(address:evaluate,port:"test"),kind:.table)]
        _=try workspace.projectLibrary.saveSource(.workflow(evaluationDocument),id:document.projectID)
        let evaluationOwner=try WorkflowAuthoringStore(library:workspace.projectLibrary,projectID:document.projectID,modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry)
        let bindings=try WorkflowPrivateFrozenModelBindings(owner:evaluationOwner,directory:workspace.workspaceDirectory)
        XCTAssertTrue(bindings.setReference(selected,for:saved,lease:evaluationOwner.lease(),expectedRevision:0))
        let policy=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:60),projectMaximumRunSeconds:nil)
        let request=try await WorkflowEvaluationRequest.capture(from:evaluationOwner,bindings:bindings,evaluate:evaluate,runtime:try XCTUnwrap(workspace.runtime),root:workspace.workspaceDirectory,executionBudget:policy)
        let done=expectation(description:"Frozen named text evaluation")
        var result:WorkflowEvaluationSessionResult?
        let frozen=WorkflowEvaluationSession(request:request,owner:evaluationOwner,bindings:bindings,workspace:workspace.workspaceDirectory,reservation:.init(acquire:{UUID()},release:{_ in}),launch:{executable,args,directory,line,exit in try WorkflowExecutionSession.launchWorker(executable,args,directory,line,exit)},onTerminal:{result=$0;done.fulfill()})
        XCTAssertTrue(frozen.start())
        await fulfillment(of:[done],timeout:60)
        XCTAssertEqual(result?.outcome.status,.completed,result?.outcome.error ?? frozen.statusMessage)

    }
}
