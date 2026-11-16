import XCTest
import Combine
@testable import KwipMLCore
@testable import KwipMLApp

final class FailedGradientHistoryTests:XCTestCase {
    @MainActor func testRealFailedTrainingRetainsVerifiedPartialCountsWithoutModelCompletion()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Actual worker required")}
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        var document=try WorkflowExamples.preparedClassifier(projectID:UUID().uuidString,device:"cpu",model:Self.model(),scaleFeatures:false)
        for i in document.nodes.indices {
            if document.nodes[i].address.localID=="cast" {document.nodes[i].payload = .dataOperator(operatorID:"kwip.data.columns",config:["rows":.array([("signal","float64"),("context","float64"),("label","int64")].map {name,type in
                .object(["source":.string(name),"included":true,"outputName":.string(name),"targetType":.string(type)])
            }),"castFailure":.string("error"),"trimText":false,"newColumns":.string("keep"),"missingPolicy":.string("error")])}
            if case .runSetup(var config)=document.nodes[i].payload {
                config["epochs"]=2;config["gradientCapture"]=try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(GradientCaptureConfiguration(everyEpochs:1,maxBatchesPerEpoch:1,maxCaptures:4)))
                document.nodes[i].payload = .runSetup(config:config)
            }
        }
        _ = try workspace.projectLibrary.saveSource(.workflow(document))
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let csv=root.appendingPathComponent("finite-observations.csv")
        let rows=(0..<64).map { i in "\(1e9*(0.5+Double(i)/128)),\(1e9*(1-Double(i)/128)),\(i%2)\n" }.joined()
        try Data(("signal,context,label\n"+rows).utf8).write(to:csv)
        XCTAssertTrue(host.bindings.setPath(csv.path,for:WorkflowExamples.sourceAddress,lease:host.owner.lease(),expectedRevision:host.bindings.revision))
        let done=expectation(description:"Real nonfinite training failure")
        let observation=host.$outcomes.dropFirst().filter{!$0.isEmpty}.prefix(1).sink{_ in done.fulfill()};defer{observation.cancel()}
        guard host.run(train:WorkflowExamples.trainAddress) else {return XCTFail(host.statusMessage)}
        await fulfillment(of:[done],timeout:60)
        if workspace.isBusy {host.cancel()}
        let outcome=try XCTUnwrap(host.outcomes.first)
        XCTAssertEqual(outcome.status,.failed)
        let originalFailure=try XCTUnwrap(outcome.error)
        XCTAssertTrue(originalFailure.hasPrefix("Validation logits: values must be finite"),originalFailure)
        let reference=WorkflowHistoryReference(projectID:document.projectID,directoryID:try XCTUnwrap(UUID(uuidString:outcome.directory.lastPathComponent)))
        guard case .verified(let verified)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Failed archive must independently verify")}
        XCTAssertNil(verified.modelContent);XCTAssertNil(verified.gradientArtifact)
        XCTAssertNil(verified.outcome.modelManifestSHA256);XCTAssertNil(verified.outcome.workerRunID)
        let evidence=try XCTUnwrap(verified.failedGradientEvidence)
        XCTAssertEqual(evidence.interruption.batches.count,1)
        XCTAssertEqual(evidence.interruption.batches[0].nonfiniteElementCount,10)
        XCTAssertEqual(evidence.interruption.batches[0].presentElementCount,15)
        XCTAssertNil(evidence.interruption.batches[0].l2Norm)
        XCTAssertEqual(evidence.completedSummaries.count,0)
        XCTAssertEqual(evidence.interruption.lastBackwardBatchIndex,0)
        XCTAssertEqual(verified.outcome.schemaVersion,3)
        let diagnostic=try XCTUnwrap(verified.outcome.failedGradientEvidence)
        try verifyProtocolAndArchiveControls(evidence,verified:verified)
        let diagnosticBytes=try Data(contentsOf:outcome.directory.appendingPathComponent(diagnostic.file.path))
        if let export=ProcessInfo.processInfo.environment["KWIPML_FAILED_GRADIENT_NATIVE_FIXTURE"] {
            let folder=URL(fileURLWithPath:export)
            XCTAssertFalse(FileManager.default.fileExists(atPath:folder.path))
            try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
            try WorkflowCodec.encode(document).write(to:folder.appendingPathComponent("Failed gradient observations.kwipml"),options:.withoutOverwriting)
            try Data(contentsOf:csv).write(to:folder.appendingPathComponent("observations.csv"),options:.withoutOverwriting)
        }
        let diagnosticURL=outcome.directory.appendingPathComponent(diagnostic.file.path)
        try FileManager.default.removeItem(at:diagnosticURL)
        let foreign=root.appendingPathComponent("foreign.json");try diagnosticBytes.write(to:foreign)
        try FileManager.default.createSymbolicLink(at:diagnosticURL,withDestinationURL:foreign)
        guard case .unavailable = WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Symlink diagnostic evidence must be refused")}
        try FileManager.default.removeItem(at:diagnosticURL)
        try diagnosticBytes.write(to:diagnosticURL)
        try Data("tampered".utf8).write(to:outcome.directory.appendingPathComponent(diagnostic.file.path))
        guard case .unavailable(let reason)=WorkflowHistoryReader.load(reference,root:root) else {return XCTFail("Modified evidence must be refused")}
        XCTAssertTrue(reason.contains(originalFailure),"The exact original failure must survive evidence refusal")
    }
    private func verifyProtocolAndArchiveControls(_ evidence:WorkflowFailedGradientEvidence,verified:VerifiedWorkflowOutcome)throws {
        let context=evidence.context
        let started:[String:Any]=["event":"started","runID":context.workerRunID,"device":context.device,"totalEpochs":context.requestedFinalEpoch,"initialEpoch":context.initialEpoch,"trainingSamples":context.trainingRows,"batchSize":min(context.trainingRows,context.requestedBatchSize),"failedGradientCapture":try JSONSerialization.jsonObject(with:WorkflowFailedGradientEvidence.wire(evidence.capability)),"gradientCapture":["schemaVersion":1,"initialEpoch":context.initialEpoch,"config":try JSONSerialization.jsonObject(with:WorkflowFailedGradientEvidence.wire(context.configuration)),"inventory":try JSONSerialization.jsonObject(with:WorkflowFailedGradientEvidence.wire(context.inventory))]]
        let interrupted=try JSONSerialization.jsonObject(with:WorkflowFailedGradientEvidence.wire(evidence.interruption)) as! [String:Any]
        func feed(_ event:[String:Any],_ collector:WorkflowFailedGradientCollector)throws {collector.consume(try JSONSerialization.data(withJSONObject:event),event:event,name:event["event"] as! String)}
        func fresh()->WorkflowFailedGradientCollector {WorkflowFailedGradientCollector(plan:verified.plan,trainingRows:context.trainingRows,initialEpoch:context.initialEpoch)}
        func result(_ collector:WorkflowFailedGradientCollector)throws->WorkflowFailedGradientEvidence? {try collector.result(workflowRunID:evidence.workflowRunID,requestSHA256:evidence.requestSHA256,preparationSHA256:evidence.preparationManifestSHA256)}
        let missing=fresh();try feed(started,missing);try feed(interrupted,missing)
        XCTAssertThrowsError(try result(missing),"Missing handled-error boundary cannot publish")
        let duplicate=fresh();try feed(started,duplicate);try feed(interrupted,duplicate);try feed(["event":"error","message":"original"],duplicate);try feed(["event":"error","message":"extra"],duplicate)
        XCTAssertThrowsError(try result(duplicate),"Duplicate terminal errors cannot publish")
        let completed=fresh();try feed(started,completed);try feed(interrupted,completed);try feed(["event":"completed"],completed);try feed(["event":"error","message":"original"],completed)
        XCTAssertThrowsError(try result(completed),"Completed model event cannot be recast as interrupted evidence")
        let late=fresh();try feed(started,late);try feed(interrupted,late);try feed(["event":"epoch"],late)
        XCTAssertNotNil(late.notice);XCTAssertNil(try result(late))
        for field in ["schemaVersion","initialEpoch"] {
            var malformed=started;var nested=malformed["gradientCapture"] as! [String:Any];nested[field]=99;malformed["gradientCapture"]=nested
            let refused=fresh();try feed(malformed,refused)
            try feed(["event":"error","message":verified.outcome.error ?? "original failure"],refused)
            XCTAssertNotNil(refused.notice,"Nested gradientCapture.\(field) must be checked")
            XCTAssertNil(try result(refused))
            if let notice=refused.notice {
                let original=verified.outcome
                let unavailable=WorkflowArchiveOutcome(schemaVersion:3,projectID:original.projectID,directoryID:original.directoryID,runID:original.runID,requestIdentity:original.requestIdentity,status:.failed,terminalPhase:.training,startedAt:original.startedAt,finishedAt:original.finishedAt,artifacts:original.artifacts.filter{$0.path != WorkflowFailedGradientEvidence.path},capture:original.capture,preparationManifestSHA256:original.preparationManifestSHA256,error:original.error,failedGradientNotice:notice)
                let restored=try WorkflowArchiveCodec.decodeOutcome(WorkflowArchiveCodec.encodeOutcome(unavailable))
                XCTAssertEqual(restored.error,original.error);XCTAssertNil(restored.failedGradientEvidence);XCTAssertEqual(restored.failedGradientNotice,notice)
            }
        }
        var badStart=started;badStart["failedGradientCapture"]=["schemaVersion":99]
        let invalid=fresh();try feed(badStart,invalid)
        XCTAssertNotNil(invalid.notice);XCTAssertNil(try result(invalid))
        var old=verified.outcome;old.schemaVersion=2
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeOutcome(old),"V2 may not smuggle in failed diagnostic descriptors")
        var completion=verified.outcome
        completion=WorkflowArchiveOutcome(schemaVersion:3,projectID:completion.projectID,directoryID:completion.directoryID,runID:completion.runID,requestIdentity:completion.requestIdentity,status:.completed,terminalPhase:.training,startedAt:completion.startedAt,finishedAt:completion.finishedAt,artifacts:completion.artifacts,capture:completion.capture,preparationManifestSHA256:completion.preparationManifestSHA256,failedGradientEvidence:completion.failedGradientEvidence)
        XCTAssertThrowsError(try WorkflowArchiveCodec.encodeOutcome(completion),"Partial observations never establish completed model acceptance")
        let data=try WorkflowFailedGradientEvidence.wire(evidence)
        for field in ["workerRunID","device"] {
            var raw=try JSONSerialization.jsonObject(with:data) as! [String:Any]
            var context=raw["context"] as! [String:Any];context[field]=field=="device" ? "mps":UUID().uuidString;raw["context"]=context
            XCTAssertThrowsError(try WorkflowFailedGradientEvidence.decode(JSONSerialization.data(withJSONObject:raw),plan:verified.plan,workflowRunID:evidence.workflowRunID,requestSHA256:evidence.requestSHA256,preparationSHA256:evidence.preparationManifestSHA256,trainingRows:evidence.context.trainingRows,initialEpoch:evidence.context.initialEpoch))
        }
        XCTAssertThrowsError(try WorkflowFailedGradientEvidence.decode(data,plan:verified.plan,workflowRunID:UUID().uuidString,requestSHA256:evidence.requestSHA256,preparationSHA256:evidence.preparationManifestSHA256,trainingRows:context.trainingRows,initialEpoch:context.initialEpoch))
        XCTAssertThrowsError(try WorkflowFailedGradientEvidence.decode(data,plan:verified.plan,workflowRunID:evidence.workflowRunID,requestSHA256:String(repeating:"0",count:64),preparationSHA256:evidence.preparationManifestSHA256,trainingRows:context.trainingRows,initialEpoch:context.initialEpoch))
    }
    static func model()->GraphDocument {
        var nodes:[GraphNode]=[.init(id:"input",type:"input",config:["features":2]),.init(id:"head",type:"linear",config:["in_features":2,"out_features":1])]
        var edges:[GraphEdge]=[.init(id:"0",source:"input",target:"head")],previous="head"
        for i in 0..<4 {
            let id="b\(i)";nodes.append(.init(id:id,type:"bilinear",config:["in1_features":1,"in2_features":1,"out_features":1,"bias":true]))
            edges.append(.init(id:"b\(i)a",source:previous,target:id,targetPort:"a"));edges.append(.init(id:"b\(i)b",source:previous,target:id,targetPort:"b"));previous=id
        }
        nodes += [.init(id:"bound",type:"tanh"),.init(id:"score",type:"linear",config:["in_features":1,"out_features":2]),.init(id:"output",type:"output")]
        edges += [.init(id:"bound",source:previous,target:"bound"),.init(id:"score",source:"bound",target:"score"),.init(id:"output",source:"score",target:"output")]
        return GraphDocument(name:"Real saturated backward failure",nodes:nodes,edges:edges)
    }
}
