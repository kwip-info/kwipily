import XCTest
import Combine
@testable import KwipMLCore
@testable import KwipMLApp

final class DataLoadingFailedGradientTests:XCTestCase {
    @MainActor func testSelectedReaderFailureRetainsActualSubsetBatchCounts()async throws {
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Actual worker required")}
        let root=try canonicalBackendTemporaryDirectory();defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
        await verifyRuntimeForIntegration(workspace)
        var document=try WorkflowExamples.preparedClassifier(projectID:UUID().uuidString,device:"cpu",model:FailedGradientHistoryTests.model(),scaleFeatures:false)
        document.schemaVersion=13
        var loaderValues=DataLoaderConfiguration.defaults;loaderValues["batch_size"]=4
        let train=try XCTUnwrap(document.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        guard case .train(var settings)=document.nodes[train].payload else {return XCTFail("Missing Train")}
        settings["datasetAccess"] = .object(["kind":.string("subset"),"indices":.array((0..<8).map {.integer(Int64($0))}),"max_rows":100000])
        settings["batching"] = .object(loaderValues)
        document.nodes[train].payload = .train(config:settings)
        for i in document.nodes.indices {
            if document.nodes[i].address.localID=="cast" {
                document.nodes[i].payload = .dataOperator(operatorID:"kwip.data.columns",config:[
                    "rows":.array([("signal","float64"),("context","float64"),("label","int64")].map {name,type in
                        .object(["source":.string(name),"included":.bool(true),"outputName":.string(name),"targetType":.string(type)])
                    }),"newColumns":.string("keep"),"castFailure":.string("error"),"trimText":.bool(false),"missingPolicy":.string("error")])
            }
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
        XCTAssertTrue(outcome.error?.contains("Non-finite training loss") == true,outcome.error ?? "Missing original failure")
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
        XCTAssertEqual(evidence.context.trainingRows,8)
        XCTAssertEqual(evidence.interruption.batches.first?.trainingRows,4)
    }
}
