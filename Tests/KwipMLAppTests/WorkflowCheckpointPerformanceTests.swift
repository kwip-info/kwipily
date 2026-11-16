import Foundation
import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Opt-in diagnostic benchmark; timings are observations, never a machine-independent pass threshold.
final class WorkflowCheckpointPerformanceTests:XCTestCase {
    @MainActor func testNearLimitParentValidationAndArchiveCost() async throws {
        guard ProcessInfo.processInfo.environment["KWIPML_CHECKPOINT_BENCHMARK"] == "1" else {throw XCTSkip("Enable the checkpoint I/O benchmark explicitly")}
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("source"),destination=root.appendingPathComponent("destination")
        try FileManager.default.createDirectory(at:source,withIntermediateDirectories:true)
        try FileManager.default.createDirectory(at:destination,withIntermediateDirectories:false)
        // Synthetic bytes exercise bounded I/O and hashes only, not tensor deserialization.
        let checkpoint=Data(repeating:0x5A,count:96_000_000)
        let metrics=Data("[{\"epoch\":1,\"trainLoss\":0.5,\"validationLoss\":0.6,\"accuracy\":0.5}]".utf8)
        let manifest:[String:Any]=["schemaVersion":1,"status":"completed","runID":UUID().uuidString,
            "checkpoint":["schemaVersion":1,"completedEpoch":1,"bytes":checkpoint.count,
                "sha256":WorkflowArchiveCodec.hash(checkpoint),"metricsSHA256":WorkflowArchiveCodec.hash(metrics),
                "contract":["runtime":["device":"cpu"]]]]
        let contents=["checkpoint.pt":checkpoint,"metrics.json":metrics,"manifest.json":try JSONSerialization.data(withJSONObject:manifest)]
        var files:[WorkflowArchiveFile]=[]
        for (name,bytes) in contents {
            try bytes.write(to:source.appendingPathComponent(name))
            files.append(.init(path:name,bytes:bytes.count,sha256:WorkflowArchiveCodec.hash(bytes)))
        }
        let snapshot=WorkflowCheckpointSnapshot(reference:.init(projectID:UUID().uuidString,directoryID:UUID()),runID:UUID().uuidString,
            requestIdentity:String(repeating:"a",count:64),manifestSHA256:WorkflowArchiveCodec.hash(contents["manifest.json"]!),directory:source,files:files)
        let start=ProcessInfo.processInfo.systemUptime
        let record=try snapshot.parentRecord()
        let validated=ProcessInfo.processInfo.systemUptime
        let archived=try snapshot.archive(to:destination)
        let copied=ProcessInfo.processInfo.systemUptime
        XCTAssertEqual(record.evidence.completedEpoch,1)
        XCTAssertEqual(archived.count,3)
        let saved=try Data(contentsOf:destination.appendingPathComponent("continuation-parent/checkpoint.pt"))
        XCTAssertEqual(WorkflowArchiveCodec.hash(saved),WorkflowArchiveCodec.hash(checkpoint))
        print(String(format:"CHECKPOINT_BENCHMARK bytes=%d parentRecord_ms=%.3f archive_ms=%.3f synchronous_total_ms=%.3f",checkpoint.count,(validated-start)*1000,(copied-validated)*1000,(copied-start)*1000))
        guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"] else {throw XCTSkip("Enable a development runtime for session scheduling measurement")}
        let developmentRoot=URL(fileURLWithPath:development)
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:developmentRoot,allowDevelopment:true)
        var document=try WorkflowCodec.migrateToVersion3(WorkflowExamples.preparedParallelClassifier(projectID:snapshot.reference.projectID,device:"cpu"))
        let index=try XCTUnwrap(document.nodes.firstIndex {$0.address == WorkflowExamples.trainAddress})
        document.nodes[index].payload = .train(config:["trainingMode":.string("exactResume")])
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:.builtIn(),
            dataRegistry:.load(from:developmentRoot.appendingPathComponent("runner/dataflow-operators.json")))
        let selected=WorkflowCheckpointReference(projectID:document.projectID,directoryID:snapshot.reference.directoryID.uuidString,
            runID:snapshot.runID,requestIdentity:snapshot.requestIdentity,modelManifestSHA256:snapshot.manifestSHA256)
        let request=try WorkflowExecutionRequest.capture(from:owner,train:WorkflowExamples.trainAddress,runtime:runtime,checkpointParent:selected)
        let csv=root.appendingPathComponent("source.csv")
        try Data("signal,context,label\n1,2,0\n2,1,1\n".utf8).write(to:csv)
        let finished=expectation(description:"Capture completes before deliberate worker boundary stop")
        var outcome:WorkflowExecutionOutcome?,launches=0,heartbeatTimes:[Double]=[]
        let runDirectory=root.appendingPathComponent(UUID().uuidString)
        let session=WorkflowExecutionSession(request:request,owner:owner,sources:[WorkflowExamples.sourceAddress:try .capture(csv)],directory:runDirectory,
            reservation:.init(acquire:{UUID()},release:{_ in}),checkpointSnapshot:snapshot,
            launch:{_,_,_,_,_ in launches+=1;throw WorkflowPhaseError("Benchmark stops at worker launch")},onTerminal:{outcome=$0;finished.fulfill()})
        let runStart=ProcessInfo.processInfo.systemUptime
        XCTAssertTrue(session.start())
        let returned=ProcessInfo.processInfo.systemUptime
        let heartbeat=Task { @MainActor in
            while !Task.isCancelled {
                heartbeatTimes.append(ProcessInfo.processInfo.systemUptime)
                try? await Task.sleep(nanoseconds:2_000_000)
            }
        }
        await fulfillment(of:[finished],timeout:20)
        heartbeat.cancel()
        XCTAssertEqual(launches,1);XCTAssertEqual(outcome?.status,.failed)
        XCTAssertEqual(outcome?.error,"Benchmark stops at worker launch")
        XCTAssertFalse(owner.isBusy)
        XCTAssertTrue(FileManager.default.fileExists(atPath:runDirectory.appendingPathComponent("capture.json").path))
        let samples=[runStart]+heartbeatTimes+[ProcessInfo.processInfo.systemUptime]
        let maximumGap=zip(samples,samples.dropFirst()).map {$1-$0}.max() ?? 0
        print(String(format:"CHECKPOINT_SESSION_BENCHMARK bytes=%d start_return_ms=%.3f main_actor_callbacks=%d maximum_observed_gap_ms=%.3f",checkpoint.count,(returned-runStart)*1000,heartbeatTimes.count,maximumGap*1000))

    }
}
