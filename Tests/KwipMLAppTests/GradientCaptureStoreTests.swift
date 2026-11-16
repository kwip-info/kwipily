import XCTest
import CryptoKit
import KwipMLCore
@testable import KwipMLApp

@MainActor final class GradientCaptureStoreTests:XCTestCase {
    let id="00000000-0000-0000-0000-000000000001"
    var config:[String:Any] {["schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":1,"maxCaptures":2]}
    var inventory:[String:Any] {["parameterCount":1,"elementCount":2,"parameters":[["name":"layers.a.weight","nodeID":"a","shape":[1,2],"elementCount":2]]]}
    func data(_ value:Any)throws->Data {try JSONSerialization.data(withJSONObject:value,options:.sortedKeys)}
    var descriptor:[String:Any] {["schemaVersion":1,"id":"kwip.gradient-summary.v1","supportedDevices":["cpu","mps"],"phase":"afterBackwardBeforeStep","limits":["epochs":10000,"batchesPerEpoch":8,"captures":1024,"artifactBytes":8000000,"chunkElements":65536]]}
    func start(_ store:GradientCaptureStore)throws {
        store.installCapability(try data(descriptor))
        store.begin(config:try GradientCaptureCodec.decodeConfiguration(data(config)))
        store.consumeStarted(try data(["event":"started","runID":id,"device":"cpu","initialEpoch":0,"totalEpochs":3,"trainingSamples":4,"batchSize":2,"gradientCapture":["schemaVersion":1,"config":config,"initialEpoch":0,"inventory":inventory]]))
    }
    func event(_ epoch:Int)->[String:Any] {
        let batch:[String:Any]=["batchIndex":0,"trainingRows":2,"presentParameterCount":1,"presentElementCount":2,"missingParameterCount":0,"missingElementCount":0,"finiteElementCount":2,"nonfiniteElementCount":0,"zeroElementCount":0,"l2Norm":sqrt(2.0),"maxAbs":1.0]
        return ["event":"gradientSummary","schemaVersion":1,"runID":id,"epoch":epoch,"totalBatches":2,"scheduledBatchCount":1,"observedBatchCount":epoch<=2 ? 1:0,"totalTrainingRows":4,"observedTrainingRows":epoch<=2 ? 2:0,"captureBudgetRemaining":max(0,2-epoch),"status":epoch<=2 ? "captured":"budgetExhausted","batches":epoch<=2 ? [batch]:[]]
    }
    func testLiveSequenceRetainsExplicitBudgetRowsAndInterruptsAsPartial()throws {
        let store=GradientCaptureStore();try start(store)
        for epoch in 1...3 {store.consumeSummary(try data(event(epoch)))}
        XCTAssertEqual(store.status,.capturing,store.message);XCTAssertEqual(store.summaries.count,3)
        store.interrupt(status:"Cancelled")
        XCTAssertEqual(store.status,.partial);XCTAssertEqual(store.summaries.count,3);XCTAssertNil(store.artifact)
        store.reset();XCTAssertEqual(store.status,.off);XCTAssertTrue(store.summaries.isEmpty)
    }
    func testResumedRunStartsFreshBudgetWithoutParentEpochs()throws {
        let store=GradientCaptureStore();store.installCapability(try data(descriptor))
        store.begin(config:try GradientCaptureCodec.decodeConfiguration(data(config)))
        store.consumeStarted(try data(["event":"started","runID":id,"device":"cpu","initialEpoch":2,"totalEpochs":4,"trainingSamples":4,"batchSize":2,"gradientCapture":["schemaVersion":1,"config":config,"initialEpoch":2,"inventory":inventory]]))
        var resumed=event(1);resumed["epoch"]=3
        store.consumeSummary(try data(resumed))
        XCTAssertEqual(store.status,.capturing,store.message);XCTAssertEqual(store.summaries.map(\.epoch),[3])
        XCTAssertEqual(store.summaries.first?.captureBudgetRemaining,1)
    }
    func testOutOfOrderForeignRunUnknownFieldsAndBudgetOverflowWithholdEvidence()throws {
        for kind in ["order","foreign","unknown","budget","rows"] {
            let store=GradientCaptureStore();try start(store)
            var invalid=event(1)
            switch kind {
            case "order":invalid=event(2)
            case "foreign":invalid["runID"]=UUID().uuidString
            case "unknown":invalid["invented"]=true
            case "budget":invalid["captureBudgetRemaining"]=2
            default:invalid["totalTrainingRows"]=5
            }
            store.consumeSummary(try data(invalid))
            XCTAssertEqual(store.status,.invalid,kind);XCTAssertTrue(store.summaries.isEmpty,kind)
            store.consumeSummary(try data(event(1)));XCTAssertTrue(store.summaries.isEmpty)
        }
    }
    func artifactData()throws->Data {
        try data(["schemaVersion":1,"id":"kwip.gradient-summary.v1","runID":id,"device":"cpu","phase":"afterBackwardBeforeStep","config":config,"initialEpoch":0,"finalEpoch":3,"inventory":inventory,"summaries":(1...3).map(event)])
    }
    func manifest(_ artifact:Data)throws->Data {
        try data(["status":"completed","runID":id,"runtime":["device":"cpu"],"gradientCapture":["schemaVersion":1,"file":"gradient-summary.json","sha256":SHA256.hash(data:artifact).map {String(format:"%02x",$0)}.joined(),"bytes":artifact.count,"observedEpochCount":3,"capturedEpochCount":2,"observedBatchCount":2,"initialEpoch":0,"finalEpoch":3]])
    }
    func testRestoreVerifiesExactArtifactAndRejectsTamperAndSymlink()throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:dir)}
        let bytes=try artifactData(),manifest=try manifest(bytes),url=dir.appendingPathComponent("gradient-summary.json")
        try bytes.write(to:url)
        let store=GradientCaptureStore();store.restore(directory:dir,manifestData:manifest)
        XCTAssertEqual(store.status,.verified,store.message);XCTAssertEqual(store.summaries.count,3)
        store.interrupt(status:"Unrelated request rejected");XCTAssertEqual(store.status,.verified)
        try (bytes+Data(" ".utf8)).write(to:url)
        store.restore(directory:dir,manifestData:manifest);XCTAssertEqual(store.status,.invalid);XCTAssertNil(store.artifact)
        try FileManager.default.removeItem(at:url)
        let other=dir.appendingPathComponent("other.json");try bytes.write(to:other)
        try FileManager.default.createSymbolicLink(at:url,withDestinationURL:other)
        store.restore(directory:dir,manifestData:manifest);XCTAssertEqual(store.status,.invalid)
    }
    func testLiveCompletionRequiresMatchingCompletionMetadataAndEveryLiveEpoch()throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:dir)}
        let bytes=try artifactData(),manifest=try manifest(bytes)
        try bytes.write(to:dir.appendingPathComponent("gradient-summary.json"))
        let object=try XCTUnwrap(JSONSerialization.jsonObject(with:manifest) as? [String:Any])
        let metadata=try data(try XCTUnwrap(object["gradientCapture"]))
        let store=GradientCaptureStore();try start(store)
        for epoch in 1...3 {store.consumeSummary(try data(event(epoch)))}
        store.complete(directory:dir,manifestData:manifest,completedMetadata:metadata)
        XCTAssertEqual(store.status,.verified)
        try start(store);store.consumeSummary(try data(event(1)))
        store.complete(directory:dir,manifestData:manifest,completedMetadata:metadata)
        XCTAssertEqual(store.status,.invalid);XCTAssertTrue(store.summaries.isEmpty)
        try start(store);for epoch in 1...3 {store.consumeSummary(try data(event(epoch)))}
        store.complete(directory:dir,manifestData:manifest)
        XCTAssertEqual(store.status,.invalid)
    }
    func testLegacyManifestIsOffAndCannotInheritPriorLiveRows()throws {
        let store=GradientCaptureStore();try start(store);store.consumeSummary(try data(event(1)))
        store.restore(directory:FileManager.default.temporaryDirectory,manifestData:try data(["status":"completed","runID":id,"runtime":["device":"cpu"]]))
        XCTAssertEqual(store.status,.off);XCTAssertTrue(store.summaries.isEmpty);XCTAssertNil(store.artifact)
    }
    func testActualWorkerStreamCompletesAndReopensThroughAppBoundary()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Development worker unavailable")}
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:dir,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:dir)}
        let graph=dir.appendingPathComponent("graph.json"),output=dir.appendingPathComponent("run")
        try JSONEncoder().encode(ExampleFlows.parallelClassifier).write(to:graph)
        let process=Process(),pipe=Pipe();process.executableURL=python;process.currentDirectoryURL=root
        process.arguments=[root.appendingPathComponent("runner/main.py").path,"--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","4","--batch-size","32","--capture-gradients","--gradient-every-epochs","2","--gradient-max-batches-per-epoch","2","--gradient-max-captures","3"]
        process.standardOutput=pipe;try process.run();let stream=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus,0)
        let store=GradientCaptureStore();store.installCapability(try data(descriptor))
        store.begin(config:GradientCaptureConfiguration(everyEpochs:2,maxBatchesPerEpoch:2,maxCaptures:3))
        var completion:Data?
        for line in String(decoding:stream,as:UTF8.self).split(separator:"\n") {
            let bytes=Data(line.utf8),event=try XCTUnwrap(JSONSerialization.jsonObject(with:bytes) as? [String:Any])
            switch event["event"] as? String {
            case "started":store.consumeStarted(bytes)
            case "gradientSummary":store.consumeSummary(bytes)
            case "completed":completion=try data(try XCTUnwrap(event["gradientCapture"]))
            default:break
            }
        }
        XCTAssertEqual(store.status,.capturing,store.message)
        XCTAssertEqual(store.summaries.map(\.observedBatchCount),[2,0,1,0])
        let manifest=try Data(contentsOf:output.appendingPathComponent("manifest.json"))
        store.complete(directory:output,manifestData:manifest,completedMetadata:try XCTUnwrap(completion))
        XCTAssertEqual(store.status,.verified,store.message)
        let saved=try XCTUnwrap(store.artifact)
        let reopened=GradientCaptureStore();reopened.restore(directory:output,manifestData:manifest)
        XCTAssertEqual(reopened.status,.verified,reopened.message);XCTAssertEqual(reopened.artifact,saved)
    }
    func testCapabilitiesRejectUnknownOrChangedContract()throws {
        let store=GradientCaptureStore();store.installCapability(try data(descriptor));XCTAssertTrue(store.canCapture)
        var invalid=descriptor;invalid["invented"]=true
        store.installCapability(try data(invalid));XCTAssertFalse(store.canCapture)
        store.installCapability(nil);XCTAssertTrue(store.supportedDevices.isEmpty)
    }
}
