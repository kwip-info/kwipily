import XCTest
import CryptoKit
@testable import KwipMLCore

final class GradientCaptureTests:XCTestCase {
    private let runID="11111111-1111-1111-1111-111111111111"
    private func fixture()->[String:Any] {
        let config:[String:Any]=["schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":1,"maxCaptures":2]
        let inventory:[String:Any]=["parameterCount":1,"elementCount":2,"parameters":[["name":"layers.m.weight","nodeID":"m","shape":[1,2],"elementCount":2]]]
        let batch:[String:Any]=["batchIndex":0,"trainingRows":4,"presentParameterCount":1,"presentElementCount":2,"missingParameterCount":0,"missingElementCount":0,"finiteElementCount":2,"nonfiniteElementCount":0,"zeroElementCount":0,"l2Norm":5.0,"maxAbs":4.0]
        let event:[String:Any]=["event":"gradientSummary","schemaVersion":1,"runID":runID,"epoch":1,"totalBatches":2,"scheduledBatchCount":1,"observedBatchCount":1,"totalTrainingRows":8,"observedTrainingRows":4,"captureBudgetRemaining":1,"status":"captured","batches":[batch]]
        return ["schemaVersion":1,"id":"kwip.gradient-summary.v1","runID":runID,"device":"cpu","phase":"afterBackwardBeforeStep","config":config,"initialEpoch":0,"finalEpoch":1,"inventory":inventory,"summaries":[event]]
    }
    private func decode(_ value:[String:Any])throws->GradientCaptureArtifact {
        try GradientCaptureCodec.decodeArtifact(JSONSerialization.data(withJSONObject:value),expectedRunID:runID,expectedDevice:"cpu")
    }
    func testLegacySettingsOmitCaptureAndStrictConfigurationBounds()throws {
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let old=try encoder.encode(ExecutionSettings())
        XCTAssertFalse(String(decoding:old,as:UTF8.self).contains("gradientCapture"))
        XCTAssertEqual(old,try encoder.encode(JSONDecoder().decode(ExecutionSettings.self,from:old)))
        var enabled=ExecutionSettings();enabled.gradientCapture=GradientCaptureConfiguration()
        XCTAssertEqual(try JSONDecoder().decode(ExecutionSettings.self,from:encoder.encode(enabled)),enabled)
        for bad in [#"{"schemaVersion":1,"everyEpochs":0,"maxBatchesPerEpoch":1,"maxCaptures":2}"#,#"{"schemaVersion":1,"everyEpochs":true,"maxBatchesPerEpoch":1,"maxCaptures":2}"#,#"{"schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":9,"maxCaptures":2}"#,#"{"schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":1,"maxCaptures":1025}"#,#"{"schemaVersion":1,"everyEpochs":1,"maxBatchesPerEpoch":1,"maxCaptures":2,"enabled":true}"#] {
            XCTAssertThrowsError(try JSONDecoder().decode(GradientCaptureConfiguration.self,from:Data(bad.utf8)))
        }
    }
    func testArtifactIdentityCountsCadenceBudgetAndUnknownKeys()throws {
        let valid=fixture();XCTAssertEqual(try decode(valid).summaries.first?.batches.first?.l2Norm,5)
        for key in ["runID","device","phase","id"] {var bad=valid;bad[key]="wrong";XCTAssertThrowsError(try decode(bad),key)}
        var unknown=valid;unknown["extra"]=true;XCTAssertThrowsError(try decode(unknown))
        for (key,value) in [("epoch",2),("observedBatchCount",2),("scheduledBatchCount",2),("captureBudgetRemaining",2),("observedTrainingRows",9)] {
            var bad=valid,event=(bad["summaries"] as! [[String:Any]])[0];event[key]=value;bad["summaries"]=[event];XCTAssertThrowsError(try decode(bad),key)
        }
        var bad=valid;bad["summaries"]=[];XCTAssertThrowsError(try decode(bad))
    }
    func testNullNormsOnlyForAbsentOrNonfiniteGradients()throws {
        for nonfinite in [false,true] {
            var value=fixture(),event=(value["summaries"] as! [[String:Any]])[0],batch=(event["batches"] as! [[String:Any]])[0]
            batch["l2Norm"]=NSNull();batch["maxAbs"]=NSNull()
            if nonfinite {batch["finiteElementCount"]=1;batch["nonfiniteElementCount"]=1}
            else {batch["presentParameterCount"]=0;batch["presentElementCount"]=0;batch["missingParameterCount"]=1;batch["missingElementCount"]=2;batch["finiteElementCount"]=0}
            event["batches"]=[batch];value["summaries"]=[event];XCTAssertNoThrow(try decode(value))
            batch["l2Norm"]=0;event["batches"]=[batch];value["summaries"]=[event];XCTAssertThrowsError(try decode(value))
        }
        var value=fixture(),event=(value["summaries"] as! [[String:Any]])[0],batch=(event["batches"] as! [[String:Any]])[0]
        batch["l2Norm"]=NSNull();batch["maxAbs"]=NSNull();event["batches"]=[batch];value["summaries"]=[event];XCTAssertThrowsError(try decode(value))
    }
    func testResumeCadenceBudgetCoverageAndMetadataHash()throws {
        var value=fixture()
        value["initialEpoch"]=1;value["finalEpoch"]=4
        value["config"]=["schemaVersion":1,"everyEpochs":2,"maxBatchesPerEpoch":2,"maxCaptures":1]
        let original=(value["summaries"] as! [[String:Any]])[0]
        var skip=original;skip["epoch"]=2;skip["scheduledBatchCount"]=0;skip["observedBatchCount"]=0;skip["observedTrainingRows"]=0;skip["captureBudgetRemaining"]=1;skip["status"]="notScheduled";skip["batches"]=[]
        var captured=original;captured["epoch"]=3;captured["scheduledBatchCount"]=2;captured["captureBudgetRemaining"]=0
        var end=skip;end["epoch"]=4;end["captureBudgetRemaining"]=0
        value["summaries"]=[skip,captured,end]
        let data=try JSONSerialization.data(withJSONObject:value),artifact=try decode(value)
        XCTAssertEqual(artifact.summaries.map(\.status),[.notScheduled,.captured,.notScheduled])
        var meta:[String:Any]=["schemaVersion":1,"file":"gradient-summary.json","sha256":SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined(),"bytes":data.count,"observedEpochCount":3,"capturedEpochCount":1,"observedBatchCount":1,"initialEpoch":1,"finalEpoch":4]
        let metadata=try GradientCaptureCodec.decodeMetadata(JSONSerialization.data(withJSONObject:meta))
        XCTAssertNoThrow(try GradientCaptureCodec.decodeArtifact(data,expectedRunID:runID,expectedDevice:"cpu",metadata:metadata))
        meta["sha256"]=String(repeating:"0",count:64)
        let wrong=try GradientCaptureCodec.decodeMetadata(JSONSerialization.data(withJSONObject:meta))
        XCTAssertThrowsError(try GradientCaptureCodec.decodeArtifact(data,expectedRunID:runID,expectedDevice:"cpu",metadata:wrong))
        meta["capturedEpochCount"]=2000;meta["observedEpochCount"]=3000;meta["finalEpoch"]=3001
        XCTAssertThrowsError(try GradientCaptureCodec.decodeMetadata(JSONSerialization.data(withJSONObject:meta)))
        var exhausted=skip;exhausted["epoch"]=5;exhausted["scheduledBatchCount"]=2;exhausted["captureBudgetRemaining"]=0;exhausted["status"]="budgetExhausted"
        value["finalEpoch"]=5;value["summaries"]=[skip,captured,end,exhausted];XCTAssertNoThrow(try decode(value))
        exhausted["batches"]=(original["batches"] as! [[String:Any]]);value["summaries"]=[skip,captured,end,exhausted];XCTAssertThrowsError(try decode(value))
    }
    func testMalformedInventoryNonfiniteNormAndStartedBoundary()throws {
        let valid=fixture()
        var started:[String:Any]=["schemaVersion":1,"config":valid["config"]!,"initialEpoch":0,"inventory":valid["inventory"]!]
        XCTAssertNoThrow(try GradientCaptureCodec.decodeStarted(JSONSerialization.data(withJSONObject:started)))
        started["extra"]=1;XCTAssertThrowsError(try GradientCaptureCodec.decodeStarted(JSONSerialization.data(withJSONObject:started)))
        for shape in [[0,2],[4097,1],[1,1,2]] {
            var value=valid,inventory=value["inventory"] as! [String:Any],p=(inventory["parameters"] as! [[String:Any]])[0]
            p["shape"]=shape;inventory["parameters"]=[p];value["inventory"]=inventory;XCTAssertThrowsError(try decode(value))
        }
        var value=valid,event=(value["summaries"] as! [[String:Any]])[0],batch=(event["batches"] as! [[String:Any]])[0]
        batch["zeroElementCount"]=3;event["batches"]=[batch];value["summaries"]=[event];XCTAssertThrowsError(try decode(value))
        let json=String(decoding:try JSONSerialization.data(withJSONObject:valid,options:.sortedKeys),as:UTF8.self).replacingOccurrences(of:"\"l2Norm\":5",with:"\"l2Norm\":1e309")
        XCTAssertThrowsError(try GradientCaptureCodec.decodeArtifact(Data(json.utf8),expectedRunID:runID,expectedDevice:"cpu"))
        XCTAssertThrowsError(try GradientCaptureCodec.decodeArtifact(Data(repeating:32,count:8_000_001),expectedRunID:runID,expectedDevice:"cpu"))
    }

    func testActualWorkerArtifactAndStreamUseSameStrictContract()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python=root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.fileExists(atPath:python.path) else {throw XCTSkip("Python runtime unavailable")}
        let temp=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);try FileManager.default.createDirectory(at:temp,withIntermediateDirectories:true);defer{try? FileManager.default.removeItem(at:temp)}
        let graph=temp.appendingPathComponent("graph.json"),output=temp.appendingPathComponent("run")
        try JSONEncoder().encode(ExampleFlows.parallelClassifier).write(to:graph)
        let process=Process(),pipe=Pipe();process.executableURL=python;process.currentDirectoryURL=root
        process.arguments=[root.appendingPathComponent("runner/main.py").path,"--graph",graph.path,"--output",output.path,"--device","cpu","--epochs","4","--batch-size","32","--capture-gradients","--gradient-every-epochs","2","--gradient-max-batches-per-epoch","2","--gradient-max-captures","3"]
        process.standardOutput=pipe;try process.run();let stream=pipe.fileHandleForReading.readDataToEndOfFile();process.waitUntilExit();XCTAssertEqual(process.terminationStatus,0)
        let raw=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:output.appendingPathComponent("manifest.json"))) as? [String:Any])
        let metadata=try GradientCaptureCodec.decodeMetadata(JSONSerialization.data(withJSONObject:try XCTUnwrap(raw["gradientCapture"])))
        let artifactData=try Data(contentsOf:output.appendingPathComponent("gradient-summary.json"))
        let rawArtifact=try XCTUnwrap(JSONSerialization.jsonObject(with:artifactData) as? [String:Any]),runID=try XCTUnwrap(rawArtifact["runID"] as? String)
        let artifact=try GradientCaptureCodec.decodeArtifact(artifactData,expectedRunID:runID,expectedDevice:"cpu",metadata:metadata)
        XCTAssertEqual(artifact.summaries.map(\.observedBatchCount),[2,0,1,0])
        var previous:GradientEpochSummary?
        for line in String(decoding:stream,as:UTF8.self).split(separator:"\n") {
            let data=Data(line.utf8),event=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any])
            if event["event"] as? String == "started" {
                let started=try GradientCaptureCodec.decodeStarted(JSONSerialization.data(withJSONObject:try XCTUnwrap(event["gradientCapture"])))
                XCTAssertEqual(started.inventory,artifact.inventory)
            }
            if event["event"] as? String == "gradientSummary" {
                previous=try GradientCaptureCodec.decodeEvent(data,runID:runID,config:artifact.config,inventory:artifact.inventory,previous:previous)
            }
        }
        XCTAssertEqual(previous,artifact.summaries.last)
    }

}
