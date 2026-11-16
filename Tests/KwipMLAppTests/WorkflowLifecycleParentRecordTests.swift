import Foundation
import XCTest
@testable import KwipMLApp

final class WorkflowLifecycleParentRecordTests:XCTestCase {
    private func fixture(epochs:[Int]=[1,2],accuracy:Double=0.5)throws->[String:Data] {
        let checkpoint=Data("checkpoint fixture".utf8)
        let metrics=try JSONSerialization.data(withJSONObject:epochs.map {
            ["epoch":Double($0),"trainLoss":0.4,"validationLoss":0.5,"accuracy":accuracy]
        })
        let manifest:[String:Any]=["schemaVersion":1,"status":"completed","runID":UUID().uuidString,
            "checkpoint":["schemaVersion":1,"completedEpoch":2,"contract":["runtime":["device":"cpu"]],"bytes":checkpoint.count,
                          "sha256":WorkflowArchiveCodec.hash(checkpoint),"metricsSHA256":WorkflowArchiveCodec.hash(metrics)]]
        return ["manifest.json":try JSONSerialization.data(withJSONObject:manifest),"checkpoint.pt":checkpoint,"metrics.json":metrics]
    }

    func testRejectsRehashedIncompleteOrUnorderedMetrics()throws {
        for epochs in [[1],[2,1],[1,1],[1,3]] {
            XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(fixture(epochs:epochs)))
        }
        XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(fixture(accuracy:1.1)))
        XCTAssertEqual(try WorkflowLifecycleParentRecord.decode(fixture()).metrics.count,2)
    }

    func testRejectsChangedCheckpointAndInvalidInventory()throws {
        var contents=try fixture()
        contents["checkpoint.pt"]=Data("changed".utf8)
        XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(contents))
        contents=try fixture()
        contents["extra.json"]=Data("{}".utf8)
        XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(contents))
        contents.removeValue(forKey:"extra.json")
        contents.removeValue(forKey:"metrics.json")
        XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(contents))
    }
    func testDeviceComesFromParentContractAndNeverDefaultsToCPU()throws {
        for device in ["cpu","mps","auto",""] {
            var contents=try fixture()
            var manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:contents["manifest.json"]!) as? [String:Any])
            var checkpoint=manifest["checkpoint"] as! [String:Any]
            checkpoint["contract"]=["runtime":["device":device]]
            manifest["checkpoint"]=checkpoint
            contents["manifest.json"]=try JSONSerialization.data(withJSONObject:manifest)
            if ["cpu","mps"].contains(device) {
                XCTAssertEqual(try WorkflowLifecycleParentRecord.decode(contents).device,device)
            } else {XCTAssertThrowsError(try WorkflowLifecycleParentRecord.decode(contents))}
        }
    }

}
