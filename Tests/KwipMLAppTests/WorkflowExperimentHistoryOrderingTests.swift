import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentHistoryOrderingTests:XCTestCase {
    func testRecordedCreationTimeOrdersGroupsAndLegacyRemainsReadable() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString
        let ids=["00000000-0000-0000-0000-000000000001","00000000-0000-0000-0000-000000000002","00000000-0000-0000-0000-000000000003"]
        for (index,id) in ids.enumerated() {
            let directory=root.appendingPathComponent("workflow-experiments/\(project)/\(id)")
            try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
            let hash=String(repeating:"a",count:64)
            var record:[String:Any]=["schemaVersion":1,"groupID":id,"projectID":project,
                "train":["domain":"control","localID":"train"],"sourceRequestIdentity":hash,
                "inputsManifestSHA256":hash,"authoringSHA256":hash,"revision":0,"stopRequested":false,
                "variants":[["ordinal":1,"requestIdentity":hash,"status":"pending"]]]
            if index != 2 {record["createdAtUnixSeconds"]=Double(100+index)}
            try JSONSerialization.data(withJSONObject:record).write(to:directory.appendingPathComponent("group.json"))
        }
        let listing=try WorkflowExperimentHistoryReader.list(root:root,projectID:project)
        XCTAssertTrue(listing.issues.isEmpty)
        XCTAssertEqual(listing.entries.map(\.groupID),[ids[1],ids[0],ids[2]])
        XCTAssertEqual(listing.entries.first?.createdAtUnixSeconds,101)
        XCTAssertNil(listing.entries.last?.createdAtUnixSeconds)
        // An invalid timestamp must not become a fabricated date or poison valid siblings.
        let file=root.appendingPathComponent("workflow-experiments/\(project)/\(ids[0])/group.json")
        var invalid=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
        invalid["createdAtUnixSeconds"] = -1
        try JSONSerialization.data(withJSONObject:invalid).write(to:file)
        let damaged=try WorkflowExperimentHistoryReader.list(root:root,projectID:project)
        XCTAssertEqual(damaged.entries.count,2);XCTAssertEqual(damaged.issues.count,1)
    }
}
