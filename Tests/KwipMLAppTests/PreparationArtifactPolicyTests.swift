import XCTest
import KwipMLCore
@testable import KwipMLApp

final class PreparationArtifactPolicyTests: XCTestCase {
    func testLegacyCapacityDoesNotAdmitNewSegments() throws {
        let old = try JSONEncoder().encode(["artifacts": JSONValue.array([])])
        let policy = try PreparationArtifactPolicy(manifest: old)
        XCTAssertEqual(policy.totalBytes, 200_000_000)
        XCTAssertEqual(policy.fileBytes("ordinary.json"), 50_000_000)
        let unversioned: [String: JSONValue] = ["artifacts":.array([.object(["kind":.string("tableSegment"),"path":.string("table-pack-"+String(repeating:"a",count:32)+".kwipdata")])])]
        XCTAssertThrowsError(try PreparationArtifactPolicy(manifest: JSONEncoder().encode(unversioned)))
    }
    func testCapturedSegmentCapacityCannotWidenUnrelatedFiles() throws {
        let captured = try ExecutionWorkloadPolicy(preset:.balanced)
        let name = "table-pack-"+String(repeating:"a",count:32)+".kwipdata"
        var manifest: [String:JSONValue] = ["tableStorageVersion":.integer(1),"workloadPolicy":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(captured)),"artifacts":.array([.object(["kind":.string("tableSegment"),"path":.string(name)])])]
        let policy = try PreparationArtifactPolicy(manifest:JSONEncoder().encode(manifest))
        XCTAssertEqual(policy.totalBytes, Int(captured.limit(.retainedBytes)))
        XCTAssertEqual(policy.fileBytes(name),policy.totalBytes)
        XCTAssertEqual(policy.fileBytes("unlisted.kwipdata"),50_000_000)
        manifest["tableStorageVersion"] = .integer(2)
        XCTAssertThrowsError(try PreparationArtifactPolicy(manifest:JSONEncoder().encode(manifest)))
    }
}

extension PreparationArtifactPolicyTests {
    func testSegmentedQueryRequiresCapturedPolicyAndCanonicalPinnedInputs() throws {
        let workload = try ExecutionWorkloadPolicy(preset:.balanced)
        let name = "input/table-pack-"+String(repeating:"a",count:32)+".kwipdata"
        let files: [WorkflowArchiveFile] = [
            .init(path:"input/view-query.json",bytes:200,sha256:String(repeating:"b",count:64)),
            .init(path:"input/view.table.json",bytes:500,sha256:String(repeating:"c",count:64)),
            .init(path:name,bytes:60_000_000,sha256:String(repeating:"d",count:64))]
        let args = ["--query-table-view","/tmp/captured/input/view-query.json"]
        XCTAssertNoThrow(try WorkflowTableViewQueryToolContract.validate(arguments:args,files:files,directory:nil,workload:workload))
        XCTAssertThrowsError(try WorkflowTableViewQueryToolContract.validate(arguments:args,files:files,directory:nil))
        var wrong = files
        wrong[2] = .init(path:"input/table-pack-../../private.kwipdata",bytes:200,sha256:String(repeating:"d",count:64))
        XCTAssertThrowsError(try WorkflowTableViewQueryToolContract.validate(arguments:args,files:wrong,directory:nil,workload:workload))
        let small = try workload.overriding(["retainedBytes":1_000_000])
        XCTAssertThrowsError(try WorkflowTableViewQueryToolContract.validate(arguments:args,files:files,directory:nil,workload:small))
    }
}

extension PreparationArtifactPolicyTests {
    func testCapturedViewIndexUsesRetentionButCannotRaiseOrdinaryMetadataFrame() throws {
        let policy=try ExecutionWorkloadPolicy(preset:.balanced)
        let document:[String:JSONValue] = ["tableStorageVersion":.integer(1),"workloadPolicy":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(policy)),"artifacts":.array([.object(["kind":.string("tableViewIndex"),"path":.string("view-003.sqlite3")])])]
        let admitted=try PreparationArtifactPolicy(manifest:JSONEncoder().encode(document))
        XCTAssertEqual(admitted.fileBytes("view-003.sqlite3"),Int(policy.limit(.retainedBytes)))
        XCTAssertEqual(admitted.fileBytes("ordinary.json"),50_000_000)
        XCTAssertEqual(try WorkflowPreparationInspectionToolContract.fileLimit(path:"input/preparation/view-003.sqlite3",workload:policy),Int(policy.limit(.retainedBytes)))
    }
}
