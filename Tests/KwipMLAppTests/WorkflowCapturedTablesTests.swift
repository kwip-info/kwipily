import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCapturedTablesTests:XCTestCase {
    func testAdmissionRequiresTheAcceptedManifestPinAndNeverInventsOldCaptureIndexes()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let bytes=try WorkflowArchiveCodec.encode(["artifacts":JSONValue.array([])])
        XCTAssertThrowsError(try WorkflowCapturedTables.admit(bytes:bytes,expectedSHA256:String(repeating:"0",count:64),directory:root,graph:.init(),runtime:runtime,storageLimitBytes:nil))
        XCTAssertNil(try WorkflowCapturedTables.admit(bytes:bytes,expectedSHA256:WorkflowArchiveCodec.hash(bytes),directory:root,graph:.init(),runtime:runtime,storageLimitBytes:nil))
    }
    func testCoverageCannotGrantAQueryForForeignOrTraversalArtifacts()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let pin=String(repeating:"a",count:64)
        for (node,path) in [("foreign","table.json"),("source","../table.json")] {
            let manifest:[String:JSONValue]=[
                "artifacts":.array([.object(["kind":.string("table"),"nodeID":.string(node),"port":.string("table"),"path":.string(path),"sha256":.string(pin),"bytes":10])]),
                "executionCoverage":.object(["nodes":.object([node:.object(["rowCounts":.object(["table":1])])])])]
            let bytes=try WorkflowArchiveCodec.encode(manifest)
            XCTAssertThrowsError(try WorkflowCapturedTables.admit(bytes:bytes,expectedSHA256:WorkflowArchiveCodec.hash(bytes),directory:root,
                graph:.init(nodes:[.init(id:"source",operatorID:"kwip.data.sourceCSV")]),runtime:runtime,storageLimitBytes:nil))
        }
    }
}
