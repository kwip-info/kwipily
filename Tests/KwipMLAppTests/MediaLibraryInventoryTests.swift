import XCTest
@testable import KwipMLApp
@testable import KwipMLCore

final class MediaLibraryInventoryTests:XCTestCase {
    func testOwnedMetadataOnlyInventoryProtectsIncompleteRecordsAndFiltersProject()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString,operation=UUID().uuidString
        let directory=root.appendingPathComponent("workflow-operations/\(project)/\(operation)/artifacts")
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
        let pin=String(repeating:"a",count:64),name="media-"+pin+".kwipmedia"
        let manifest:[String:Any]=["schemaVersion":1,"kind":"dataflowExecution","status":"completed","artifacts":[["kind":"audioSamples","path":name,"sha256":pin,"bytes":12]]]
        try JSONSerialization.data(withJSONObject:manifest).write(to:directory.appendingPathComponent("manifest.json"))
        try Data(repeating:0,count:12).write(to:directory.appendingPathComponent(name))
        try FileManager.default.createDirectory(at:root.appendingPathComponent("workflow-operations/\(project)/\(UUID().uuidString)"),withIntermediateDirectories:true)
        let report=try MediaLibraryInventory.scan(workspace:root,projectID:project)
        XCTAssertEqual(report.items.count,1);XCTAssertEqual(report.items[0].bytes,12)
        XCTAssertEqual(report.protectedRecords,1)
        XCTAssertTrue(try MediaLibraryInventory.scan(workspace:root,projectID:UUID().uuidString).items.isEmpty)
        try FileManager.default.removeItem(at:directory.appendingPathComponent(name))
        let missing=try MediaLibraryInventory.scan(workspace:root,projectID:project)
        XCTAssertEqual(missing.items.count,0);XCTAssertEqual(missing.protectedRecords,2)
    }
    func testInventoryNeverFollowsExternalSymbolicLinks()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        try FileManager.default.createSymbolicLink(at:root.appendingPathComponent("workflow-runs"),withDestinationURL:root)
        XCTAssertThrowsError(try MediaLibraryInventory.scan(workspace:root))
    }
    func testRecordCategoriesDoNotCallValidNonmediaOrUnknownFormatsCorrupt()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let project=UUID().uuidString
        func record(_ text:String?)throws {
            let directory=root.appendingPathComponent("workflow-operations/\(project)/\(UUID().uuidString)/artifacts")
            try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
            if let text {try Data(text.utf8).write(to:directory.appendingPathComponent("manifest.json"))}
        }
        try record(#"{"schemaVersion":1,"kind":"dataflowExecution","status":"completed","artifacts":[{"kind":"table","path":"table.parquet"}]}"#)
        try record(#"{"schemaVersion":1,"backendID":"sklearn","task":"clustering","stateFormat":"numpy"}"#)
        try record(#"{"schemaVersion":999,"kind":"futureFormat","artifacts":[]}"#)
        try record(nil)
        try record("broken JSON")
        try record(#"{"schemaVersion":1,"kind":"dataflowExecution","status":"completed","artifacts":null}"#)
        let before=try FileManager.default.subpathsOfDirectory(atPath:root.path).sorted()
        let report=try MediaLibraryInventory.scan(workspace:root)
        XCTAssertEqual(report.inspectedRecords,6)
        XCTAssertTrue(report.items.isEmpty)
        XCTAssertEqual(report.completedRecordsWithoutMedia,1)
        XCTAssertEqual(report.recordsWithoutManifest,1)
        XCTAssertEqual(report.otherManifestRecords,2)
        XCTAssertEqual(report.unreadableRecords,2)
        XCTAssertEqual(report.protectedRecords,5,"Categorizing unknown records must not relax protection")
        XCTAssertEqual(try FileManager.default.subpathsOfDirectory(atPath:root.path).sorted(),before)
    }

}
