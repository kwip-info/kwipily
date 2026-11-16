import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TableViewQueryTests: XCTestCase {
    @MainActor func testRecordedDisplayLabelsPreserveColumnIdentityAndSortIndex() {
        let page=TableViewPage(schemaVersion:1,columns:[
            ["name":.string("mean_0"),"displayLabel":.string("Mean · usage"),"dtype":.string("float64"),"nullable":false],
            ["name":.string("rows"),"dtype":.string("int64"),"nullable":false]],rows:[],positions:[],offset:0,nextOffset:nil,totalRows:0,matchedRows:0,cellTruncated:false,sourceSHA256:String(repeating:"a",count:64),materializedIndex:nil)
        let columns=TableViewOutputView.columns(page)
        XCTAssertEqual(columns.map(\.label),["Mean · usage","rows"])
        XCTAssertEqual(columns.map(\.name),["mean_0","rows"])
        XCTAssertEqual(columns.map(\.id),[0,1])
    }

    func testStructuredWorkerFailureRetainsUsefulBoundedMessageOnly() {
        XCTAssertEqual(TableViewQueryExecution.boundedWorkerError(["event":"error", "message":"  Exact managed query authority refused.\n"]), "Exact managed query authority refused.")
        XCTAssertNil(TableViewQueryExecution.boundedWorkerError(["event":"tableViewPage", "message":"not an error"]))
        XCTAssertNil(TableViewQueryExecution.boundedWorkerError(["event":"error", "message":12]))
        XCTAssertNil(TableViewQueryExecution.boundedWorkerError(["event":"error", "message":"  "]))
        XCTAssertEqual(TableViewQueryExecution.boundedWorkerError(["event":"error", "message":String(repeating:"x",count:5000)])?.unicodeScalars.count, 2000)
    }

    private var query: TableViewQuery { .init(offset: 0, limit: 100, search: "", sourceSHA256: String(repeating: "a", count: 64), indexSHA256: String(repeating: "b", count: 64)) }
    private var event: [String: Any] {
        ["event": "tableViewPage", "schemaVersion": 1,
         "columns": [["name": "value", "dtype": "string", "nullable": true]],
         "rows": [["first"], [NSNull()]], "positions": [0, 1], "offset": 0,
         "nextOffset": 2, "totalRows": 501_000, "matchedRows": 501_000,
         "cellTruncated": false, "sourceSHA256": String(repeating: "a", count: 64)]
    }
    func testPageKeepsOnlyReturnedRowsAndRejectsForeignOrUnboundedEvidence() throws {
        let page = try TableViewPage.decode(JSONSerialization.data(withJSONObject: event), query: query)
        XCTAssertEqual(page.rows.count, 2); XCTAssertEqual(page.totalRows, 501_000)
        let mutations: [(String, Any)] = [("sourceSHA256", String(repeating: "c", count: 64)), ("positions", [1, 1]), ("nextOffset", 100), ("rows", [[String(repeating: "x", count: 2001)], [NSNull()]])]
        for (key, value) in mutations {
            var changed = event; changed[key] = value
            XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject: changed), query: query))
        }
    }
    func testQueryBoundsAndExactInputPaths() throws {
        XCTAssertNoThrow(try query.validate())
        XCTAssertThrowsError(try TableViewQuery(offset: -1, limit: 100, search: "", sourceSHA256: query.sourceSHA256, indexSHA256: query.indexSHA256).validate())
        XCTAssertThrowsError(try TableViewQuery(offset: 0, limit: 101, search: "", sourceSHA256: query.sourceSHA256, indexSHA256: query.indexSHA256).validate())
        let root = URL(fileURLWithPath: "/private/owned")
        let files = [WorkflowArchiveFile(path: "input/view-query.json", bytes: 200, sha256: query.sourceSHA256), WorkflowArchiveFile(path: "input/view.sqlite3", bytes: 50_000_000, sha256: query.indexSHA256)]
        XCTAssertNoThrow(try WorkflowTableViewQueryToolContract.validate(arguments: ["--query-table-view", "/private/owned/input/view-query.json"], files: files, directory: root))
        XCTAssertThrowsError(try WorkflowTableViewQueryToolContract.validate(arguments: ["--query-table-view", "/private/other/input/view-query.json"], files: files, directory: root))
        XCTAssertThrowsError(try WorkflowTableViewQueryToolContract.fileLimit(path: "input/../view.sqlite3"))
    }
    func testViewReferenceRequiresCompleteCoverageAndBothPins() throws {
        let index: [String: Any] = ["path": "view-001.sqlite3", "bytes": 12000, "sha256": query.indexSHA256,
            "kind": "tableViewIndex", "nodeID": "view", "port": "table", "sourceSHA256": query.sourceSHA256]
        var manifest: [String: Any] = [
            "artifacts": [index, ["path": "node-000.json", "bytes": 50, "sha256": query.sourceSHA256, "kind": "table", "nodeID": "view", "port": "table"]],
            "tableViews": ["view": ["schemaVersion": 1, "kind": "tableViewIndex", "sourceSHA256": query.sourceSHA256, "rowCount": 2,
                "columns": [["name": "value", "dtype": "string", "nullable": true]], "index": index]],
            "executionCoverage": ["nodes": ["view": ["completeness": "complete", "rowCounts": ["table": 2]]]]]
        func decode() throws -> [String: JSONValue] { try JSONDecoder().decode([String: JSONValue].self, from: JSONSerialization.data(withJSONObject: manifest)) }
        let root = URL(fileURLWithPath: "/private/verified/artifacts")
        let references = try TableViewArtifactReference.parse(manifest: decode(), directory: root, expectedNodes: ["view"])
        XCTAssertEqual(references["view"]?.sourceSHA256, query.sourceSHA256)
        var views=try XCTUnwrap(manifest["tableViews"] as? [String:[String:Any]])
        views["view"]?["schemaVersion"]=2;manifest["tableViews"]=views
        XCTAssertEqual(try TableViewArtifactReference.parse(manifest:decode(),directory:root,expectedNodes:["view"])["view"]?.sourceSHA256,query.sourceSHA256)
        views["view"]?["schemaVersion"]=3;manifest["tableViews"]=views
        XCTAssertThrowsError(try TableViewArtifactReference.parse(manifest:decode(),directory:root,expectedNodes:["view"]))
        views["view"]?["schemaVersion"]=2;manifest["tableViews"]=views
        manifest["executionCoverage"] = ["nodes": ["view": ["completeness": "sampled", "rowCounts": ["table": 2]]]]
        XCTAssertThrowsError(try TableViewArtifactReference.parse(manifest: decode(), directory: root, expectedNodes: ["view"]))
    }

    func testSortedRequestRoundTripAllowsNonmonotonicDistinctRowPositions() throws {
        var sorted = query; sorted.sortColumn = 0; sorted.sortDescending = true
        XCTAssertEqual(try TableViewQuery.decode(JSONEncoder().encode(sorted)), sorted)
        var response = event; response["positions"] = [100, 3]
        XCTAssertNoThrow(try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: sorted))
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: query))
        response["positions"] = [3, 3]
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: sorted))
        sorted.sortColumn = 128
        XCTAssertThrowsError(try sorted.validate())
    }

    func testDirectPinnedRangePageHasNoIndexAndCannotClaimGlobalSearch() throws {
        var table = TableViewQuery(schemaVersion:3,offset:0,limit:100,search:"",sourceSHA256:query.sourceSHA256,indexSHA256:query.sourceSHA256,inputKind:"table")
        var response=event;response["verification"]="pinnedRanges"
        let page=try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:table)
        XCTAssertEqual(page.verification,"pinnedRanges");XCTAssertNil(page.materializedIndex)
        table.search="first"
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:table))
        table.search="";table.sortColumn=0
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:table))
        table.sortColumn=nil;response["verification"]="fullAudit"
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:table))
        response["verification"]="pinnedRanges";response["positions"]=[1,2]
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:table))
    }

    func testLazyTableQueryPinsInputAndAdmitsOnlyFixedGeneratedIndex() throws {
        var table=query;table.schemaVersion=3;table.inputKind="table"
        XCTAssertThrowsError(try table.validate(),"The lazy input must pin the same captured table as its source.")
        table = .init(schemaVersion:3,offset:0,limit:100,search:"",sourceSHA256:query.sourceSHA256,indexSHA256:query.sourceSHA256,inputKind:"table")
        XCTAssertEqual(try TableViewQuery.decode(JSONEncoder().encode(table)),table)
        var response=event
        response["materializedIndex"]=["path":"artifacts/view.sqlite3","bytes":12000,"sha256":query.indexSHA256]
        XCTAssertNoThrow(try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:table))
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:query),"Legacy index queries cannot introduce another output file.")
        response["materializedIndex"]=["path":"../view.sqlite3","bytes":12000,"sha256":query.indexSHA256]
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:response),query:table))
        table.schemaVersion=2
        XCTAssertThrowsError(try table.validate())
    }

}

extension TableViewQueryTests {
    func testTemporalQueryPageAdmitsOnlyCanonicalIntegersAndExactMetadata() throws {
        var response = event
        let column: [String: Any] = ["name": "moment", "dtype": "datetime", "nullable": true, "temporal": ["version": 1, "unit": "microsecond", "timezone": "UTC"]]
        response["columns"] = [column]; response["rows"] = [[Int64(-1)], [NSNull()]]
        let page = try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: query)
        XCTAssertEqual(page.rows.first, [.integer(-1)])
        for bad: Any in ["1970-01-01T00:00:00Z", true, 0.5, Int64.max] {
            response["rows"] = [[bad], [NSNull()]]
            XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: query))
        }
        response["rows"] = [[Int64(0)], [NSNull()]]
        for metadata: [String: Any] in [["version": 1, "unit": "millisecond", "timezone": "UTC"], ["version": 1, "unit": "microsecond"], ["version": 1, "unit": "microsecond", "timezone": "UTC", "extra": true]] {
            var invalid = column; invalid["temporal"] = metadata; response["columns"] = [invalid]
            XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: query))
        }
    }
}


extension TableViewQueryTests {
    func testDisplayLabelsPreserveLegacyScalarAndTemporalPageCompatibility() throws {
        let schemas: [[String: Any]] = [
            ["name": "encoded", "dtype": "float64", "nullable": false],
            ["name": "moment", "dtype": "datetime", "nullable": false, "temporal": ["version": 1, "unit": "microsecond", "timezone": "UTC"]]
        ]
        for schema in schemas {
            for label in ["Category = café", "", String(repeating: "x", count: 400)] {
                var column = schema; column["displayLabel"] = label
                var response = event; response["columns"] = [column]; response["rows"] = [[Int64(0)], [Int64(1)]]
                XCTAssertNoThrow(try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: query))
            }
            for label: Any in [true, 42, NSNull(), String(repeating: "x", count: 401)] {
                var column = schema; column["displayLabel"] = label
                var response = event; response["columns"] = [column]; response["rows"] = [[Int64(0)], [Int64(1)]]
                XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject: response), query: query))
            }
        }
    }
}
