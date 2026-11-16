import XCTest
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class FileImagePresentationTests: XCTestCase {
    func testPixelJumpCoordinatesAreBoundedWholeNumbers() {
        XCTAssertEqual(FileImagePixelNavigation.coordinate("0", limit: 16384), 0)
        XCTAssertEqual(FileImagePixelNavigation.coordinate("16383", limit: 16384), 16383)
        XCTAssertEqual(FileImagePixelNavigation.coordinate(" 0012\n", limit: 20), 12)
        for invalid in ["", " ", "-1", "+1", "1.5", "1e2", "16384", "999999999999999999999", "１２"] {
            XCTAssertNil(FileImagePixelNavigation.coordinate(invalid, limit: 16384), invalid)
        }
        XCTAssertEqual(FileImagePixelNavigation.coordinate("0", limit: 1), 0)
        XCTAssertNil(FileImagePixelNavigation.coordinate("1", limit: 1))
        XCTAssertNil(FileImagePixelNavigation.coordinate("0", limit: 0))
        XCTAssertNil(FileImagePixelNavigation.coordinate("0", limit: -1))
    }
    private var reference: [String: Any] {
        ["schemaVersion": 1, "kind": "imagePixelsRef", "sha256": String(repeating: "a", count: 64),
         "width": 20, "height": 10, "channels": 3, "dtype": "uint8", "layout": "HWC"]
    }
    func testTypedCellsExposeGeometryAndSearchMetadataWithoutRawReferenceJSON() throws {
        let grid = CapturedDataGrid(preview: ["columns": [["name":"image","dtype":"image"]], "rows": [[reference], [NSNull()]]])
        XCTAssertNotNil(grid.rows[0].cells[0].image)
        XCTAssertTrue(grid.rows[0].cells[0].text.contains("20 × 10"))
        XCTAssertFalse(grid.rows[0].cells[0].text.contains("sha256"))
        XCTAssertEqual(grid.filteredRows("uint8").count, 1)
        XCTAssertTrue(grid.filteredRows(String(repeating: "a", count: 64)).isEmpty)
        XCTAssertEqual(grid.displayRows(query: "", sort: .init(column: 0, descending: true)).map(\.index), [0,1])
        var invalid=reference;invalid["path"]="/private/original.png"
        XCTAssertNil(CapturedDataGrid.cell(invalid).image)
    }
    func testResolverRefusesUnlistedReferenceAndUnsafeInventory() throws {
        let ref = try FileImageReference.decode(JSONSerialization.data(withJSONObject: reference))
        let resolver = try FileImageArtifactResolver(directory: URL(fileURLWithPath: "/not-opened"), verifiedArtifacts: [])
        XCTAssertFalse(resolver.contains(ref))
        XCTAssertThrowsError(try resolver.read(ref))
        XCTAssertThrowsError(try FileImageArtifactResolver(directory: URL(fileURLWithPath: "/not-opened"), verifiedArtifacts: [
            .init(path: "image-../outside", bytes: 1, sha256: ref.sha256)]))
    }
    private func container(dtype: String = "uint8", layout: String = "HWC", mode: String = "RGB", channels: Int = 3, pixels: Data = Data([1,2,3,4,5,6])) throws -> (Data, FileImageReference) {
        let header: [String: Any] = ["schemaVersion":1,"kind":"imagePixels","width":2,"height":1,"channels":channels,"dtype":dtype,"layout":layout,"byteOrder":"little",
            "policy":["orientation":"stored","colorMode":"preserve","palette":"reject","actualMode":mode]]
        let raw = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys,.withoutEscapingSlashes])
        var data = Data("KWIPIMG1\n".utf8)
        for shift in [24,16,8,0] { data.append(UInt8((raw.count >> shift) & 255)) }
        data.append(raw); data.append(pixels)
        return (data, try FileImageReference(sha256: WorkflowArchiveCodec.hash(data), width: 2, height: 1, channels: channels, dtype: dtype, layout: layout))
    }
    func testVerifiedContainerOpensCapturedPixelsAndRejectsTamperAndSymlink() throws {
        let temporary = FileManager.default.temporaryDirectory
        guard let resolved = realpath(temporary.path, nil) else { throw WorkflowFileLeaseError.unsafePath }
        let parent = URL(fileURLWithPath: String(cString: resolved)); free(resolved)
        let folder = parent.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let (bytes, ref) = try container(), name = "image-\(ref.sha256).kwipimg"
        let file = folder.appendingPathComponent(name)
        try bytes.write(to: file)
        let resolver = try FileImageArtifactResolver(directory: folder, verifiedArtifacts: [.init(path:name,bytes:bytes.count,sha256:ref.sha256)])
        let value = try resolver.read(ref)
        XCTAssertEqual(FileImagePixels.values(value,x:1,y:0), ["4","5","6"])
        XCTAssertEqual(try FileImagePixels.thumbnail(value)?.width, 2)
        var changed = bytes; changed[changed.count-1] ^= 1; try changed.write(to:file)
        XCTAssertThrowsError(try resolver.read(ref))
        try FileManager.default.removeItem(at:file)
        let target=folder.appendingPathComponent("unlisted");try bytes.write(to:target)
        try FileManager.default.createSymbolicLink(at:file,withDestinationURL:target)
        XCTAssertThrowsError(try resolver.read(ref))
        let moved = folder.appendingPathExtension("moved")
        defer { try? FileManager.default.removeItem(at:moved) }
        try FileManager.default.moveItem(at:folder,to:moved)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:false)
        try bytes.write(to:folder.appendingPathComponent(name))
        XCTAssertThrowsError(try resolver.read(ref))
    }
    func testRawPixelAccessPreservesCHWAndLittleEndianWithoutDisplayNormalization() throws {
        let (bytes, ref) = try container(layout:"CHW",pixels:Data([1,4,2,5,3,6]))
        let value = try FileImageReference.decode(container:bytes,expected:ref)
        XCTAssertEqual(FileImagePixels.values(value,x:1,y:0), ["4","5","6"])
        let (wide, wideRef) = try container(dtype:"uint16",mode:"I;16",channels:1,pixels:Data([0x34,0x12,0xff,0xff]))
        let raw = try FileImageReference.decode(container:wide,expected:wideRef)
        XCTAssertEqual(FileImagePixels.values(raw,x:0,y:0), ["4660"])
        XCTAssertEqual(FileImagePixels.values(raw,x:1,y:0), ["65535"])
        XCTAssertNil(try FileImagePixels.thumbnail(raw))
    }

    func testPagedImageCellsRequireStrictTypedReferencesAndRefuseScalarSort() throws {
        var query = TableViewQuery(offset:0,limit:100,search:"",sourceSHA256:String(repeating:"b",count:64),indexSHA256:String(repeating:"c",count:64))
        var event: [String: Any] = ["event":"tableViewPage","schemaVersion":1,
            "columns":[["name":"pixels","dtype":"image","nullable":true]],"rows":[[reference]],"positions":[0],
            "offset":0,"nextOffset":NSNull(),"totalRows":1,"matchedRows":1,"cellTruncated":false,"sourceSHA256":query.sourceSHA256]
        XCTAssertNoThrow(try TableViewPage.decode(JSONSerialization.data(withJSONObject:event),query:query))
        query.sortColumn=0
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:event),query:query))
        query.sortColumn=nil
        var invalid=reference;invalid["path"]="original.png";event["rows"]=[[invalid]]
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:event),query:query))
        event["rows"]=[["/private/original.png"]]
        XCTAssertThrowsError(try TableViewPage.decode(JSONSerialization.data(withJSONObject:event),query:query))
    }

}
