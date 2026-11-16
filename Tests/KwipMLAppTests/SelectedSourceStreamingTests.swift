import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SelectedSourceStreamingTests: XCTestCase {
    func testPinnedCopyUsesBoundedChunksAndCleansCancelledOutput() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("selected.csv")
        let bytes=Data(repeating:65,count:3*65_536+17);try bytes.write(to:source)
        let receipt=try DiagnosticFileIO.copySelectedFile(source,root:root,parts:["captured.csv"],limit:bytes.count)
        XCTAssertEqual(receipt.bytes,bytes.count)
        XCTAssertEqual(receipt.sha256,WorkflowArchiveCodec.hash(bytes))
        XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent(receipt.path)),bytes)
        var calls=0
        XCTAssertThrowsError(try DiagnosticFileIO.copySelectedFile(source,root:root,parts:["cancelled.csv"],limit:bytes.count,checkCancellation:{
            calls+=1
            let size=(try? FileManager.default.attributesOfItem(atPath:root.appendingPathComponent("cancelled.csv").path)[.size] as? NSNumber)?.intValue ?? 0
            if size>0 {throw CancellationError()}
        }))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("cancelled.csv").path))
        XCTAssertEqual(try Data(contentsOf:source),bytes)
        XCTAssertThrowsError(try DiagnosticFileIO.copySelectedFile(source,root:root,parts:["captured.csv"],limit:bytes.count))
    }
    func testOversizeAndSymlinkSourcesAreRefusedWithoutPublishing() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("selected.csv");try Data([1,2,3]).write(to:source)
        let link=root.appendingPathComponent("link.csv");try FileManager.default.createSymbolicLink(at:link,withDestinationURL:source)
        for (url,limit) in [(source,2),(link,3)] {
            XCTAssertThrowsError(try DiagnosticFileIO.copySelectedFile(url,root:root,parts:["refused.csv"],limit:limit))
            XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("refused.csv").path))
        }
    }
}


extension SelectedSourceStreamingTests {
    func testChangingSourceDuringCopyCannotPublishHybridSnapshot() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("selected.csv")
        let bytes=Data(repeating:65,count:3*65_536);try bytes.write(to:source)
        var calls=0
        XCTAssertThrowsError(try DiagnosticFileIO.copySelectedFile(source,root:root,parts:["changed.csv"],limit:bytes.count,checkCancellation:{
            calls+=1
            let size=(try? FileManager.default.attributesOfItem(atPath:root.appendingPathComponent("changed.csv").path)[.size] as? NSNumber)?.intValue ?? 0
            if size>0 && calls<100 {try Data([9]).write(to:source);calls=100}
        }))
        XCTAssertFalse(FileManager.default.fileExists(atPath:root.appendingPathComponent("changed.csv").path))
        XCTAssertEqual(try Data(contentsOf:source),Data([9]))
    }
}

extension SelectedSourceStreamingTests {
    func testBorrowedSourceDescriptorCopiesFromStartWithoutChangingCallerOffsetOrOwnership() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("selected.bin");try Data([1,2,3]).write(to:source)
        let input=try FileHandle(forReadingFrom:source);defer {try? input.close()}
        try input.seek(toOffset:2)
        let receipt=try DiagnosticFileIO.copyRegularDescriptor(input.fileDescriptor,root:root,parts:["copy.bin"],limit:3)
        XCTAssertEqual(receipt.bytes,3)
        XCTAssertEqual(try Data(contentsOf:root.appendingPathComponent(receipt.path)),Data([1,2,3]))
        XCTAssertEqual(try input.read(upToCount:1),Data([3]))
    }
}
