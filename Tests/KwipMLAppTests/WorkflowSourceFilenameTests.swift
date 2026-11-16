import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowSourceFilenameTests: XCTestCase {
    func testCapturesKeepFormatWithoutPrivateBasename() {
        XCTAssertEqual(WorkflowSourceFilename.archive(index: 2, source: URL(fileURLWithPath: "/private/Payroll.XLSX")), "source-2.xlsx")
        XCTAssertEqual(WorkflowSourceFilename.archive(index: 0, source: URL(fileURLWithPath: "/private/no-extension")), "source-0.data")
        XCTAssertTrue(WorkflowSourceFilename.validArchive("source-0.csv", index: 0))
        for name in ["source-0../x", "source-0.XLSX", "source-0.", "source-00.xlsx", "source-1.xlsx"] {
            XCTAssertFalse(WorkflowSourceFilename.validArchive(name, index: 0))
        }
    }
    func testExperimentPreservesFormatAndPinsBytesAfterOriginalRemoval() throws {
        let parent = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: parent) }
        let original = parent.appendingPathComponent("private-book.xlsx"), payload = Data([0x50,0x4b,0x03,0x04,1,2,3])
        try payload.write(to: original)
        let address = WorkflowNodeAddress(domain: .data, localID: "source")
        let result = try WorkflowExperimentInputs.capture(addresses: [address], parent: parent) { _ in
            WorkflowCapturedSource(url: original, sha256: WorkflowPhaseArtifacts.hash(payload), byteCount: payload.count)
        }
        try FileManager.default.removeItem(at: original)
        let captured = try XCTUnwrap(result.sources[address])
        XCTAssertEqual(captured.url.lastPathComponent, "source-0.xlsx")
        XCTAssertEqual(try Data(contentsOf: captured.url), payload)
        XCTAssertNoThrow(try result.verify())
        try Data([0,1,2,3,4,5,6]).write(to: captured.url)
        XCTAssertThrowsError(try result.verify())
    }
}
