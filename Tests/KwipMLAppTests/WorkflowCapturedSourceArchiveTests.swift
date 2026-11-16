import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowCapturedSourceArchiveTests: XCTestCase {
    private let address = WorkflowNodeAddress(domain: .data, localID: "files")
    private func root() throws -> URL {
        let url = FileManager.default.temporaryDirectory.resolvingSymlinksInPath().appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }
    private func collection(in root: URL, metadataOnly: Bool = false, complete: Bool = true) throws -> WorkflowCapturedSource {
        let bytes = Data("immutable content".utf8), hash = WorkflowArchiveCodec.hash(bytes)
        let relative = "members/000000.png"
        var captures: [FileInputCapturedMember] = []
        if !metadataOnly {
            try DiagnosticFileIO.writeCapturedFileInputMember(bytes, root: root, parts: ["members", "000000.png"])
            captures = [.init(capturePath: relative, url: root.appendingPathComponent(relative), sha256: hash, byteCount: bytes.count)]
        }
        let manifest = FileInputCollectionManifest(schemaVersion: 1, kind: "kwipFileCollection", sourceMode: "directory", readMode: metadataOnly ? "metadata" : "content", complete: complete,
            members: [.init(relativePath: "photos/image.png", capturePath: metadataOnly ? nil : relative,
                            bytes: Int64(bytes.count), sha256: metadataOnly ? nil : hash, extension: "png", status: metadataOnly ? "metadata" : "content", modifiedAt: nil, createdAt: nil, error: nil)], selection: try FileInputConfiguration.validate(["sourceMode": .string("directory"), "readMode": .string(metadataOnly ? "metadata" : "content")]).filter { FileInputCollectionManifest.selectionKeys.contains($0.key) })
        let data = try manifest.encoded()
        try DiagnosticFileIO.writeImmutable(data, root: root, parts: ["manifest.json"], limit: 2_000_000)
        return .init(url: root.appendingPathComponent("manifest.json"), sha256: WorkflowArchiveCodec.hash(data), byteCount: data.count, collectionMembers: captures, isFileCollection: true)
    }
    func testCollectionRetainsExactManifestAndMemberBytesIndependentOfOriginal() throws {
        let original = try root(), archive = try root(), source = try collection(in: original)
        let entry = try WorkflowCapturedSourceArchive.publish(source, address: address, index: 0, root: archive, prefix: "inputs/")
        XCTAssertEqual(entry.path, "inputs/source-0.collection/\(source.sha256)/manifest.json")
        XCTAssertEqual(entry.collectionMembers?.count, 1)
        try FileManager.default.removeItem(at: original)
        let reopened = try WorkflowCapturedSourceArchive.open(entry, root: archive, prefix: "inputs/")
        XCTAssertTrue(reopened.isFileCollection)
        XCTAssertEqual(reopened.collectionMembers.first?.capturePath, "members/000000.png")
        XCTAssertEqual(try Data(contentsOf: XCTUnwrap(reopened.collectionMembers.first).url), Data("immutable content".utf8))
    }
    func testCollectionRejectsMissingChangedExtraAndSymlinkMembers() throws {
        let original = try root(), archive = try root(), source = try collection(in: original)
        let entry = try WorkflowCapturedSourceArchive.publish(source, address: address, index: 0, root: archive, prefix: "inputs/")
        let member = try XCTUnwrap(entry.collectionMembers?.first), url = archive.appendingPathComponent(member.path)
        let data = try Data(contentsOf: url)
        try Data("altered".utf8).write(to: url)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.open(entry, root: archive, prefix: "inputs/"))
        try data.write(to: url)
        let extra = url.deletingLastPathComponent().appendingPathComponent("not-declared.png")
        try data.write(to: extra)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.open(entry, root: archive, prefix: "inputs/"))
        try FileManager.default.removeItem(at: extra)
        try FileManager.default.removeItem(at: url)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.open(entry, root: archive, prefix: "inputs/"))
        try FileManager.default.createSymbolicLink(at: url, withDestinationURL: source.collectionMembers[0].url)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.open(entry, root: archive, prefix: "inputs/"))
    }
    func testMetadataOnlyCollectionKeepsEmptyInventoryAndIncompletePreviewIsRefused() throws {
        let source = try collection(in: root(), metadataOnly: true), archive = try root()
        let entry = try WorkflowCapturedSourceArchive.publish(source, address: address, index: 1, root: archive, prefix: "")
        XCTAssertEqual(entry.collectionMembers, [])
        XCTAssertTrue(try WorkflowCapturedSourceArchive.open(entry, root: archive, prefix: "").isFileCollection)
        let preview = try collection(in: root(), complete: false)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.publish(preview, address: address, index: 2, root: archive, prefix: ""))
    }
    func testDeclaredInventoryCannotDropAddOrRelocateMembers() throws {
        let original = try root(), archive = try root(), source = try collection(in: original)
        let missing = WorkflowCapturedSource(url: source.url, sha256: source.sha256, byteCount: source.byteCount, isFileCollection: true)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.publish(missing, address: address, index: 0, root: archive, prefix: ""))
        let elsewhere = original.appendingPathComponent("same-content.png")
        try Data("immutable content".utf8).write(to: elsewhere)
        let first = source.collectionMembers[0]
        let relocated = WorkflowCapturedSource(url: source.url, sha256: source.sha256, byteCount: source.byteCount,
            collectionMembers: [.init(capturePath: first.capturePath, url: elsewhere, sha256: first.sha256, byteCount: first.byteCount)], isFileCollection: true)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.publish(relocated, address: address, index: 0, root: archive, prefix: ""))
        let traversal = WorkflowArchiveSource(address: address, path: "source-0.collection/\(source.sha256)/manifest.json", bytes: source.byteCount, sha256: source.sha256,
            collectionMembers: [.init(path: "source-0.collection/../outside.png", bytes: 1, sha256: first.sha256)])
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.validate(traversal, prefix: ""))
    }
    func testLegacyScalarEnvelopeAndTenMBBoundArePreserved() throws {
        let original = try root(), archive = try root(), bytes = Data("x\n1\n".utf8)
        let url = original.appendingPathComponent("table.csv"); try bytes.write(to: url)
        let entry = try WorkflowCapturedSourceArchive.publish(WorkflowCapturedSource.capture(url), address: address, index: 0, root: archive, prefix: "inputs/")
        XCTAssertNil(entry.collectionMembers)
        let encoded = try WorkflowArchiveCodec.encode(entry)
        XCTAssertFalse(String(decoding: encoded, as: UTF8.self).contains("collectionMembers"))
        XCTAssertFalse(try WorkflowCapturedSourceArchive.open(entry, root: archive, prefix: "inputs/").isFileCollection)
        let large = Data(repeating: 1, count: 10_000_001)
        XCTAssertThrowsError(try DiagnosticFileIO.writeImmutable(large, root: archive, parts: ["ordinary.bin"], limit: large.count))
        XCTAssertThrowsError(try DiagnosticFileIO.writeCapturedFileInputMember(large, root: archive, parts: ["ordinary.bin"]))
        XCTAssertNoThrow(try DiagnosticFileIO.writeCapturedFileInputMember(large, root: archive, parts: ["members", "owned.bin"]))
        let oversized = WorkflowCapturedSource(url: url, sha256: entry.sha256, byteCount: bytes.count,
            collectionMembers: [.init(capturePath: "members/large.png", url: url, sha256: entry.sha256, byteCount: 67_108_865)], isFileCollection: true)
        XCTAssertThrowsError(try WorkflowCapturedSourceArchive.declaredByteCount(oversized))
    }
    func testExperimentCaptureRetainsClosureAndRejectsLaterMemberTampering() throws {
        let source = try collection(in: root()), parent = try root()
        let inputs = try WorkflowExperimentInputs.capture(addresses: [address], parent: parent) { _ in source }
        let captured = try XCTUnwrap(inputs.sources[address])
        XCTAssertTrue(captured.isFileCollection)
        XCTAssertEqual(captured.collectionMembers.count, 1)
        XCTAssertNoThrow(try inputs.verify())
        try Data("changed".utf8).write(to: captured.collectionMembers[0].url)
        XCTAssertThrowsError(try inputs.verify())
    }
}
