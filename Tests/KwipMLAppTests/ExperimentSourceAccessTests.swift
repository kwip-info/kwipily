import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ExperimentSourceAccessTests:XCTestCase {
    private func fixture()throws->(URL,SourceAccess,URL,WorkflowNodeAddress) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let source=root.appendingPathComponent("original.csv")
        try Data("x,y\n1,2\n".utf8).write(to:source)
        return(root,SourceAccess(directory:root.appendingPathComponent("permissions")),source,.init(domain:.data,localID:"source"))
    }
    @MainActor func testCallerDirectoryCaptureUsesGrantAndPreservesOwnershipOnFailure() async throws {
        let(root,access,source,address)=try fixture()
        let directory=root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        try Data("held".utf8).write(to:directory.appendingPathComponent("owner.lock"))
        try access.remember(source)
        let result=try await access.captureExperimentInputs(paths:[address:source.path],directory:directory)
        XCTAssertEqual(result.directory,directory)
        try Data("changed".utf8).write(to:source)
        XCTAssertEqual(try Data(contentsOf:XCTUnwrap(result.sources[address]).url),Data("x,y\n1,2\n".utf8))
        let other=root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:other,withIntermediateDirectories:false)
        try Data("held".utf8).write(to:other.appendingPathComponent("owner.lock"))
        let bookmark=try XCTUnwrap(FileManager.default.contentsOfDirectory(at:access.directory,includingPropertiesForKeys:nil).first)
        try Data("corrupt".utf8).write(to:bookmark)
        do {_=try await access.captureExperimentInputs(paths:[address:source.path],directory:other);XCTFail("Corrupt grant bypassed")} catch {XCTAssertFalse(error is CancellationError)}
        XCTAssertEqual(try Data(contentsOf:other.appendingPathComponent("owner.lock")),Data("held".utf8))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:other.path),["owner.lock"])
    }
    @MainActor func testBackgroundCaptureRetainsPrivateCopies() async throws {
        let(root,access,source,address)=try fixture()
        try access.remember(source)
        let inputs=try await access.captureExperimentInputs(paths:[address:source.path],parent:root)
        try Data("changed".utf8).write(to:source)
        XCTAssertEqual(try Data(contentsOf:XCTUnwrap(inputs.sources[address]).url),Data("x,y\n1,2\n".utf8))
        XCTAssertNoThrow(try inputs.verify())
    }
    func testCancellationDuringCaptureDiscardsPartialGroup() async throws {
        let(root,_,source,address)=try fixture()
        let task=Task.detached {
            try WorkflowExperimentInputs.capture(addresses:[address],parent:root) {_ in
                let snapshot=try WorkflowCapturedSource.capture(source)
                withUnsafeCurrentTask {$0?.cancel()}
                return snapshot
            }
        }
        do {_=try await task.value;XCTFail("Cancelled capture must not return a group")}
        catch {XCTAssertTrue(error is CancellationError)}
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["original.csv"])
    }
    func testCorruptSavedGrantCannotFallBackToPlainPath() async throws {
        let(root,access,source,address)=try fixture()
        try access.remember(source)
        let bookmark=try XCTUnwrap(FileManager.default.contentsOfDirectory(at:access.directory,includingPropertiesForKeys:nil).first)
        try Data("corrupt bookmark".utf8).write(to:bookmark)
        do {
            _=try await access.captureExperimentInputs(paths:[address:source.path],parent:root)
            XCTFail("Saved permission failure cannot silently bypass its grant")
        } catch {XCTAssertFalse(error is CancellationError)}
        XCTAssertEqual(Set(try FileManager.default.contentsOfDirectory(atPath:root.path)),["original.csv","permissions"])
    }
    func testMissingSourceDoesNotLeavePartialGroup() async throws {
        let(root,access,_,address)=try fixture()
        do {
            _=try await access.captureExperimentInputs(paths:[address:root.appendingPathComponent("missing.csv").path],parent:root)
            XCTFail("Missing input must fail")
        } catch {XCTAssertFalse(error is CancellationError)}
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["original.csv"])
    }
}
