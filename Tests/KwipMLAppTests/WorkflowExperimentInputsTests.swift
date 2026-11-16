import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentInputsTests:XCTestCase {
    private func root()throws->URL {
        let url=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:url,withIntermediateDirectories:true)
        addTeardownBlock {try? FileManager.default.removeItem(at:url)}
        return url
    }
    func testCallerDirectorySurvivesPartialReadAndNeverRewritesExistingData()throws {
        let directory=try root(),a=WorkflowNodeAddress(domain:.data,localID:"a"),b=WorkflowNodeAddress(domain:.data,localID:"b")
        for name in ["ownership.json","owner.lock","activity.lock"] {try Data(name.utf8).write(to:directory.appendingPathComponent(name))}
        var reads=0
        XCTAssertThrowsError(try WorkflowExperimentInputs.captureBytes(addresses:[a,b],directory:directory) {_ in
            reads+=1;if reads==2 {throw DataFlowContractError("read failed")};return Data("x\n1\n".utf8)
        })
        XCTAssertEqual(reads,2)
        for name in ["ownership.json","owner.lock","activity.lock"] {XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent(name)),Data(name.utf8))}
        let source=directory.appendingPathComponent("source-0.csv"),captured=try Data(contentsOf:source)
        XCTAssertThrowsError(try WorkflowExperimentInputs.captureBytes(addresses:[a],directory:directory) {_ in Data("replacement".utf8)})
        XCTAssertEqual(try Data(contentsOf:source),captured)
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("inputs.json").path))
    }
    func testCallerDirectoryIDIsManifestIdentity()throws {
        let directory=try root(),address=WorkflowNodeAddress(domain:.data,localID:"a")
        try Data("held".utf8).write(to:directory.appendingPathComponent("owner.lock"))
        let result=try WorkflowExperimentInputs.captureBytes(addresses:[address],directory:directory) {_ in Data("x\n1\n".utf8)}
        XCTAssertEqual(result.directory,directory)
        let manifest=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:directory.appendingPathComponent("inputs.json"))) as? [String:Any])
        XCTAssertEqual(manifest["groupID"] as? String,directory.lastPathComponent)
        XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent("owner.lock")),Data("held".utf8))
        XCTAssertNoThrow(try result.verify())
    }
    func testCancellationInCallerCapturePreservesOwnershipAndPartialBytes() async throws {
        let directory=try root(),a=WorkflowNodeAddress(domain:.data,localID:"a"),b=WorkflowNodeAddress(domain:.data,localID:"b")
        try Data("owned".utf8).write(to:directory.appendingPathComponent("ownership.json"))
        let task=Task.detached {
            try WorkflowExperimentInputs.captureBytes(addresses:[a,b],directory:directory) {address in
                if address == b {withUnsafeCurrentTask {$0?.cancel()}}
                return Data("x\n1\n".utf8)
            }
        }
        do {_=try await task.value;XCTFail("Cancelled capture returned inputs")} catch {XCTAssertTrue(error is CancellationError)}
        XCTAssertEqual(try Data(contentsOf:directory.appendingPathComponent("ownership.json")),Data("owned".utf8))
        XCTAssertTrue(FileManager.default.fileExists(atPath:directory.appendingPathComponent("source-0.csv").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("inputs.json").path))
    }
    func testCapturesOnceAndRetainsBytesAfterOriginalChanges()throws {
        let root=try root(),file=root.appendingPathComponent("original.csv")
        let bytes=Data("x,y\n1,2\n".utf8);try bytes.write(to:file)
        let address=WorkflowNodeAddress(domain:.data,localID:"source")
        var calls=0
        let inputs=try WorkflowExperimentInputs.capture(addresses:[address],parent:root) {_ in
            calls+=1;return try WorkflowCapturedSource.capture(file)
        }
        try Data("x,y\n9,9\n".utf8).write(to:file)
        XCTAssertEqual(calls,1)
        let captured=try XCTUnwrap(inputs.sources[address])
        XCTAssertEqual(try Data(contentsOf:captured.url),bytes)
        XCTAssertNotEqual(captured.url,file)
        XCTAssertNoThrow(try inputs.verify())
        XCTAssertEqual(try WorkflowExperimentInputs.open(directory:inputs.directory,expectedManifestSHA256:inputs.manifestSHA256),inputs)
        try Data("tampered".utf8).write(to:captured.url)
        XCTAssertThrowsError(try inputs.verify())
        XCTAssertThrowsError(try WorkflowExperimentInputs.open(directory:inputs.directory,expectedManifestSHA256:inputs.manifestSHA256))
    }
    func testManifestMutationAndSourceSymlinkReject()throws {
        let root=try root(),file=root.appendingPathComponent("source.csv")
        try Data("x\n1\n".utf8).write(to:file)
        let address=WorkflowNodeAddress(domain:.data,localID:"source")
        let inputs=try WorkflowExperimentInputs.capture(addresses:[address],parent:root) {_ in
            try WorkflowCapturedSource.capture(file)
        }
        let manifest=inputs.directory.appendingPathComponent("inputs.json")
        let bytes=try Data(contentsOf:manifest)
        try (bytes+Data(" ".utf8)).write(to:manifest)
        XCTAssertThrowsError(try inputs.verify(),"Even decodable changed manifest bytes require the original pin")
        try bytes.write(to:manifest)
        let copied=try XCTUnwrap(inputs.sources[address]).url
        try FileManager.default.removeItem(at:copied)
        try FileManager.default.createSymbolicLink(at:copied,withDestinationURL:file)
        XCTAssertThrowsError(try inputs.verify(),"A matching external file must not replace a private source copy")
    }
    func testFailedCaptureRemovesOnlyItsPartialGroup()throws {
        let root=try root(),sentinel=root.appendingPathComponent("keep")
        try Data("keep".utf8).write(to:sentinel)
        let a=WorkflowNodeAddress(domain:.data,localID:"a"),b=WorkflowNodeAddress(domain:.data,localID:"b")
        var calls=0
        XCTAssertThrowsError(try WorkflowExperimentInputs.capture(addresses:[a,b],parent:root) {_ in
            calls+=1
            if calls==2 {throw DataFlowContractError("unavailable")}
            return try WorkflowCapturedSource.capture(sentinel)
        })
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:root.path),["keep"])
    }
    func testRejectsDuplicateWrongDomainAndMismatchedSnapshot()throws {
        let root=try root(),file=root.appendingPathComponent("source.csv")
        try Data("x\n1\n".utf8).write(to:file)
        let a=WorkflowNodeAddress(domain:.data,localID:"a")
        for addresses in [[a,a],[.init(domain:.model,localID:"a")]] {
            XCTAssertThrowsError(try WorkflowExperimentInputs.capture(addresses:addresses,parent:root) {_ in
                XCTFail("Must reject before reading");return try WorkflowCapturedSource.capture(file)
            })
        }
        XCTAssertThrowsError(try WorkflowExperimentInputs.capture(addresses:[a],parent:root) {_ in
            .init(url:file,sha256:String(repeating:"0",count:64),byteCount:4)
        })
    }
}
