import XCTest
import KwipMLCore
@testable import KwipMLApp
final class RunHistoryPolicyTests:XCTestCase {
    func testFuturePolicyCannotRewriteAcceptedRunAndMissingLegacyDefaultsToFull()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=RunHistoryPolicyStore(root:root,projectID:UUID().uuidString),id=UUID().uuidString
        XCTAssertEqual(try store.decision(id:id),.all)
        try store.save(.none)
        try store.capture(id:id,policy:store.load())
        try store.save(.all)
        XCTAssertEqual(try store.decision(id:id),.none)
        XCTAssertThrowsError(try store.capture(id:id,policy:.all))
        XCTAssertEqual(try store.load(),.all)
    }
    func testCorruptPolicyRefusesRatherThanEnablingCapture()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:root)}
        let store=RunHistoryPolicyStore(root:root,projectID:UUID().uuidString)
        try store.save(.none)
        try Data("{}".utf8).write(to:store.fileURL)
        XCTAssertThrowsError(try store.load())
        XCTAssertThrowsError(try store.decision(id:"../outside"))
    }
}
