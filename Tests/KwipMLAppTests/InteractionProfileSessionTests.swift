import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class InteractionProfileSessionTests: XCTestCase {
    func testOnlyExplicitBoundedDevelopmentCaptureIsAllowed() {
        let env = ["KWIPML_INTERACTION_PROFILE":"/tmp/kwipml-profiler-test.json", "KWIPML_INTERACTION_PROFILE_SECONDS":"10"]
        XCTAssertNil(InteractionProfileSession.request(bundleIdentifier:"info.kwip.KwipML", environment:env))
        XCTAssertNil(InteractionProfileSession.request(bundleIdentifier:nil, environment:env))
        XCTAssertNil(InteractionProfileSession.request(bundleIdentifier:"info.kwip.KwipML.POC", environment:[:]))
        XCTAssertEqual(InteractionProfileSession.request(bundleIdentifier:"info.kwip.KwipML.POC", environment:env)?.seconds,10)
        for value in ["nan","inf","0","301","-1"] {
            var changed=env;changed["KWIPML_INTERACTION_PROFILE_SECONDS"]=value
            XCTAssertNil(InteractionProfileSession.request(bundleIdentifier:"info.kwip.KwipML.POC", environment:changed))
        }
        var changed=env;changed["KWIPML_INTERACTION_PROFILE"]="relative.json"
        XCTAssertNil(InteractionProfileSession.request(bundleIdentifier:"info.kwip.KwipML.POC", environment:changed))
    }
    func testCapturePublicationIsCompleteAndNeverOverwritesExistingFile() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        defer {try? FileManager.default.removeItem(at:folder)}
        let url=folder.appendingPathComponent("capture.json")
        let recorder=InteractionProfiler();recorder.start();recorder.measure(.selection) {}
        let report=recorder.stop()
        try InteractionProfileSession.writeCapture(report,to:url)
        let bytes=try Data(contentsOf:url)
        XCTAssertEqual(try JSONDecoder().decode(InteractionProfiler.Report.self,from:bytes).samples.count,1)
        XCTAssertThrowsError(try InteractionProfileSession.writeCapture(report,to:url))
        XCTAssertEqual(try Data(contentsOf:url),bytes)
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath:folder.path),["capture.json"])
    }
}
