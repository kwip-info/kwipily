import XCTest
import KwipMLCore
@testable import KwipMLApp

final class SklearnPreparedWorkerRequestTests: XCTestCase {
    func testPrivatePreparedRequestAndStrictRuntimeContract() throws {
        let selection = PreparedTrainingSelection(manifestPath: "/owned/input/preparation/manifest.json", manifestSHA256: String(repeating: "a", count: 64), graphDigest: String(repeating: "b", count: 64), trainingNode: "training", trainingPort: "table", validationNode: "validation", validationPort: "table", featureColumns: ["signal", "context"], targetColumn: "label")
        var request = try SklearnPreparedWorkerRequest(configuration: SklearnEstimatorConfiguration.defaultSettings, selection: selection)
        XCTAssertNil(try SklearnPreparedWorkerRequest.decode(request.encoded()).capturePredictions)
        request.capturePredictions = true
        XCTAssertEqual(try SklearnPreparedWorkerRequest.decode(request.encoded()), request)
        XCTAssertNoThrow(try SklearnRuntimeCapabilities.decode(JSONEncoder().encode(SklearnRuntimeCapabilities.expected)))
        var invalid = try JSONSerialization.jsonObject(with: request.encoded()) as! [String: Any]
        invalid["capturePredictions"] = 1
        XCTAssertThrowsError(try SklearnPreparedWorkerRequest.decode(JSONSerialization.data(withJSONObject: invalid)))
    }
    func testRealWorkerOutputAndPredictionTamper() throws {
        guard let root = ProcessInfo.processInfo.environment["KWIPML_SKLEARN_NATIVE_FIXTURE"] else { throw XCTSkip("Provide a real sklearn prepared run fixture.") }
        let directory = URL(fileURLWithPath: root)
        let request = try SklearnPreparedWorkerRequest.decode(Data(contentsOf: directory.appendingPathComponent("request.json")))
        let output = directory.appendingPathComponent("run")
        let result = try SklearnWorkflowResult.verify(directory: output, request: request)
        XCTAssertEqual(result.evaluation.count, 8)
        XCTAssertTrue(result.evaluation.logLoss.isFinite)
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.copyItem(at: output, to: scratch); defer { try? FileManager.default.removeItem(at: scratch) }
        let manifestPath = scratch.appendingPathComponent("manifest.json")
        var manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestPath)) as! [String: Any]
        var predictions = manifest["predictions"] as! [String: Any]
        predictions["decisionScores"] = Array(repeating: 999.0, count: 8); manifest["predictions"] = predictions
        try JSONSerialization.data(withJSONObject: manifest).write(to: manifestPath)
        XCTAssertThrowsError(try SklearnWorkflowResult.verify(directory: scratch, request: request))
    }
}
