import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class UniversalSampleCaptureIdentityTests: XCTestCase {
    func testCaptureChangesRequestIdentityWithoutChangingCompiledModelOrSource() throws {
        let (root, owner, runtime) = try WorkflowExecutionRequestTests().setup()
        defer { try? FileManager.default.removeItem(at: root) }
        let budget = try ExecutionPreferences.factory.resolve().budget
        let train = WorkflowNodeAddress(domain: .control, localID: "train")
        XCTAssertThrowsError(try WorkflowExecutionRequest.capture(from: owner, train: train, runtime: runtime, sampleCapture: .factory))
        let first = try WorkflowExecutionRequest.capture(from: owner, train: train, runtime: runtime,
            executionBudget: budget, sampleCapture: .factory)
        let second = try WorkflowExecutionRequest.capture(from: owner, train: train, runtime: runtime,
            executionBudget: budget, sampleCapture: .init(activationValues: false, predictionRows: false))
        let repeated = try WorkflowExecutionRequest.capture(from: owner, train: train, runtime: runtime,
            executionBudget: budget, sampleCapture: .factory)
        XCTAssertNotEqual(first.requestIdentity, second.requestIdentity)
        XCTAssertEqual(first.requestIdentity, repeated.requestIdentity)
        XCTAssertEqual(first.document, second.document)
        XCTAssertEqual(first.plan, second.plan)
        XCTAssertEqual(first.modelRegistrySHA256, second.modelRegistrySHA256)
        XCTAssertEqual(first.sampleCapture, .init(activationValues:true,predictionRows:true,performanceMeasurements:false,retainComponentOutputs:true,retainDiagnosticDetails:true))
        XCTAssertEqual(second.sampleCapture?.predictionRows, false)
    }
    func testWriterSelectsFormatThreeAndStrictlyRejectsDowngradeOrPartialPolicy() throws {
        let (root, owner, runtime) = try WorkflowExecutionRequestTests().setup()
        defer { try? FileManager.default.removeItem(at: root) }
        let request = try WorkflowExecutionRequest.capture(from: owner,
            train: .init(domain: .control, localID: "train"), runtime: runtime,
            executionBudget: ExecutionPreferences.factory.resolve().budget,
            sampleCapture: .init(activationValues: false, predictionRows: true))
        let folder = root.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("source.csv"), bytes = Data("a,b,label\n1,2,0\n".utf8)
        try bytes.write(to: source)
        let sources = Dictionary(uniqueKeysWithValues: request.plan.sourceAddresses.map {
            ($0, WorkflowCapturedSource(url: source, sha256: WorkflowArchiveCodec.hash(bytes), byteCount: bytes.count))
        })
        _ = try WorkflowArchiveCaptureWriter.write(request: request, runID: UUID(), sources: sources,
            directory: folder, modelRegistry: owner.modelRegistry, dataRegistry: owner.dataRegistry)
        let data = try Data(contentsOf: folder.appendingPathComponent("capture.json"))
        let capture = try WorkflowArchiveCodec.decodeCapture(data)
        XCTAssertEqual(capture.captureFormatVersion, 3)
        XCTAssertEqual(capture.sampleCapture, request.sampleCapture)
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(capture), data)
        let edits: [(inout [String: Any]) -> Void] = [
            { $0["captureFormatVersion"] = 2 }, { $0.removeValue(forKey: "captureFormatVersion") },
            { $0.removeValue(forKey: "executionBudget") }, { $0.removeValue(forKey: "sampleCapture") },
            { $0["sampleCapture"] = NSNull() },
            { $0["sampleCapture"] = ["schemaVersion": 1, "activationValues": false, "predictionRows": true, "unknown": false] }
        ]
        for edit in edits {
            var value = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any]); edit(&value)
            XCTAssertThrowsError(try WorkflowArchiveCodec.decodeCapture(JSONSerialization.data(withJSONObject: value)))
        }
        var legacy = capture; legacy.sampleCapture = nil; legacy.captureFormatVersion = 2
        let old = try WorkflowArchiveCodec.encodeCapture(legacy)
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(old)), old)
        legacy.captureFormatVersion = nil; legacy.executionBudget = nil
        let oldest = try WorkflowArchiveCodec.encodeCapture(legacy)
        XCTAssertEqual(try WorkflowArchiveCodec.encodeCapture(WorkflowArchiveCodec.decodeCapture(oldest)), oldest)
    }
}
