import XCTest
import KwipMLCore
@testable import KwipMLApp

final class DerivativeResultsPresentationTests: XCTestCase {
    private func request(_ id: String, start: Int64 = 2, targets: [DerivativeTargetResult]) -> DerivativeRequestResult {
        // Projection fixture for the already-verified result type; verification has separate tests.
        .init(request: .object(["id": .string(id), "output": .object(["node": .string("second"), "port": .string("out")]),
            "partition": .string("validation"), "row_start": .integer(start), "row_count": 2]), outputShape: [2, 1], targets: targets)
    }
    private func observed(_ values: [Double] = [0, 0]) -> DerivativeTargetResult {
        .init(target: .object(["kind": .string("input"), "node": .string("input"), "port": .string("out")]),
              name: "input.out", shape: [2, 1], dtype: .float32, status: .observed, values: values, reportedCause: nil)
    }
    private func unavailable(_ cause: String, dtype: ContractDType = .float32) -> DerivativeTargetResult {
        .init(target: .object(["kind": .string("parameter"), "node": .string("first"), "parameter": .string("weight")]),
              name: "layers.first.weight", shape: [2, 2], dtype: dtype, status: .unavailable, values: nil, reportedCause: cause)
    }
    private func project(_ requests: [DerivativeRequestResult]) throws -> DerivativeResultsPresentation {
        try .init(results: .init(runID: "verified-public-run", device: .cpu, requests: requests))
    }
    func testNamedRequestPhasePartitionAndRowRangePreserveCapturedOrder() throws {
        let p = try project([request("later", start: 4, targets: [observed()]), request("earlier", targets: [observed()])])
        XCTAssertEqual(p.runID, "verified-public-run"); XCTAssertEqual(p.device, .cpu)
        XCTAssertEqual(p.requests.map(\.id), ["later", "earlier"])
        let r = try XCTUnwrap(p.requests.first)
        XCTAssertEqual(r.output, "second.out"); XCTAssertEqual(r.phase, "Final snapshot evaluation")
        XCTAssertEqual(r.partition, "validation"); XCTAssertEqual(r.rowStart, 4); XCTAssertEqual(r.rowCount, 2)
        XCTAssertEqual(r.outputShape, [2, 1])
    }
    func testObservedZerosRemainShapedValuesAndNeverBecomeUnavailable() throws {
        let r = try XCTUnwrap(project([request("zero", targets: [observed()])]).requests.first)
        let t = try XCTUnwrap(r.targets.first)
        XCTAssertEqual(t.name, "input.out"); XCTAssertEqual(t.shape, [2, 1]); XCTAssertEqual(t.dtype, .float32)
        XCTAssertEqual(t.status, .observed); XCTAssertEqual(t.statusText, "Observed")
        XCTAssertEqual(t.tensor, LearningTensor(shape: [2, 1], values: [0, 0]))
        XCTAssertTrue(t.explanation.lowercased().contains("zero"))
        XCTAssertTrue(DerivativeResultsPresentation.explanation.contains("small-change effect"))
    }
    func testUnavailableCausesRemainDistinctFromObservedTensorAndTargetOrder() throws {
        let causes = ["integerInput", "frozenParameter", "disconnected", "sparseGradient"]
        for cause in causes {
            let r = try XCTUnwrap(project([request("effects", targets: [unavailable(cause), observed([1, -2])])]).requests.first)
            XCTAssertEqual(r.targets.map(\.name), ["layers.first.weight", "input.out"])
            let t = try XCTUnwrap(r.targets.first)
            XCTAssertEqual(t.status, .unavailable); XCTAssertEqual(t.statusText, "Unavailable"); XCTAssertNil(t.tensor)
            XCTAssertFalse(t.explanation.isEmpty); XCTAssertFalse(t.explanation.lowercased().contains("observed zero"))
            XCTAssertEqual(r.targets.last?.tensor?.values, [1, -2])
        }
    }
    func testEmptyPublicResultsAndMalformedProjectionBoundaries() throws {
        XCTAssertTrue(try project([]).requests.isEmpty)
        XCTAssertThrowsError(try project([request("invalid", start: -1, targets: [observed()])]))
        XCTAssertThrowsError(try project([request("invalid", targets: [observed([1])])]))
        XCTAssertThrowsError(try project([request("invalid", targets: [unavailable("unknownCause")])]))
        XCTAssertThrowsError(try project(Array(repeating: request("repeat", targets: [observed()]), count: 33)))
    }
}
