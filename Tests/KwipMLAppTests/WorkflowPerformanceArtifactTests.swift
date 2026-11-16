import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowPerformanceArtifactTests: XCTestCase {
    private let graph = ExampleFlows.parallelClassifier
    private let policy = TrainingSampleCapturePolicy(activationValues: false, predictionRows: false, performanceMeasurements: true)
    private let graphHash = String(repeating: "a", count: 64)
    private func artifact() throws -> [String:JSONValue] {
        let order = GraphValidator.validate(graph, registry: try .builtIn()).orderedNodeIDs
        return ["schemaVersion":1,"runID":.string("recorded-run"),"graphHash":.string(graphHash),"device":.string("cpu"),"stage":.string("validationCapture"),"selection":.string("firstValidationRows"),"maximumRows":8,"policy":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(policy)),"timingMethod":.string("hostWall"),"memoryMethod":.string("processResidentBytes"),"records":.array(order.map { id in
            .object(["nodeID":.string(id),"durationNanoseconds":250000,"timingUnavailableReason":.null,"residentBytesBefore":1048576,"residentBytesAfter":2097152,"memoryUnavailableReason":.null])
        })]
    }
    private func decode(_ object:[String:JSONValue], pin:JSONValue? = nil, run:String = "recorded-run", device:String = "cpu", requested:TrainingSampleCapturePolicy? = nil) throws -> WorkflowPerformanceCapture {
        let data = try JSONEncoder().encode(object)
        let metadata = pin ?? .object(["schemaVersion":1,"file":.string("performance.json"),"bytes":.integer(Int64(data.count)),"sha256":.string(WorkflowPhaseArtifacts.hash(data))])
        return try WorkflowPerformanceCapture.decode(data,metadata:metadata,runID:run,graphHash:graphHash,device:device,graph:graph,policy:requested ?? policy)
    }
    func testExactMeasurementIdentityAndUnavailableSamplesRemainDistinct() throws {
        var value = try artifact()
        let captured = try decode(value)
        XCTAssertEqual(captured.records.count, graph.nodes.count)
        XCTAssertEqual(captured.records.first?.durationNanoseconds,250000)
        XCTAssertEqual(captured.records.first?.residentBytesAfter,2097152)
        guard case .array(var records) = value["records"],case .object(var first) = records[0] else { return XCTFail() }
        first["durationNanoseconds"] = .null; first["timingUnavailableReason"] = .string("clockUnavailable")
        first["residentBytesBefore"] = .null; first["memoryUnavailableReason"] = .string("residentMemoryUnavailable")
        records[0] = .object(first); value["records"] = .array(records)
        let partial = try decode(value)
        XCTAssertNil(partial.records.first?.durationNanoseconds)
        XCTAssertNil(partial.records.first?.residentBytesBefore)
        XCTAssertEqual(partial.records.first?.residentBytesAfter,2097152)
    }
    func testTamperForeignIdentityWrongOrderAndInvalidRangesRefuse() throws {
        let value = try artifact()
        XCTAssertThrowsError(try decode(value,pin:.object(["schemaVersion":1,"file":.string("performance.json"),"bytes":1,"sha256":.string(graphHash)])))
        XCTAssertThrowsError(try decode(value,run:"foreign"))
        XCTAssertThrowsError(try decode(value,device:"mps"))
        XCTAssertThrowsError(try decode(value,requested:.factory))
        var changed = value; changed["graphHash"] = .string(String(repeating:"b",count:64)); XCTAssertThrowsError(try decode(changed))
        changed = value; changed["maximumRows"] = 9; XCTAssertThrowsError(try decode(changed))
        guard case .array(let rows) = value["records"],case .object(let first) = rows[0] else {return XCTFail()}
        let invalidDurations:[JSONValue] = [-1,.number(0.5),.bool(true),86400000000001,.null]
        for duration in invalidDurations {
            var row = first; row["durationNanoseconds"] = duration
            var all = rows; all[0] = .object(row); changed = value; changed["records"] = .array(all)
            XCTAssertThrowsError(try decode(changed))
        }
        changed = value; changed["records"] = .array(Array(rows.reversed())); XCTAssertThrowsError(try decode(changed))
        changed = value; changed["records"] = .array(rows+rows); XCTAssertThrowsError(try decode(changed))
        changed = value; changed["extra"] = .string("untrusted"); XCTAssertThrowsError(try decode(changed))
        changed = value; changed["runID"] = .string(String(repeating:"x",count:140000)); XCTAssertThrowsError(try decode(changed))
    }
}
