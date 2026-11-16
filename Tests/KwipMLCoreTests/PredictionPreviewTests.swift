import XCTest
@testable import KwipMLCore

final class PredictionPreviewTests: XCTestCase {
    func testIntegerCoordinatesPreserveValuesAboveDoublePrecision() throws {
        let preview = try PredictionPreview.decode(Data(#"{"predictionInput":{"dtype":"int64","sampleShape":[1,2]},"predictionRows":[{"input":[[9007199254740993,2]],"label":0,"predicted":1}],"evaluationCount":12}"#.utf8))
        XCTAssertEqual(preview.rows[0].inputs, ["9007199254740993", "2"])
        XCTAssertEqual(preview.coordinates, ["[0,0]", "[0,1]"])
        XCTAssertEqual(preview.evaluationCount, 12)
        XCTAssertEqual(preview.rows[0].label, 0)
        XCTAssertEqual(preview.rows[0].predicted, 1)
    }
    func testFloatRowsRemainNumericAndObservationCountIsBounded() throws {
        let data = Data(#"{"predictionInput":{"dtype":"float32","sampleShape":[2]},"predictionRows":[{"input":[1.25,-2],"label":1,"predicted":1}],"evaluationCount":8}"#.utf8)
        let preview = try PredictionPreview.decode(data)
        XCTAssertEqual(preview.rows[0].inputs, ["1.25", "-2"])
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["evaluationCount"] = 0
        XCTAssertThrowsError(try PredictionPreview.decode(JSONSerialization.data(withJSONObject: object)))
        object["evaluationCount"] = 500
        let row = try XCTUnwrap((object["predictionRows"] as? [[String: Any]])?.first)
        object["predictionRows"] = Array(repeating: row, count: 201)
        XCTAssertThrowsError(try PredictionPreview.decode(JSONSerialization.data(withJSONObject: object)))
    }
    func testRejectsRaggedWrongTypeLabelsAndOversizedShape() {
        let inputs = [
            #"{"dtype":"int64","sampleShape":[2]},"predictionRows":[{"input":[1.5,2],"label":0,"predicted":1}]"#,
            #"{"dtype":"int64","sampleShape":[1,2]},"predictionRows":[{"input":[1,2],"label":0,"predicted":1}]"#,
            #"{"dtype":"float32","sampleShape":[2]},"predictionRows":[{"input":[1,2],"label":2,"predicted":1}]"#,
            #"{"dtype":"int64","sampleShape":[9223372036854775807,2]},"predictionRows":[]"#
        ]
        for input in inputs {
            XCTAssertThrowsError(try PredictionPreview.decode(Data(("{\"predictionInput\":" + input + ",\"evaluationCount\":12}").utf8)))
        }
    }
    func testExplicitOmissionRequiresVersionedStatusAndEmptyRows() throws {
        var object: [String: Any] = ["schemaVersion":2,"captureStatus":"omittedByPolicy",
            "predictionInput":["dtype":"float32","sampleShape":[2]],"predictionRows":[],"evaluationCount":12]
        func decode() throws -> PredictionPreview { try .decode(JSONSerialization.data(withJSONObject: object)) }
        XCTAssertTrue(try decode().rows.isEmpty)
        object["predictionRows"] = [["input":[1,2],"label":0,"predicted":1]]
        XCTAssertThrowsError(try decode())
        object["predictionRows"] = []
        for status: Any in ["captured","unknown",NSNull()] {
            object["captureStatus"] = status
            XCTAssertThrowsError(try decode())
        }
        object.removeValue(forKey:"captureStatus")
        XCTAssertThrowsError(try decode())
        object["captureStatus"] = "omittedByPolicy"
        object["schemaVersion"] = 1
        XCTAssertThrowsError(try decode())
    }

    func testLiveCapturePolicyMakesOmissionExplicit() throws {
        var object: [String: Any] = ["sampleCapture":["schemaVersion":1,"activationValues":false,"predictionRows":false],
            "predictionInput":["dtype":"float32","sampleShape":[2]],"predictionRows":[],"evaluationCount":12]
        func decode() throws -> PredictionPreview { try .decode(JSONSerialization.data(withJSONObject: object)) }
        XCTAssertEqual(try decode().captureStatus, .omittedByPolicy)
        object["sampleCapture"] = ["schemaVersion":1,"activationValues":false,"predictionRows":0]
        XCTAssertThrowsError(try decode())
        object["sampleCapture"] = NSNull()
        XCTAssertThrowsError(try decode())
    }

}
