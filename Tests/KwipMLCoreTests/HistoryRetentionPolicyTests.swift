import XCTest
@testable import KwipMLCore

final class HistoryRetentionPolicyTests:XCTestCase {
    func testOffStopsOptionalCollectionButConnectedDataAlwaysRemainsRequired()throws {
        let preferences=TrainingSampleCapturePolicy(activationValues:true,predictionRows:false,performanceMeasurements:true)
        let off=preferences.applyingHistory(outputs:false,diagnostics:false,requiresPredictionTable:false)
        XCTAssertEqual(off.schemaVersion,3);XCTAssertFalse(off.savesComponentOutputs);XCTAssertFalse(off.savesDiagnosticDetails)
        XCTAssertFalse(off.activationValues);XCTAssertFalse(off.predictionRows);XCTAssertFalse(off.performanceMeasurements)
        XCTAssertEqual(try JSONDecoder().decode(TrainingSampleCapturePolicy.self,from:JSONEncoder().encode(off)),off)
        let required=preferences.applyingHistory(outputs:false,diagnostics:false,requiresPredictionTable:true)
        XCTAssertTrue(required.predictionRows);XCTAssertFalse(required.activationValues)
        for value in [TrainingSampleCapturePolicy(activationValues:true,predictionRows:false,retainComponentOutputs:false),TrainingSampleCapturePolicy(activationValues:false,predictionRows:false,performanceMeasurements:true,retainDiagnosticDetails:false)] {
            XCTAssertThrowsError(try JSONEncoder().encode(value))
        }
        let raw=try JSONEncoder().encode(off)
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:raw) as? [String:Any]);object["retainComponentOutputs"]=nil
        XCTAssertThrowsError(try JSONDecoder().decode(TrainingSampleCapturePolicy.self,from:JSONSerialization.data(withJSONObject:object)))
    }
}
