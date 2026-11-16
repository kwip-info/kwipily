import XCTest
import SwiftUI
import AppKit
import KwipMLCore
@testable import KwipMLApp

final class TensorHeatmapEvidenceTests:XCTestCase {
    private func summary()->JSONValue {
        try! JSONDecoder().decode(JSONValue.self,from:Data(#"{"schemaVersion":1,"sourceShape":[2,2],"rowAxis":0,"columnAxis":1,"aggregation":"finiteMean","planes":[{"index":0,"fixedIndices":[],"mean":[2.5],"minimum":[1],"maximum":[4],"finiteCount":[4],"nanCount":[0],"positiveInfinityCount":[0],"negativeInfinityCount":[0]}],"planeShape":[2,2],"gridShape":[1,1],"totalPlanes":1,"coverage":"complete"}"#.utf8))
    }
    @MainActor func testOverviewReportsAllRowsToItsParentScrollDocument() async throws {
        guard case .object(var raw)=summary(),case .array(let originals)?=raw["planes"],case .object(let original)=originals.first else {return XCTFail()}
        raw["sourceShape"] = .array([32,2,2]);raw["rowAxis"]=1;raw["columnAxis"]=2;raw["totalPlanes"]=32
        raw["planes"] = .array((0..<32).map {index in
            var plane=original;plane["index"] = .number(Double(index));plane["fixedIndices"] = .array([.number(Double(index))]);return .object(plane)
        })
        let checked=try TensorHeatmapSummary(value:.object(raw))
        let host=NSHostingView(rootView:TensorHeatmapView(summary:checked,accent:.pink,initialOverview:true).frame(width:400))
        host.layoutSubtreeIfNeeded();await Task.yield();host.layoutSubtreeIfNeeded()
        let size=host.fittingSize
        XCTAssertEqual(size.width,400,accuracy:1)
        XCTAssertGreaterThan(size.height,1200,"Every overview row must extend the parent scroll document, rather than overflow a nested 130-point viewport")
        XCTAssertLessThan(size.height,5000,"Captured planes remain bounded")
    }
    func testStrictSummaryAndIndependentCoverage()throws {
        let checked=try TensorHeatmapSummary(value:summary(),expectedShape:[2,2])
        XCTAssertEqual(checked.planes[0].mean,[2.5])
        XCTAssertThrowsError(try TensorHeatmapSummary(value:summary(),expectedShape:[4,1]))
        guard case .object(var raw)=summary(),case .array(var planes)?=raw["planes"],case .object(var plane)=planes[0] else {return XCTFail()}
        plane["finiteCount"] = .array([3]);planes[0] = .object(plane);raw["planes"] = .array(planes)
        XCTAssertThrowsError(try TensorHeatmapSummary(value:.object(raw)))
        plane["finiteCount"] = .array([4]);plane["surprise"]=true;planes[0] = .object(plane);raw["planes"] = .array(planes)
        XCTAssertThrowsError(try TensorHeatmapSummary(value:.object(raw)))
    }
    func testVirtualDisplayRowsDoNotBecomeScalarOrVectorSourceAxes()throws {
        let scalar=try JSONDecoder().decode(JSONValue.self,from:Data(#"{"schemaVersion":1,"sourceShape":[],"rowAxis":null,"columnAxis":null,"aggregation":"finiteMean","planes":[{"index":0,"fixedIndices":[],"mean":[3],"minimum":[3],"maximum":[3],"finiteCount":[1],"nanCount":[0],"positiveInfinityCount":[0],"negativeInfinityCount":[0]}],"planeShape":[1,1],"gridShape":[1,1],"totalPlanes":1,"coverage":"complete"}"#.utf8))
        XCTAssertNil(try TensorHeatmapSummary(value:scalar).rowAxis)
        guard case .object(var vector)=scalar else {return XCTFail()}
        vector["sourceShape"] = .array([1]);vector["columnAxis"]=0
        let checked=try TensorHeatmapSummary(value:.object(vector))
        XCTAssertNil(checked.rowAxis);XCTAssertEqual(checked.columnAxis,0)
        vector["rowAxis"]=0
        XCTAssertThrowsError(try TensorHeatmapSummary(value:.object(vector)))
    }
    func testPlaybackRejectsDuplicateEpochsUnknownFieldsAndUnrequestedFrames()throws {
        let frame:JSONValue = .object(["epoch":1,"sourceRows":2,"summary":summary()])
        var raw:[String:JSONValue] = ["schemaVersion":1,"source":.string("validationFirstBatch"),"frames":.array([frame])]
        XCTAssertEqual(try TensorHeatmapPlayback(value:.object(raw),maximumFrames:6).frames.first?.epoch,1)
        XCTAssertThrowsError(try TensorHeatmapPlayback(value:.object(raw),maximumFrames:0))
        raw["frames"] = .array([frame,frame])
        XCTAssertThrowsError(try TensorHeatmapPlayback(value:.object(raw)))
        raw["frames"] = .array([frame]);raw["unexpected"]=true
        XCTAssertThrowsError(try TensorHeatmapPlayback(value:.object(raw)))
    }
    func testCaptureSettingsPreserveHistoryAndOldWireFormats()throws {
        let encoder=JSONEncoder(),decoder=JSONDecoder()
        let old=TrainingSampleCapturePolicy.factory
        XCTAssertFalse(String(data:try encoder.encode(old),encoding:.utf8)!.contains("heatmap"))
        let requested=old.changing(tensorHeatmaps:true,heatmapFrameLimit:12)
        XCTAssertEqual(try decoder.decode(TrainingSampleCapturePolicy.self,from:encoder.encode(requested)),requested)
        XCTAssertEqual(requested.changing(performanceMeasurements:true).heatmapFrameLimit,12)
        let off=requested.applyingHistory(outputs:false,diagnostics:false,requiresPredictionTable:false)
        XCTAssertFalse(off.tensorHeatmaps);XCTAssertEqual(off.heatmapFrameLimit,0)
        XCTAssertEqual(requested.changing(tensorHeatmaps:false).heatmapFrameLimit,0)
        XCTAssertThrowsError(try encoder.encode(requested.changing(heatmapFrameLimit:25)))
    }
    func testSelectionReportChecksMeasuredLedgerAndIndependentStopping()throws {
        let policy=TrainingControlPolicy.factory
        var raw:[String:JSONValue]=["schemaVersion":1,"policy":try JSONDecoder().decode(JSONValue.self,from:JSONEncoder().encode(policy)),"completedEpoch":3,"selectedEpoch":3,"validationLoss":2,"stopReason":.string("epochLimit")]
        XCTAssertNotNil(try TrainingControlReport.verify(.object(raw),policy:policy,epochLimit:3,losses:[3,1,2]))
        raw["selectedEpoch"]=2
        XCTAssertThrowsError(try TrainingControlReport.verify(.object(raw),policy:policy,epochLimit:3,losses:[3,1,2]))
    }
}
