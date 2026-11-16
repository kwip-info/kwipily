import XCTest
import KwipMLCore
@testable import KwipMLApp

final class BagResultPresentationTests:XCTestCase {
    private func descriptor(_ weighted:Bool=true)->JSONValue {.object(["kind":.string("bagPacket"),"configuration":.object(["layout":.string("packed"),"per_sample_weights":.bool(weighted)])])}
    private func preview(_ rows:[BagPredictionRow],status:ObjectiveCaptureStatus = .captured,count:Int=8)->BagPredictionPreview {
        .init(runID:"verified",evaluationCount:count,inputDescriptor:descriptor(),captureStatus:status,rows:rows)
    }
    private func tensor(_ values:[Int],dtype:ContractDType = .float32)throws->ObjectiveResultTensor {
        try .init(json:.array(values.map { .integer(Int64($0)) }),shape:[values.count],dtype:dtype)
    }
    private func scalar(_ value:Int)throws->ObjectiveResultTensor {try .init(json:.integer(Int64(value)),shape:[],dtype:.float32)}
    private func binary(_ ids:[Int64],label:Int?=1,predicted:Int?=0)->BagPredictionRow {
        .init(input:.init(indices:ids,perSampleWeights:ids.map {_ in 0.5}),label:label,predicted:predicted,output:nil,targets:[:],objectiveOutput:nil,decoded:nil)
    }
    private func objective()throws->BagPredictionRow {
        try .init(input:.init(indices:[4,2],perSampleWeights:[1,0.5]),label:nil,predicted:nil,output:tensor([2,3]),targets:["target":tensor([1,3]),"importance":scalar(2)],objectiveOutput:scalar(1),decoded:tensor([1],dtype:.int64))
    }
    private func evaluation(raw:Bool=true)->ObjectiveEvaluation {
        .init(componentID:"loss_mse",observations:8,activeCount:16,ignoredCount:0,maskedCount:0,effectiveScalar:0.25,aggregationNumerator:4,aggregationDenominator:16,aggregationConvention:"global",rawShape:[8,2],accuracy:nil,rawPreview:raw ? .init(shape:[8,2],dtype:.float32,values:[1,0],captureStatus:.captured):nil)
    }
    func testWholeBagTableKeepsIDsWeightsAndExplicitEmptyWithoutDenseCoordinates() throws {
        let sections=BagResultProjection.sections(preview:preview([binary([4,2]),binary([])]))
        let inputs=try XCTUnwrap(sections.first {$0.id=="inputs"})
        XCTAssertEqual(inputs.columns,["Observation","Items","Item IDs","Floating weights"])
        XCTAssertEqual(inputs.rows[0],["1","2","4, 2","0.5, 0.5"])
        XCTAssertEqual(inputs.rows[1],["2","0","Empty bag","Empty bag"])
        let classes=try XCTUnwrap(sections.first {$0.id=="binary"})
        XCTAssertEqual(classes.columns,["Observation","Expected class","Predicted class","Match"])
        XCTAssertEqual(classes.rows[0],["1","1","0","No"])
        XCTAssertFalse(sections.flatMap(\.columns).contains { $0.contains("input[") })
        XCTAssertFalse(sections.contains {$0.id=="summary"})
    }
    func testArbitraryObjectiveRolesShapesAndMeasuredScoresHaveNoInventedClassLabels() throws {
        let sections=BagResultProjection.sections(preview:preview([try objective()]),evaluation:evaluation())
        XCTAssertTrue(try XCTUnwrap(sections.first {$0.id=="produced"}).caption.contains("shape [2]"))
        XCTAssertTrue(try XCTUnwrap(sections.first {$0.id=="target.importance"}).caption.contains("scalar []"))
        XCTAssertEqual(try XCTUnwrap(sections.first {$0.id=="target.target"}).rows,[["1","1, 3"]])
        XCTAssertEqual(try XCTUnwrap(sections.first {$0.id=="summary"}).rows.first,["Result score","0.25"])
        XCTAssertTrue(sections.contains {$0.id=="individualScores"});XCTAssertTrue(sections.contains {$0.id=="classSelection"});XCTAssertTrue(sections.contains {$0.id=="rawPreview"})
        XCTAssertFalse(sections.flatMap(\.columns).contains("Match"));XCTAssertFalse(sections.contains {$0.id=="binary"})
        let summary=try XCTUnwrap(sections.first {$0.id=="summary"})
        XCTAssertFalse(summary.rows.contains {$0.first=="Accuracy"})
    }
    func testOmittedRowsDoNotHideMeasuredEvaluationOrInventInputs() throws {
        let sections=BagResultProjection.sections(preview:preview([],status:.omittedByPolicy),evaluation:evaluation())
        XCTAssertEqual(sections.first {$0.id=="capture"}?.rows,[["Omitted by policy"]])
        XCTAssertTrue(sections.contains {$0.id=="summary"});XCTAssertTrue(sections.contains {$0.id=="rawPreview"})
        XCTAssertFalse(sections.contains {$0.id=="inputs" || $0.id=="produced" || $0.id=="binary"})
    }
    func testVisibleBoundsAndIncompleteClassificationNeverManufactureMatch() throws {
        let rows=Array(repeating:binary(Array(0..<20).map(Int64.init),label:1,predicted:nil),count:40)
        let sections=BagResultProjection.sections(preview:preview(rows,count:80))
        let inputs=try XCTUnwrap(sections.first {$0.id=="inputs"})
        XCTAssertEqual(inputs.rows.count,32);XCTAssertTrue(inputs.caption.contains("32 of 40"));XCTAssertTrue(inputs.rows[0][2].contains("first 16 of 20"))
        XCTAssertFalse(sections.contains {$0.id=="binary"})
        let unweighted=BagPredictionPreview(runID:"u",evaluationCount:1,inputDescriptor:descriptor(false),captureStatus:.captured,rows:[.init(input:.init(indices:[1],perSampleWeights:nil),label:0,predicted:0,output:nil,targets:[:],objectiveOutput:nil,decoded:nil)])
        XCTAssertEqual(BagResultProjection.sections(preview:unweighted).first {$0.id=="inputs"}?.columns,["Observation","Items","Item IDs"])
    }
}
