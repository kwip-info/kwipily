import XCTest
@testable import KwipMLCore

final class CanvasNodeGeometryTests: XCTestCase {
    func testLegacyVerticalPreferenceCannotTransposeTheUniversalCanvas() {
        let point = CGPoint(x:75,y:180)
        let horizontal = CanvasCoordinateProjection(settings:.init(flowDirection:.leftToRight))
        let legacyVertical = CanvasCoordinateProjection(settings:.init(flowDirection:.topToBottom))
        XCTAssertEqual(legacyVertical,horizontal)
        XCTAssertEqual(legacyVertical.project(point),horizontal.project(point))
        XCTAssertEqual(legacyVertical.inverse(legacyVertical.project(point)).x,point.x,accuracy:0.000001)
    }
    func testLegacyDetailValuesAllUseTheSameTable() {
        let baseline = CanvasNodeGeometry.size(detail:.medium,direction:.leftToRight,inputCount:2,outputCount:3)
        for detail in CanvasNodeDetail.allCases {
            for direction in CanvasFlowDirection.allCases {
                XCTAssertEqual(CanvasNodeGeometry.size(detail:detail,direction:direction,inputCount:2,outputCount:3),baseline)
            }
        }
    }
    func testEveryPortHasItsOwnRowAndWireMeetsTheEdgeIcon() {
        for direction in CanvasFlowDirection.allCases {
            let size = CanvasNodeGeometry.size(detail:.small,direction:direction,inputCount:12,outputCount:2)
            let inputs = (0..<12).map {CanvasNodeGeometry.portAnchor(size:size,direction:direction,output:false,index:$0,count:12)}
            let outputs = (0..<2).map {CanvasNodeGeometry.portAnchor(size:size,direction:direction,output:true,index:$0,count:2)}
            XCTAssertGreaterThan(inputs[0].y,40, "Ports must clear the header")
            XCTAssertGreaterThan(outputs[0].y,inputs.last!.y, "Output rows follow input rows")
            for anchor in inputs + outputs {XCTAssertGreaterThan(anchor.y,0);XCTAssertLessThan(anchor.y,size.height)}
            XCTAssertEqual(inputs[0].x,0)
            XCTAssertEqual(outputs[0].x,size.width)
            for pair in zip(inputs + outputs,(inputs + outputs).dropFirst()) {XCTAssertGreaterThanOrEqual(pair.1.y-pair.0.y,28)}
            XCTAssertTrue(inputs.allSatisfy {$0.x == inputs[0].x})
            XCTAssertTrue(outputs.allSatisfy {$0.x > size.width/2})
        }
    }
    func testSinkSourceAndPortlessFramesHaveNoFabricatedPorts() {
        let sink = CanvasNodeGeometry.size(detail:.large,direction:.leftToRight,inputCount:12,outputCount:0)
        let source = CanvasNodeGeometry.size(detail:.small,direction:.leftToRight,inputCount:0,outputCount:12)
        XCTAssertEqual(sink,source)
        XCTAssertGreaterThan(sink.height,CanvasNodeGeometry.size(detail:.medium,direction:.leftToRight,inputCount:0,outputCount:0).height)
    }
}
