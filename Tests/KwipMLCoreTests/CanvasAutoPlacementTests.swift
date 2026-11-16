import XCTest
import CoreGraphics
@testable import KwipMLCore

final class CanvasAutoPlacementTests:XCTestCase {
    func testReplacementAvoidsOccupiedPositionAcrossPresentations() throws {
        let registry=try ComponentRegistry.builtIn()
        for direction in [CanvasFlowDirection.leftToRight,.topToBottom] {
            for detail in [CanvasNodeDetail.small,.medium,.large] {
                let settings=CanvasPresentationSettings(flowDirection:direction,nodeDetail:detail)
                let graph=GraphDocument(nodes:[GraphNode(id:"input",type:"input",x:80,y:140),GraphNode(id:"output",type:"output",x:440,y:140)])
                let point=CanvasAutoPlacement.origin(type:"transpose",graph:graph,registry:registry,catalog:registry,presentation:settings,start:CGPoint(x:80,y:140))
                XCTAssertEqual(point.x,260,accuracy:0.001)
                XCTAssertEqual(point.y,140,accuracy:0.001)
                XCTAssertEqual(graph.nodes[1].x,440)
            }
        }
    }
    func testRepeatedAddsContinueOntoFreeRows() throws {
        let registry=try ComponentRegistry.builtIn()
        var graph=GraphDocument()
        for index in 0..<12 {
            let point=CanvasAutoPlacement.origin(type:"linear",graph:graph,registry:registry,catalog:registry,presentation:CanvasPresentationSettings(),start:CGPoint(x:60,y:420))
            XCTAssertEqual(point.x,60+Double(index%4)*180,accuracy:0.001)
            XCTAssertEqual(point.y,420+Double(index/4)*120,accuracy:0.001)
            graph.nodes.append(GraphNode(id:String(index),type:"linear",x:point.x,y:point.y))
        }
    }
    func testDenseInsertionDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["KWIPML_CANVAS_BENCHMARK"] == "1" else {throw XCTSkip("Enable dense insertion timing explicitly")}
        let registry=try ComponentRegistry.builtIn()
        for count in [256,1024] {
            let nodes=(0..<count).map {index in GraphNode(id:String(index),type:"linear",x:60+Double(index%4)*180,y:420+Double(index/4)*120)}
            let graph=GraphDocument(nodes:nodes)
            let start=ProcessInfo.processInfo.systemUptime
            let point=CanvasAutoPlacement.origin(type:"linear",graph:graph,registry:registry,catalog:registry,presentation:CanvasPresentationSettings(),start:CGPoint(x:60,y:420))
            let elapsed=(ProcessInfo.processInfo.systemUptime-start)*1000
            XCTAssertEqual(point.x,60,accuracy:0.001)
            XCTAssertEqual(point.y,420+Double(count/4)*120,accuracy:0.001)
            print(String(format:"CANVAS_INSERTION_BENCHMARK nodes=%d elapsed_ms=%.3f",count,elapsed))
        }
    }
    func testIndexedPlacementMatchesRectangleSearchForIrregularLayouts() throws {
        let registry=try ComponentRegistry.builtIn()
        for direction in [CanvasFlowDirection.leftToRight,.topToBottom] {
            for detail in [CanvasNodeDetail.small,.medium,.large] {
                let settings=CanvasPresentationSettings(flowDirection:direction,nodeDetail:detail)
                let projection=CanvasCoordinateProjection(settings:settings,maximumInputCount:registry.components.map {$0.inputs.count}.max() ?? 0,maximumOutputCount:registry.components.map {$0.outputs.count}.max() ?? 0)
                let size=CanvasNodeGeometry.size(detail:detail,direction:direction,inputCount:1,outputCount:1)
                for offset in [0.0,12.0,29.0,-20.0,1000000.0] {
                    var nodes:[GraphNode]=[]
                    for i in 0..<80 {
                        let x=60.0+Double(i%4)*180.0+offset
                        let y=420.0+Double(i/4)*120.0+Double(i%3)*7.0
                        nodes.append(GraphNode(id:String(i),type:"linear",x:x,y:y))
                    }
                    let graph=GraphDocument(nodes:nodes)
                    let occupied=nodes.map {CGRect(origin:projection.project(CGPoint(x:$0.x,y:$0.y)),size:size).insetBy(dx:-12,dy:-12)}
                    var expected:CGPoint?
                    for i in 0..<200 {
                        let point=CGPoint(x:60+Double(i%4)*180,y:420+Double(i/4)*120)
                        let rect=CGRect(origin:projection.project(point),size:size)
                        if !occupied.contains(where:{$0.intersects(rect)}) {expected=point;break}
                    }
                    XCTAssertEqual(CanvasAutoPlacement.origin(type:"linear",graph:graph,registry:registry,catalog:registry,presentation:settings,start:CGPoint(x:60,y:420)),try XCTUnwrap(expected))
                }
            }
        }
    }
    func testWideCatalogAndExtremePositionsMatchExactSearch() throws {
        let normal=ComponentDefinition(id:"normal",inputs:["in"],outputs:["out"])
        let wide=ComponentDefinition(id:"wide",inputs:(0..<128).map {"in"+String($0)},outputs:["out"])
        // A creation catalog may offer a larger node than any existing definition.
        let registry=ComponentRegistry(components:[normal])
        let catalog=ComponentRegistry(components:[normal,wide])
        for direction in [CanvasFlowDirection.leftToRight,.topToBottom] {
            for detail in [CanvasNodeDetail.small,.medium,.large] {
                let settings=CanvasPresentationSettings(flowDirection:direction,nodeDetail:detail)
                let projection=CanvasCoordinateProjection(settings:settings,maximumInputCount:1,maximumOutputCount:1)
                let oldSize=CanvasNodeGeometry.size(detail:detail,direction:direction,inputCount:1,outputCount:1)
                let newSize=CanvasNodeGeometry.size(detail:detail,direction:direction,inputCount:128,outputCount:1)
                let nodes=[GraphNode(id:"a",type:"normal",x:65,y:420),GraphNode(id:"b",type:"normal",x:238,y:547),GraphNode(id:"far",type:"normal",x:1e100,y:1e100),GraphNode(id:"negative",type:"normal",x:-1e100,y:-1e100)]
                let occupied=nodes.map {CGRect(origin:projection.project(CGPoint(x:$0.x,y:$0.y)),size:oldSize).insetBy(dx:-12,dy:-12)}
                var expected:CGPoint?
                for index in 0..<516 {
                    let point=CGPoint(x:60+Double(index%4)*180,y:420+Double(index/4)*120)
                    let rect=CGRect(origin:projection.project(point),size:newSize)
                    if !occupied.contains(where:{$0.intersects(rect)}) {expected=point;break}
                }
                let actual=CanvasAutoPlacement.origin(type:"wide",graph:GraphDocument(nodes:nodes),registry:registry,catalog:catalog,presentation:settings,start:CGPoint(x:60,y:420))
                XCTAssertEqual(actual,try XCTUnwrap(expected))
                XCTAssertFalse(occupied.contains {$0.intersects(CGRect(origin:projection.project(actual),size:newSize))})
            }
        }
    }
}
