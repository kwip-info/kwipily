import XCTest
@testable import KwipMLCore

final class CanvasNodeIndexTests:XCTestCase {
    func testLookupPreservesFirstOccurrenceAndRefreshesMovedNodes() {
        let first=GraphNode(id:"a",type:"linear",x:20,y:30)
        let duplicate=GraphNode(id:"a",type:"output",x:900,y:900)
        let index=CanvasNodeIndex(nodes:[first,duplicate])
        XCTAssertEqual(index.node("a"),first)
        XCTAssertNil(index.node("missing"))
        var moved=first;moved.x=45
        XCTAssertEqual(CanvasNodeIndex(nodes:[moved]).node("a")?.x,45)
        XCTAssertEqual(index.node("a")?.x,20)
    }
    func testDenseWireLookupDiagnostic() throws {
        guard ProcessInfo.processInfo.environment["KWIPML_CANVAS_BENCHMARK"] == "1" else {throw XCTSkip("Enable render lookup timing explicitly")}
        let nodes=(0..<1024).map {GraphNode(id:String($0),type:"linear",x:Double($0),y:0)}
        let edges=(1..<1024).map {GraphEdge(source:String($0-1),target:String($0))}
        let start=ProcessInfo.processInfo.systemUptime
        var baseline=0.0
        for edge in edges {
            baseline += try XCTUnwrap(nodes.first {$0.id == edge.source}).x
            baseline += try XCTUnwrap(nodes.first {$0.id == edge.target}).x
        }
        let scanned=ProcessInfo.processInfo.systemUptime
        let index=CanvasNodeIndex(nodes:nodes)
        var indexed=0.0
        for edge in edges {
            indexed += try XCTUnwrap(index.node(edge.source)).x
            indexed += try XCTUnwrap(index.node(edge.target)).x
        }
        let finished=ProcessInfo.processInfo.systemUptime
        XCTAssertEqual(indexed,baseline)
        print(String(format:"CANVAS_WIRE_LOOKUP_BENCHMARK nodes=1024 edges=1023 scan_ms=%.3f build_and_lookup_ms=%.3f",(scanned-start)*1000,(finished-scanned)*1000))
    }
}
