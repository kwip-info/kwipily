import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowModelOperationProgressTests: XCTestCase {
    private let run = "run-identity"
    private var graph: GraphDocument { .init(nodes:[.init(id:"a"),.init(id:"b")],edges:[.init(id:"ab",source:"a",target:"b")]) }
    private var origins: [String:CompositeNodeOrigin] { ["a":.init(instancePath:["block1"],localNodeID:"layer",definition:nil),"b":.init(instancePath:["block2"],localNodeID:"layer",definition:nil)] }
    private func event(_ name:String,_ sequence:Int,node:String?=nil)->[String:Any] {
        var value:[String:Any] = ["event":name,"schemaVersion":1,"runID":run,"stage":"validationCapture","sequence":sequence]
        if let node {value["nodeID"]=node}
        if name == "modelCaptureStarted" {value["operationCount"]=2}
        return value
    }
    func testRealFrameProjectsRepeatedBlockInstancesAndClears() throws {
        var feed = try WorkflowModelOperationProgress(graph:graph,origins:origins,workerRunID:run)
        XCTAssertTrue(feed.consume(event("modelCaptureStarted",1)))
        XCTAssertTrue(feed.consume(event("modelNodeStarted",2,node:"a")))
        XCTAssertEqual(feed.activeOrigin?.instancePath,["block1"])
        XCTAssertEqual(feed.activeNode(in:[]),"block1")
        XCTAssertEqual(feed.activeNode(in:["block1"]),"layer")
        XCTAssertNil(feed.activeNode(in:["block2"]))
        XCTAssertTrue(feed.consume(event("modelNodeCompleted",3,node:"a")))
        XCTAssertTrue(feed.consume(event("modelNodeStarted",4,node:"b")))
        XCTAssertEqual(feed.activeIncomingEdgeIDs,["ab"])
        XCTAssertEqual(feed.activeNode(in:[]),"block2")
        XCTAssertTrue(feed.consume(event("modelNodeCompleted",5,node:"b")))
        XCTAssertTrue(feed.consume(event("modelCaptureCompleted",6)))
        XCTAssertNil(feed.activeOrigin)
        XCTAssertFalse(feed.consume(event("modelNodeStarted",7,node:"a")))
    }
    func testUnknownStaleDuplicateAndDependencyEventsNeverMoveActivity() throws {
        var feed = try WorkflowModelOperationProgress(graph:graph,origins:origins,workerRunID:run)
        XCTAssertFalse(feed.consume(event("modelNodeStarted",1,node:"a")))
        var booleanVersion=event("modelCaptureStarted",1);booleanVersion["schemaVersion"]=true
        XCTAssertFalse(feed.consume(booleanVersion))
        XCTAssertTrue(feed.consume(event("modelCaptureStarted",1)))
        XCTAssertFalse(feed.consume(event("modelNodeStarted",2,node:"unknown")))
        XCTAssertFalse(feed.consume(event("modelNodeStarted",2,node:"b")))
        var stale=event("modelNodeStarted",2,node:"a");stale["runID"]="other"
        XCTAssertFalse(feed.consume(stale))
        XCTAssertTrue(feed.consume(event("modelNodeStarted",2,node:"a")))
        XCTAssertFalse(feed.consume(event("modelNodeCompleted",2,node:"a")))
        XCTAssertFalse(feed.consume(event("modelNodeStarted",3,node:"b")))
        feed.clear()
        XCTAssertNil(feed.activeOrigin)
        XCTAssertFalse(feed.consume(event("modelNodeCompleted",3,node:"a")))
    }
    func testOriginCoverageMustMatchCapturedGraph() {
        XCTAssertThrowsError(try WorkflowModelOperationProgress(graph:graph,origins:["a":origins["a"]!],workerRunID:run))
    }
}
