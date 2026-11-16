import XCTest
import KwipMLCore
@testable import KwipMLApp

func routingLessonGraph(_ task:TensorRoutingLearningTask)->GraphDocument {
    var nodes=task.sourceRoles.map {GraphNode(id:$0,type:"input",name:$0,config:["fixture":.string($0)])}
    nodes += [.init(id:"op",type:task.componentID,config:task.configuration),.init(id:"out",type:"output")]
    var edges:[GraphEdge]=[]
    func link(_ source:String,_ sourcePort:String,_ target:String,_ targetPort:String) {edges.append(.init(id:"e\(edges.count)",source:source,sourcePort:sourcePort,target:target,targetPort:targetPort))}
    switch task {
    case .gather:link("x","out","op","source");link("indices","out","op","indices");link("op","out","out","in")
    case .einsumCombine:link("left","out","op","a");link("right","out","op","b");link("op","out","out","in")
    case .einsumDiagonal:link("x","out","op","a");link("op","out","out","in")
    case .chunk:link("x","out","op","in");link("op","part4","out","in")
    case .splitSize,.splitSections:
        nodes += [.init(id:"first",type:"concat",config:["dim":2]),.init(id:"second",type:"concat",config:["dim":2])]
        let p=task == .splitSize ? [2,0,1]:[2,1,0]
        link("x","out","op","in");link("op","part\(p[0])","first","a");link("op","part\(p[1])","first","b");link("first","out","second","a");link("op","part\(p[2])","second","b");link("second","out","out","in")
    }
    return .init(name:"Named parts",nodes:nodes,edges:edges)
}
final class TensorRoutingCanvasTests:XCTestCase {
    @MainActor func testPaletteInputsBindDeclaredRolesAndNoExtraSources()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        for task in TensorRoutingLearningTask.allCases {
            let session=TensorLearningDraftSession(repository:.init(directory:root))
            XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:task.rawValue,mode:.guided))
            let canvas=TensorLearningCanvasAdapter(session:session)
            for _ in task.sourceRoles {canvas.addNode(type:"input",x:0,y:0)}
            XCTAssertEqual(canvas.graph.nodes.map {$0.config["fixture"]?.stringValue},task.sourceRoles.map(Optional.some))
            XCTAssertFalse(canvas.canAddType("input"))
            XCTAssertTrue(session.update {$0.graph=routingLessonGraph(task)})
            XCTAssertTrue(canvas.validation.isValid,canvas.validation.issues.map(\.message).joined(separator:"; "))
            XCTAssertEqual(canvas.validation.tensorContracts["out"]?.layout.sample.dimensions,task.outputSampleShape)
        }
    }
    @MainActor func testNamedPortsDescribeActualPartShapeAndSelectorsWithoutFalseOutAlias()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root));XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:TensorRoutingLearningTask.chunk.rawValue,mode:.guided));XCTAssertTrue(session.update {$0.graph=routingLessonGraph(.chunk)})
        let canvas=TensorLearningCanvasAdapter(session:session)
        XCTAssertTrue(canvas.canvasPortDescription(nodeID:"op",port:"part4").contains("[N, 2, 1]"))
        XCTAssertTrue(canvas.canvasPortDescription(nodeID:"out",port:"in").contains("[N, 2, 1]"))
        XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:TensorRoutingLearningTask.gather.rawValue,mode:.guided));XCTAssertTrue(session.update {$0.graph=routingLessonGraph(.gather)})
        let gather=TensorLearningCanvasAdapter(session:session)
        XCTAssertTrue(gather.canvasPortDescription(nodeID:"op",port:"indices").contains("int64"))
        XCTAssertTrue(gather.canvasPortDescription(nodeID:"op",port:"source").contains("float64"))
    }
    @MainActor func testPresetAndCustomNotationDraftApplyStayStrictAndPersistPendingText()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:root)}
        let session=TensorLearningDraftSession(repository:.init(directory:root));XCTAssertTrue(session.create(projectID:UUID().uuidString,operation:TensorRoutingLearningTask.einsumCombine.rawValue,mode:.guided));XCTAssertTrue(session.update {$0.graph=routingLessonGraph(.einsumCombine)})
        let canvas=TensorLearningCanvasAdapter(session:session)
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"pattern",text:"custom"));XCTAssertTrue(canvas.applyPending(nodeID:"op",field:"pattern"))
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"equation",text:"n a b, n b c -> n a c"));XCTAssertTrue(canvas.applyPending(nodeID:"op",field:"equation"))
        XCTAssertEqual(canvas.graph.nodes.first {$0.id == "op"}?.config["equation"]?.stringValue,"nab,nbc->nac")
        XCTAssertTrue(canvas.setPending(nodeID:"op",field:"equation",text:"nab,...->nac"));XCTAssertFalse(canvas.applyPending(nodeID:"op",field:"equation"))
        XCTAssertEqual(canvas.pendingText(nodeID:"op",field:"equation"),"nab,...->nac")
        let decoded=try TensorLearningDraft.decode(session.draft!.encoded());XCTAssertEqual(decoded.pendingFields,session.draft!.pendingFields)
        XCTAssertThrowsError(try decoded.submissionData())
    }

}
