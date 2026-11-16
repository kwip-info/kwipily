import XCTest
@testable import KwipMLCore
final class WorkspaceArrangementTests:XCTestCase {
    func testOnePercentCanvasStatesAreSupportedButLowerValuesRejected() throws {
        var state=WorkspaceViewState(modelCanvas:CanvasViewState(zoom:0.01),dataCanvas:CanvasViewState(zoom:0.1))
        state.compositeCanvasStates=["root":CanvasViewState(zoom:0.01)]
        XCTAssertNoThrow(try state.validate())
        state.modelCanvas.zoom=0.009
        XCTAssertThrowsError(try state.validate())
        state.modelCanvas.zoom=1.5001
        XCTAssertThrowsError(try state.validate())
    }
    func testOnePercentNamedArrangementAndProjectViewportRoundTrip() throws {
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let library=WorkspaceArrangementLibrary(directory:directory),projectID=UUID().uuidString
        let state=WorkspaceViewState(modelCanvas:CanvasViewState(zoom:0.01),dataCanvas:CanvasViewState(zoom:0.05))
        let named=try library.save(name:"Tall flow overview",state:state)
        try library.saveProjectState(state,projectID:projectID)
        XCTAssertEqual(try library.list().first(where:{$0.id==named.id})?.state,state)
        XCTAssertEqual(try library.projectState(projectID:projectID),state)
    }
    func testNamedCRUDAndProjectStateRemainIndependent()throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let library=WorkspaceArrangementLibrary(directory:dir),project=UUID().uuidString
        var state=WorkspaceViewState();state.showLibrary=false;state.activeSection="Data";state.modelCanvas=CanvasViewState(zoom:1.2,scrollX:50,scrollY:80,selectedNodeIDs:["a"])
        let named=try library.save(name:"Inspect",state:state)
        try library.saveProjectState(state,projectID:project)
        let renamed=try library.rename(id:named.id,name:"My space")
        XCTAssertEqual(renamed.state,state);XCTAssertEqual(try library.list().map(\.name),["My space"])
        try library.delete(id:named.id)
        XCTAssertTrue(try library.list().isEmpty);XCTAssertEqual(try library.projectState(projectID:project),state)
    }
    func testInvalidStateAndTraversalAreRejectedWithoutOverwritingGoodState()throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let library=WorkspaceArrangementLibrary(directory:dir),id=UUID().uuidString
        let good=WorkspaceViewState();try library.saveProjectState(good,projectID:id)
        var bad=good;bad.modelCanvas.zoom = .nan
        XCTAssertThrowsError(try library.saveProjectState(bad,projectID:id));XCTAssertEqual(try library.projectState(projectID:id),good)
        XCTAssertThrowsError(try library.projectState(projectID:"../outside"))
        bad=good;bad.activeSection="Unknown";XCTAssertThrowsError(try library.save(name:"Invalid",state:bad))
        bad=good;bad.modelCanvas.selectedNodeIDs=Set((0..<257).map(String.init));XCTAssertThrowsError(try library.save(name:"Large",state:bad))
    }
    func testCorruptSchemaAndSymlinkRejected()throws {
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let library=WorkspaceArrangementLibrary(directory:dir),record=try library.save(name:"Good",state:WorkspaceViewState())
        let path=dir.appendingPathComponent("named/\(record.id).json")
        var json=try JSONSerialization.jsonObject(with:Data(contentsOf:path)) as! [String:Any];json["schemaVersion"]=99
        try JSONSerialization.data(withJSONObject:json).write(to:path)
        XCTAssertThrowsError(try library.list())
        try FileManager.default.removeItem(at:path)
        let target=dir.appendingPathComponent("external.json");try Data("{}".utf8).write(to:target)
        try FileManager.default.createSymbolicLink(at:path,withDestinationURL:target)
        XCTAssertThrowsError(try library.list())
    }
}
