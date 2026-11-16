import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RecurrentExposedStateEditingTests:XCTestCase {
    func testCanonicalChildOwnerRefusesRemovingExposedCellOrStartingStatePorts()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        var config=try RecurrentSequenceConfiguration.defaults(for:.lstm,registry:registry);config["state_mode"] = .string("supplied")
        let definition=CompositeDefinition(reference:.init("recurrent-owner"),title:"Memory",body:.init(nodes:[.init(id:"memory",type:"lstm",config:config)],configurationVersion:2),inputs:[.init("in",targets:[.init("memory","in")]),.init("hidden",targets:[.init("memory","hidden")]),.init("cell",targets:[.init("memory","cell")])],outputs:[.init("out",source:.init("memory","out")),.init("cell",source:.init("memory","cell"))])
        let source=CompositeProjectDocument(name:"Memory",root:.init(instances:[.init(id:"group",definition:definition.reference)]),definitions:[definition])
        let library=ProjectLibrary(directory:root),id=UUID().uuidString
        _=try library.saveDocument(.blockModel(.init(projectID:id,source:source)))
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        XCTAssertTrue(owner.beginDraft(path:["group"],lease:owner.lease()))
        let before=owner.document
        XCTAssertNil(owner.applyDraftCommand(.setRecurrent(nodeID:"memory",method:.gru),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
        config["state_mode"] = .string("zero")
        XCTAssertNil(owner.applyDraftCommand(.setRecurrent(nodeID:"memory",method:.lstm,configuration:config),lease:owner.lease(path:["group"])))
        XCTAssertEqual(owner.document,before)
    }
}
