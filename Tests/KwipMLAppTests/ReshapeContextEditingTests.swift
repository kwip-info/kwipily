import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ReshapeContextEditingTests:XCTestCase {
    func testSelectedWorkflowModeChangeIsUndoableAndRejectsStaleOrUnselectedContext()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"projection")
        var config=try ReshapeAuthoringConfiguration.defaults(for:.dimensions,registry:registry);config["shape"] = .array([2,3])
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Scores",payload:.modelPrimitive(type:"reshape",configurationVersion:2,config:config))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        defer {owner.invalidate()}
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        var inferred=try ReshapeAuthoringConfiguration.defaults(for:.flatten,registry:registry);inferred["start_dim"]=1
        XCTAssertTrue(context.setReshape(.flatten,configuration:inferred,expectedType:"reshape",expected:config),context.statusMessage)
        guard case .modelPrimitive(let actual,_,_) = owner.document.nodes[0].payload else {return XCTFail("Expected model primitive")}
        XCTAssertEqual(actual,"flatten")
        XCTAssertFalse(context.setReshape(.dimensions,configuration:config,expectedType:"reshape",expected:config))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,document.nodes)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setReshape(.flatten,configuration:inferred,expectedType:"reshape",expected:config))
        XCTAssertEqual(owner.document.nodes,document.nodes)
    }
}
