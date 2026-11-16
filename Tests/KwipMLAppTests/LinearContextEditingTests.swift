import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class LinearContextEditingTests:XCTestCase {
    func testSelectedWorkflowModeChangeIsUndoableAndRejectsStaleOrUnselectedContext()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let address=WorkflowNodeAddress(domain:.model,localID:"projection")
        var config=try LinearConfiguration.defaults(for:.explicit,registry:registry);config["out_features"]=6
        let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,name:"Scores",payload:.modelPrimitive(type:"linear",configurationVersion:2,config:config))])
        _=try library.saveSource(.workflow(document))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        defer {owner.invalidate()}
        XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
        let context=try WorkflowModelNodeContext(owner:owner,address:address)
        var inferred=try LinearConfiguration.defaults(for:.inferred,registry:registry);inferred["out_features"]=6
        XCTAssertTrue(context.setLinear(.inferred,configuration:inferred,expectedType:"linear",expected:config),context.statusMessage)
        XCTAssertEqual(try LinearConfiguration.configuration(of:owner.document.nodes[0],registry:registry).variant,.inferred)
        XCTAssertFalse(context.setLinear(.explicit,configuration:config,expectedType:"linear",expected:config))
        XCTAssertTrue(owner.undo(lease:owner.lease()))
        XCTAssertEqual(owner.document.nodes,document.nodes)
        XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
        XCTAssertFalse(context.setLinear(.inferred,configuration:inferred,expectedType:"linear",expected:config))
        XCTAssertEqual(owner.document.nodes,document.nodes)
    }
}
