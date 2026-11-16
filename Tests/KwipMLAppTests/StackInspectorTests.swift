import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class StackInspectorTests:XCTestCase {
    func testSignedAxisDraftAndOldOmittedDefault()throws {
        var draft=try TensorOperationInspectorDraft(type:"stack",values:[:])
        XCTAssertEqual(try draft.configuration(),["dim":-1])
        draft.texts["dim"]=" -2 ";XCTAssertEqual(try draft.configuration(),["dim":-2])
        for value in ["0","1.5","NaN","8","-9",""] {draft.texts["dim"]=value;XCTAssertThrowsError(try draft.configuration())}
    }
    func testInvalidUpstreamShapesKeepAttributedInspectorDiagnostic()async throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        var document=try PreparedWorkflowExample.component("stack").document(device:"cpu")
        let index=try XCTUnwrap(document.nodes.firstIndex {node in
            if case .modelPrimitive("reshape",_,_)=node.payload {return node.name=="Second sample directions"};return false
        })
        document.nodes[index].payload = .modelPrimitive(type:"reshape",configurationVersion:2,config:["shape":.array([3,2])])
        _=try library.saveSource(.workflow(document))
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:data)
        defer {owner.invalidate()}
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:data)
        let address=try XCTUnwrap(document.nodes.first {if case .modelPrimitive("stack",_,_)=$0.payload {return true};return false}?.address)
        for _ in 0..<200 {
            if canvas.modelConfigurationIssue(for:address) != nil {break}
            try await Task.sleep(for:.milliseconds(10))
        }
        XCTAssertNil(canvas.plannedInputLayout(for:address,port:"a"))
        XCTAssertTrue(try XCTUnwrap(canvas.modelConfigurationIssue(for:address)).contains("matching sample shapes"))
        XCTAssertNil(canvas.modelConfigurationIssue(for:document.nodes[index].address))
    }
}
