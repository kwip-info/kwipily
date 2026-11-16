import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowMixtureContainerAppTests:XCTestCase {
    private let root=WorkflowNodeAddress(domain:.model,localID:"mixture")
    private func fixture()throws->(URL,WorkflowAuthoringStore,WorkflowCanvasAdapter){
        let directory=try canonicalBackendTemporaryDirectory(),library=ProjectLibrary(directory:directory.appendingPathComponent("projects"))
        let source=try MixtureExpertsPractice.document();_=try library.saveSource(.workflow(source))
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let owner=try WorkflowAuthoringStore(library:library,projectID:source.projectID,modelRegistry:.builtIn(),dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry)
        addTeardownBlock{try? FileManager.default.removeItem(at:directory);UserDefaults.standard.removeObject(forKey:"workflow.inlineContainers."+source.projectID)}
        return(directory,owner,canvas)
    }
    func testOneContainerSameCanvasInspectorUndoAndReopen()throws {
        let(_,owner,canvas)=try fixture();defer{canvas.invalidate();owner.invalidate()}
        XCTAssertEqual(canvas.inlineContainers.count,1)
        let container=try XCTUnwrap(canvas.inlineContainers.first);XCTAssertEqual(container.childNodeIDs.count,12)
        XCTAssertEqual(container.boundaryLinks.filter{$0.routed && $0.direction == .input}.count,4)
        XCTAssertEqual(container.boundaryLinks.filter{$0.routed && $0.direction == .output}.count,4)
        for link in container.boundaryLinks {
            let node=try XCTUnwrap(canvas.graph.nodes.first{$0.id==link.childNodeID})
            let component=try XCTUnwrap(canvas.registry.component(type:node.type))
            XCTAssertFalse(["input","output"].contains(node.type))
            XCTAssertTrue(CanvasNodeFeedback.connectionIssues(nodeID:node.id,document:owner.document,projection:canvas.projection).isEmpty,"Container-bound inputs must not report missing connections")
            XCTAssertTrue((link.direction == .input ? component.configuredInputs(node.config):component.configuredOutputs(node.config)).contains(link.childPort),"Every binding must end at a real declared port")
        }
        XCTAssertTrue(canvas.canOpenBlock(container.id),"Inspector callback must accept inline MoE without a reusable-block definition")
        let address=WorkflowInlineContainer.address(root:root,childID:"expert_a/hidden")
        let id=try XCTUnwrap(canvas.projection.displayIDByAddress[address]);canvas.selectNode(id,extending:false)
        let context=try WorkflowModelNodeContext(owner:owner,address:address),source=owner.document
        XCTAssertTrue(context.setField("bias",value:false,expected:context.configuration),context.statusMessage)
        XCTAssertEqual(context.configuration["bias"],false)
        XCTAssertEqual(owner.selection.nodeAddresses,[address]);XCTAssertTrue(LayerParametersInspectorContext(context:context).choices.isEmpty)
        canvas.undo();var restored=owner.document;restored.revision=source.revision;XCTAssertEqual(restored,source)
        canvas.redo();let current=owner.authoringNode(at:address);if case .modelPrimitive(_,_,let config)=current?.payload{XCTAssertEqual(config["bias"],false)}else{XCTFail()}
        let reopened=try WorkflowCanvasAdapter(owner:owner,modelRegistry:owner.modelRegistry,dataRegistry:owner.dataRegistry);defer{reopened.invalidate()}
        XCTAssertEqual(reopened.inlineContainers.count,1);XCTAssertEqual(reopened.inlineContainers.first?.lanes.map(\.title),["Expert A","Expert B","Expert C","Expert D"])
        var opened=0;canvas.onOpenBlock={_,_ in opened+=1};canvas.openBlock(container.id)
        XCTAssertEqual(opened,0);XCTAssertFalse(try XCTUnwrap(canvas.inlineContainers.first).isExpanded)
        canvas.openBlock(container.id);XCTAssertTrue(try XCTUnwrap(canvas.inlineContainers.first).isExpanded)
    }
    func testMovingAndCollapseArePresentationOnlyWithTypedChainPorts()throws {
        let(_,owner,canvas)=try fixture();defer{canvas.invalidate();owner.invalidate()}
        let before=owner.document,address=WorkflowInlineContainer.address(root:root,childID:"expert_a/hidden")
        let id=try XCTUnwrap(canvas.projection.displayIDByAddress[address]),node=try XCTUnwrap(canvas.graph.nodes.first{$0.id==id})
        XCTAssertEqual(canvas.canvasPortDescriptor(nodeID:id,port:"out",direction:.output).kind,.tensor)
        canvas.moveNodes(ids:[id],dx:17,dy:13)
        XCTAssertEqual(owner.document,before);XCTAssertEqual(canvas.graph.nodes.first{$0.id==id}?.x,(node.x+17))
        canvas.toggleInlineContainer(try XCTUnwrap(canvas.inlineContainers.first).id)
        XCTAssertEqual(owner.document,before);XCTAssertFalse(canvas.graph.nodes.contains{$0.id==id})
    }
    func testDropUsesChainUnderPointerAndCannotCreateNestedMixture()throws {
        let(_,owner,canvas)=try fixture();defer{canvas.invalidate();owner.invalidate()}
        let input=WorkflowInlineContainer.address(root:root,childID:"expert_b/hidden"),id=try XCTUnwrap(canvas.projection.displayIDByAddress[input])
        let node=try XCTUnwrap(canvas.graph.nodes.first{$0.id==id}),entry=try XCTUnwrap(canvas.creationEntries.first{if case .modelPrimitive("relu",_,_)=$0.payload{return true};return false})
        let before=try WorkflowInlineContainer.definition(in:owner.document,root:root).body.nodes.count
        canvas.addNode(type:entry.id,x:node.x+600,y:node.y+20)
        XCTAssertEqual(try WorkflowInlineContainer.definition(in:owner.document,root:root).body.nodes.count,before+1,canvas.statusMessage)
        let selected=try XCTUnwrap(owner.selection.nodeAddresses.first),member=try XCTUnwrap(WorkflowInlineContainer.parse(in:owner.document,address:selected))
        XCTAssertTrue(member.childID.hasPrefix("expert_b/"))
        XCTAssertThrowsError(try WorkflowInlineContainer.apply(.insertNode(.init(id:"expert_b/bad",type:"mixture_experts")),to:owner.document,root:root,registry:owner.modelRegistry))
        canvas.undo();XCTAssertEqual(try WorkflowInlineContainer.definition(in:owner.document,root:root).body.nodes.count,before)
    }
}
