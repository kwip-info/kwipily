import XCTest
import KwipMLCore
@testable import KwipMLApp
@MainActor final class TensorOperationContextTests:XCTestCase {
    func testEveryInspectorUsesAtomicSelectedSourceAndUndo()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn(),library=ProjectLibrary(directory:root)
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:repo.appendingPathComponent("runner/dataflow-operators.json"))
        for type in TensorOperationAuthoring.componentIDs where type != "matmul" {
            let address=WorkflowNodeAddress(domain:.model,localID:"operation")
            let config=try TensorOperationAuthoring.defaults(type,registry:registry)
            let document=WorkflowDocument(projectID:UUID().uuidString,nodes:[.init(address:address,payload:.modelPrimitive(type:type,configurationVersion:2,config:config))])
            _=try library.saveSource(.workflow(document))
            let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,dataRegistry:data)
            defer {owner.invalidate()}
            XCTAssertTrue(owner.select(.init(nodeAddresses:[address]),lease:owner.lease()))
            let context=try WorkflowModelNodeContext(owner:owner,address:address)
            var edited=config
            switch type {
            case "add":edited["alpha"] = -0.5
            case "transpose":edited["dim0"] = -1
            case "permute":edited["dims"] = .array([0,1,2])
            case "concat","stack","split","chunk","gather":edited["dim"] = 1
            case "einsum":edited=["pattern":.string("diagonal"),"equation":.string("naa->na")]
            default:edited["stride"] = .array([2,1])
            }
            XCTAssertTrue(context.setTensorOperation(type,configuration:edited,expectedType:type,expected:config),context.statusMessage)
            guard case .modelPrimitive(let actual,_,let values)=owner.document.nodes[0].payload else {return XCTFail("Expected original primitive")}
            XCTAssertEqual(actual,type);XCTAssertEqual(values,try TensorOperationAuthoring.canonical(type,edited))
            XCTAssertFalse(context.setTensorOperation(type,configuration:config,expectedType:type,expected:config))
            XCTAssertTrue(owner.undo(lease:owner.lease()));XCTAssertEqual(owner.document.nodes,document.nodes)
            XCTAssertTrue(owner.select(.init(),lease:owner.lease()))
            XCTAssertFalse(context.setTensorOperation(type,configuration:edited,expectedType:type,expected:config))
        }
    }
    func testConsolidatedPlacementAndOriginalExamplesRemainAvailable()throws {
        let registry=try ComponentRegistry.builtIn()
        let entries=registry.components.filter {TensorOperationAuthoring.componentIDs.contains($0.id)}.map {ComponentRibbonEntry(id:$0.id,title:$0.title,shortTitle:$0.title,purpose:"",category:$0.category,typeIcon:$0.id)}
        let offered=TensorOperationPalette.consolidated(entries:entries)
        XCTAssertEqual(offered.count,10)
        XCTAssertFalse(offered.contains {TensorOperationPalette.replacedTypes.contains($0.typeIcon)})
        XCTAssertEqual(TensorOperationPalette.consolidated(entries:entries.filter {$0.id=="permute"}).first?.id,"permute")
        for type in TensorOperationAuthoring.componentIDs {
            XCTAssertNotNil(LearnCatalogContent.componentExampleEntries.first {$0.componentID==type})
            let document=try PreparedWorkflowExample.component(type).document(device:"cpu")
            XCTAssertTrue(document.nodes.contains {if case .modelPrimitive(let id,_,_)=$0.payload {return id==type};return false})
        }
    }
}
