import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TransformerArchiveRoundTripTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testDecodedPlanReproducesSourceAndRefusesChangedDecoder()throws {
        let registry=try ComponentRegistry.builtIn()
        let data=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        let source=try WorkflowExamples.preparedEditableTransformerClassifier(device:"cpu")
        let document=try WorkflowModelBoundary.consolidate(RetiredToolAuthoringUpgrade.plan(source))
        let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:15)
        let captured=try WorkflowArchivePlan(plan,compilerContractVersion:15)
        let bytes=try WorkflowArchiveCodec.encode(captured)
        let decoded=try JSONDecoder().decode(WorkflowArchivePlan.self,from:bytes)
        XCTAssertTrue(try decoded.matches(plan,compilerContractVersion:15))
        XCTAssertNoThrow(try decoded.validateModelSnapshot(registry:registry,context:.preparedWorkflow))
        XCTAssertEqual(try WorkflowArchiveCodec.encode(decoded),bytes)
        let root=try XCTUnwrap(document.nodes.first {if case .modelBlock(let reference,_)=$0.payload {return reference==TransformerBlock.reference};return false})
        var values=try XCTUnwrap(document.definitions.first {$0.reference==TransformerBlock.reference}?.body.nodes.first {$0.id=="decoder"}?.config)
        values["dim_feedforward"]=7
        let changed=try WorkflowInlineContainer.apply(.setAttention(childID:"decoder",configuration:values),to:document,root:root.address,registry:registry,idProvider:{"changed-transformer"})
        let different=try WorkflowCompiler.compile(changed,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:15)
        XCTAssertFalse(try decoded.matches(different,compilerContractVersion:15))
    }
    func testActualCompletedNativeArchiveIndependentlyReopens()throws {
        guard let path=ProcessInfo.processInfo.environment["KWIPML_TRANSFORMER_HISTORY_DIRECTORY"] else {throw XCTSkip("Select the synthetic native Transformer run for readback acceptance")}
        let directory=URL(fileURLWithPath:path)
        let projectID=directory.deletingLastPathComponent().lastPathComponent
        let reference=WorkflowHistoryReference(projectID:projectID,directoryID:try XCTUnwrap(UUID(uuidString:directory.lastPathComponent)))
        let root=directory.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let result=WorkflowHistoryReader.load(reference,root:root)
        guard case .verified(let receipt)=result else {return XCTFail("Native Transformer archive failed independent readback: \(result)")}
        XCTAssertEqual(receipt.outcome.status,.completed)
        XCTAssertNotNil(receipt.plan.model.provenance)
    }
}
