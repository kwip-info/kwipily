import XCTest
@testable import KwipMLCore
final class AttentionCanonicalExampleTests:XCTestCase {
    func testFiveActualFactoriesCompileWithDerivedBranchesAndMasks()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let factories:[(String,()throws->WorkflowDocument)]=[
            ("multihead_attention",{try WorkflowExamples.preparedMultiheadAttentionClassifier(device:"cpu")}),
            ("transformer",{try WorkflowExamples.preparedTransformerClassifier(device:"cpu")}),
            ("transformer_encoder",{try WorkflowExamples.preparedTransformerEncoderClassifier(device:"cpu")}),
            ("transformer_decoder",{try WorkflowExamples.preparedTransformerDecoderClassifier(device:"cpu")}),
            ("scaled_dot_product_attention",{try WorkflowExamples.preparedScaledDotProductAttentionClassifier(device:"cpu")})]
        for (id,factory) in factories {
            let document=try factory()
            let train=try XCTUnwrap(document.nodes.first {if case .train=$0.payload {return true};return false})
            let plan=try WorkflowCompiler.compile(document,train:train.address,modelRegistry:.builtIn(),dataRegistry:data,contractVersion:6)
            let graph=plan.model.graph
            XCTAssertEqual(graph.nodes.filter {$0.type=="input"}.count,1)
            XCTAssertEqual(graph.nodes.filter {$0.type==id}.count,1)
            XCTAssertEqual(graph.nodes.filter {$0.type=="attention_mask"}.count,1)
            XCTAssertTrue(graph.edges.contains {$0.sourcePort=="mask" && $0.targetPort != "in"})
            XCTAssertEqual(GraphValidator.validate(graph,registry:try .builtIn(),context:.preparedWorkflow).outputTensorContracts[.init("output","out")]?.layout.sample.dimensions,[2])
            if let path=ProcessInfo.processInfo.environment["KWIPML_ATTENTION_CANONICAL_FIXTURES"] {
                let directory=URL(fileURLWithPath:path).appendingPathComponent(id)
                try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:true)
                let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.prettyPrinted]
                try encoder.encode(document).write(to:directory.appendingPathComponent("workflow.json"))
                try encoder.encode(graph).write(to:directory.appendingPathComponent("model.json"))
                try encoder.encode(plan.dataflow).write(to:directory.appendingPathComponent("dataflow.json"))
                try encoder.encode(plan.settings).write(to:directory.appendingPathComponent("settings.json"))
            }
        }
    }
    func testAllSixPaletteExampleAliasesOpenRealMaskedArchitectures()throws {
        for id in AttentionConfiguration.componentIDs {
            let document=try PreparedWorkflowExample.component(id).document(device:"cpu")
            let train=try XCTUnwrap(document.nodes.first {if case .train=$0.payload {return true};return false})
            let types=document.nodes.compactMap {node -> String? in if case .modelPrimitive(let type,_,_)=node.payload {return type};return nil}
            XCTAssertEqual(types.filter {$0=="attention_mask"}.count,id=="rotary_position" ? 0:1)
            if id == "transformer" {
                XCTAssertTrue(document.nodes.contains {if case .modelBlock(let ref,_)=$0.payload {return ref.id==TransformerBlock.reference.id};return false})
                let definition=try XCTUnwrap(document.definitions.first {$0.reference==TransformerBlock.reference})
                XCTAssertEqual(definition.body.nodes.map(\.type),["transformer_encoder","transformer_decoder"])
            } else {XCTAssertTrue(types.contains(id == "attention_mask" ? "multihead_attention":id))}
            XCTAssertFalse(train.name.isEmpty)
        }
    }

}
