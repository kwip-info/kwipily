import XCTest
import KwipMLCore
@testable import KwipMLApp
final class AttentionTrainingPresentationTests:XCTestCase {
    private let types=["multihead_attention","transformer","transformer_encoder","transformer_decoder","scaled_dot_product_attention"]
    private func tensor(_ shape:[Int],dtype:String="float32",values:[JSONValue]=[1])->JSONValue {.object(["shape":.array(shape.map {.integer(Int64($0))}),"dtype":.string(dtype),"values":.array(values)])}
    private func fixture(_ type:String,overrides:[String:JSONValue]=[:])throws->(GraphDocument,[String:JSONValue]) {
        var graph=try ExampleFlows.attentionClassifier(componentID:type)
        if let index=graph.nodes.firstIndex(where:{$0.id=="attention"}) {graph.nodes[index].config.merge(overrides) {_,new in new}}
        let validation=GraphValidator.validate(graph,registry:try .builtIn())
        XCTAssertTrue(validation.issues.isEmpty,"\(validation.issues)")
        var previews:[String:JSONValue]=[:]
        for node in graph.nodes {
            let contracts=validation.outputTensorContracts.filter {$0.key.nodeID==node.id}
            let named=Dictionary(uniqueKeysWithValues:contracts.map {address,contract in (address.portID,tensor([8]+contract.layout.sample.dimensions,dtype:contract.dtype.rawValue,values:contract.dtype == .int64 ? [0]:[1]))})
            previews[node.id]=AttentionConfiguration.componentIDs.contains(node.type) ? .object(["outputs":.object(named)]):named["out"]!
        }
        return (graph,previews)
    }
    func testActualAttentionAndUtilityNamedRolesAcceptIncludingUnusedWeights()throws {
        for type in types {let(g,p)=try fixture(type);XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(p,graph:g,sampleCapture:nil),type)}
    }
    func testUnusedWeightsAndMasksRefuseMissingForeignShapeBatchAndBinaryValues()throws {
        let(g,original)=try fixture("multihead_attention")
        for mutation in 0..<8 {
            var p=original
            guard case .object(var row)=p["attention"],case .object(var outputs)=row["outputs"] else {return XCTFail()}
            switch mutation {
            case 0:outputs.removeValue(forKey:"weights")
            case 1:outputs["extra"]=tensor([8,2,2])
            case 2:outputs["weights"]=tensor([8,2,3])
            case 3:outputs["weights"]=tensor([7,1,2,3])
            case 4:outputs["weights"]=tensor([8,1,2,3],dtype:"int64")
            case 5:outputs["weights"]=tensor([8,1,2,3],values:Array(repeating:1,count:9))
            case 6:p["mask"] = .object(["outputs":.object(["mask":tensor([8,2,3],dtype:"int64",values:[2]),"padding":tensor([8,3],dtype:"int64",values:[0])])])
            default:p["mask"] = .object(["outputs":.object(["mask":tensor([7,2,3],dtype:"int64",values:[0]),"padding":tensor([8,3],dtype:"int64",values:[0])])])
            }
            row["outputs"] = .object(outputs);p["attention"] = .object(row)
            XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(p,graph:g,sampleCapture:nil),"mutation \(mutation)")
        }
    }
    func testOptionalWeightsAveragingAndNativeAugmentedSourcePositions()throws {
        for average in [false,true] {
            let(g,p)=try fixture("multihead_attention",overrides:["average_weights":.bool(average),"add_bias_kv":true,"add_zero_attn":true])
            guard case .object(let row)=p["attention"],case .object(let outputs)=row["outputs"],case .object(let weights)=outputs["weights"] else {return XCTFail()}
            XCTAssertEqual(weights["shape"],.array(average ? [8,2,5]:[8,1,2,5]))
            XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(p,graph:g,sampleCapture:nil))
        }
        let(g,p)=try fixture("multihead_attention",overrides:["return_weights":false])
        guard case .object(let row)=p["attention"],case .object(let outputs)=row["outputs"] else {return XCTFail()}
        XCTAssertEqual(Set(outputs.keys),["out"])
        XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(p,graph:g,sampleCapture:nil))
        var wrong=p;wrong["attention"] = .object(["outputs":.object(outputs.merging(["weights":tensor([8,1,2,3])]) {_,new in new})])
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(wrong,graph:g,sampleCapture:nil))
    }
    func testCapturePolicyAppliesToEveryNamedMaskAndResult()throws {
        for type in types {let(g,original)=try fixture(type)
            for captured in [false,true] {
                let policy=TrainingSampleCapturePolicy(activationValues:captured,predictionRows:false)
                var p=original
                for key in p.keys {guard case .object(var row)=p[key] else {return XCTFail()};row["captureStatus"] = .string(captured ? "captured":"omittedByPolicy")
                    if case .object(var outputs)=row["outputs"] {for role in outputs.keys {guard case .object(var leaf)=outputs[role] else {return XCTFail()};leaf["values"] = .array(captured ? (leaf["dtype"] == .string("int64") ? [0]:[1]):[]);outputs[role] = .object(leaf)};row["outputs"] = .object(outputs)} else {row["values"] = .array(captured ? [1]:[])}
                    p[key] = .object(row)
                }
                XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(p,graph:g,sampleCapture:policy))
                XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(original,graph:g,sampleCapture:policy))
            }
        }
    }
}
