import XCTest
import KwipMLCore
@testable import KwipMLApp

final class TensorRoutingCaptureTests: XCTestCase {
    private func tensor(_ shape: [JSONValue], _ values: [JSONValue] = [1]) -> JSONValue {
        .object(["shape": .array(shape), "dtype": .string("float32"), "values": .array(values)])
    }
    private func fixture(_ type: String = "chunk") -> (GraphDocument, [String:JSONValue]) {
        let graph = GraphDocument(nodes: [
            .init(id:"source",type:"input",config:["features":13]),
            .init(id:"routing",type:type,config:type == "chunk" ? ["chunks":6,"dim":1] : ["split_size":3,"dim":1])
        ],edges:[.init(id:"wire",source:"source",sourcePort:"out",target:"routing",targetPort:"in")])
        let parts = Dictionary(uniqueKeysWithValues:(0..<5).map { ("part\($0)",tensor([8,.integer($0 == 4 ? 1:3)])) })
        return (graph,["source":tensor([8,13]),"routing":.object(["outputs":.object(parts)])])
    }
    func testActualSplitAndUnevenChunkPortsAccept() throws {
        for type in ["split","chunk"] {
            let (graph,previews) = fixture(type)
            XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews,graph:graph,sampleCapture:nil))
        }
    }
    func testMissingExtraWrongShapeAndOversizedValueCapturesRefuse() throws {
        let (graph,previews) = fixture()
        for mutation in 0..<4 {
            var changed=previews
            guard case .object(var row)=changed["routing"],case .object(var parts)=row["outputs"] else {return XCTFail()}
            switch mutation {
            case 0: parts.removeValue(forKey:"part4")
            case 1: parts["part5"] = tensor([8,1])
            case 2: parts["part4"] = tensor([8,3])
            default: parts["part4"] = tensor([8,1],Array(repeating:1,count:9))
            }
            row["outputs"] = .object(parts);changed["routing"] = .object(row)
            XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(changed,graph:graph,sampleCapture:nil))
        }
    }
    func testPrivacyPolicyAppliesToEveryPart() throws {
        let (graph,previews)=fixture();let policy=TrainingSampleCapturePolicy(activationValues:false,predictionRows:false)
        var hidden=previews
        for key in hidden.keys {
            guard case .object(var row)=hidden[key] else {return XCTFail()}
            row["captureStatus"] = .string("omittedByPolicy")
            if case .object(var parts)=row["outputs"] {
                for port in parts.keys {if case .object(var leaf)=parts[port] {leaf["values"] = .array([]);parts[port] = .object(leaf)}}
                row["outputs"] = .object(parts)
            } else {row["values"] = .array([])}
            hidden[key] = .object(row)
        }
        XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyActivationPreviews(hidden,graph:graph,sampleCapture:policy))
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyActivationPreviews(previews,graph:graph,sampleCapture:policy))
    }
}
