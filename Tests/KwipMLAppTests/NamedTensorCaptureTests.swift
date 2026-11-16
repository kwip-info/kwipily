import XCTest
import KwipMLCore
@testable import KwipMLApp

final class NamedTensorCaptureTests:XCTestCase {
    private func event(_ outputs:[String:JSONValue],status:JSONValue?=nil)->[String:JSONValue] {
        var preview:[String:JSONValue]=["outputs":.object(outputs)]
        preview["captureStatus"]=status
        return ["previews":.object(["pool":.object(preview)]),"predictions":.array([]),"predictionRows":.array([]),"predictionInput":.object(["dtype":.string("float32"),"sampleShape":.array([2])])]
    }
    private var outputs:[String:JSONValue] {
        ["out":.object(["shape":.array([1,2,3]),"dtype":.string("float32"),"values":.array([1.5])]),
         "indices":.object(["shape":.array([1,2,3]),"dtype":.string("int64"),"values":.array([9007199254740993])])]
    }
    private func graph(_ enabled:Bool=true)->GraphDocument {
        .init(nodes:[.init(id:"pool",type:"max_pool1d",config:["return_indices":.bool(enabled)])])
    }
    func testSpatialPoolingCaptureUsesExactPortsAndPrivacyPolicy() throws {
        var spatial=graph();spatial.nodes[0].type="max_pool2d"
        var values=outputs
        for key in values.keys {
            if case .object(var fields)=values[key] {fields["shape"] = .array([1,2,3,4]);values[key] = .object(fields)}
        }
        XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyPresentation(event:event(values),graph:spatial,sampleCapture:nil))
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:event(values),graph:spatial,sampleCapture:.init(activationValues:false,predictionRows:false)))
        spatial.nodes[0].config["return_indices"]=false
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:event(values),graph:spatial,sampleCapture:nil))
    }
    func testNamedCaptureAcceptsIndependentTypesAndExactIntegers() throws {
        XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyPresentation(event:event(outputs),graph:graph(),sampleCapture:nil))
    }
    func testMissingExtraMistypedAndDisabledOutputsRefuse() {
        var missing=outputs;missing.removeValue(forKey:"indices")
        var extra=outputs;extra["other"]=outputs["out"]
        var wrong=outputs;wrong["indices"] = .object(["shape":.array([1,2,3]),"dtype":.string("float32"),"values":.array([1.5])])
        var fraction=outputs;fraction["indices"] = .object(["shape":.array([1,2,3]),"dtype":.string("int64"),"values":.array([1.5])])
        for value in [missing,extra,wrong,fraction] {
            XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:event(value),graph:graph(),sampleCapture:nil))
        }
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:event(outputs),graph:graph(false),sampleCapture:nil))
    }
    func testOmissionPolicyAppliesToEveryNamedOutput() throws {
        let policy=TrainingSampleCapturePolicy(activationValues:false,predictionRows:false)
        var empty=outputs
        for key in empty.keys {
            if case .object(var fields)=empty[key] {fields["values"] = .array([]);empty[key] = .object(fields)}
        }
        XCTAssertNoThrow(try WorkflowModelTrainingExecution.verifyPresentation(event:event(empty,status:.string("omittedByPolicy")),graph:graph(),sampleCapture:policy))
        XCTAssertThrowsError(try WorkflowModelTrainingExecution.verifyPresentation(event:event(outputs,status:.string("omittedByPolicy")),graph:graph(),sampleCapture:policy))
    }
    func testVerifiedCapturesResolveExactOriginAndCarryPolicy() throws {
        let origins = ["pool": CompositeNodeOrigin(instancePath:["block"],localNodeID:"inner",definition:nil)]
        let capture = try WorkflowVerifiedModelCaptures(event:event(outputs),graph:graph(),origins:origins,sampleCapture:nil)
        XCTAssertNil(capture.ports(instancePath:[],localNodeID:"inner"))
        XCTAssertNil(capture.ports(instancePath:["other"],localNodeID:"inner"))
        let ports = try XCTUnwrap(capture.ports(instancePath:["block"],localNodeID:"inner"))
        guard case .object(let indices) = ports["indices"] else { return XCTFail("Missing indices") }
        XCTAssertEqual(indices["values"],.array([9007199254740993]))
        XCTAssertNil(ports["missing"])
        XCTAssertThrowsError(try WorkflowVerifiedModelCaptures(event:event(outputs),graph:graph(),origins:[:],sampleCapture:nil))
        let policy = TrainingSampleCapturePolicy(activationValues:false,predictionRows:false)
        var omitted = outputs
        for key in omitted.keys {
            if case .object(var item) = omitted[key] { item["values"] = .array([]); omitted[key] = .object(item) }
        }
        let privateCapture = try WorkflowVerifiedModelCaptures(event:event(omitted,status:.string("omittedByPolicy")),graph:graph(),origins:origins,sampleCapture:policy)
        guard case .object(let hidden) = privateCapture.ports(instancePath:["block"],localNodeID:"inner")?["indices"] else { return XCTFail("Missing omitted metadata") }
        XCTAssertEqual(hidden["captureStatus"],.string("omittedByPolicy"))
    }

    func testNestedBlockBoundaryPreservesNamedPortAndRefusesMissingOutput() throws {
        let innerRef = CompositeDefinitionReference("inner"), outerRef = CompositeDefinitionReference("outer")
        let inner = CompositeDefinition(reference:innerRef,title:"Inner",body:.init(nodes:graph().nodes),inputs:[],outputs:[.init("positions",source:.init("pool","indices")),.init("values",source:.init("pool","out")),.init("absent",source:.init("pool","missing"))])
        let outer = CompositeDefinition(reference:outerRef,title:"Outer",body:.init(instances:[.init(id:"nested",definition:innerRef)]),inputs:[],outputs:[.init("selected",source:.init("nested","positions"))])
        let project = CompositeProjectDocument(root:.init(instances:[.init(id:"block",definition:outerRef)]),definitions:[inner,outer])
        let captures = try WorkflowVerifiedModelCaptures(event:event(outputs),graph:graph(),origins:["pool":.init(instancePath:["block","nested"],localNodeID:"pool",definition:innerRef)],sampleCapture:nil,project:project)
        let ports = try XCTUnwrap(captures.resolvedPorts(instancePath:[],localNodeID:"block"))
        XCTAssertEqual(ports["selected"],outputs["indices"])
        XCTAssertNil(ports["out"])
        let nested = try XCTUnwrap(captures.resolvedPorts(instancePath:["block"],localNodeID:"nested"))
        XCTAssertEqual(nested["positions"],outputs["indices"])
        XCTAssertNil(nested["absent"])
        XCTAssertNil(captures.resolvedPorts(instancePath:["wrong"],localNodeID:"nested"))
    }

    func testArchivedActivationsVerifyIdentityHashAndCaptureContract() throws {
        let runID = UUID().uuidString, graphHash = String(repeating:"a",count:64)
        let payload: [String:JSONValue] = ["schemaVersion":1,"runID":.string(runID),"graphHash":.string(graphHash),"selection":.string("firstValidationRows"),"previews":event(outputs)["previews"]!]
        let bytes = try JSONEncoder().encode(payload)
        let metadata: JSONValue = .object(["schemaVersion":1,"file":.string("activations.json"),"sha256":.string(WorkflowPhaseArtifacts.hash(bytes)),"bytes":.integer(Int64(bytes.count))])
        let decoded = try WorkflowActivationArtifact.decode(bytes,metadata:metadata,runID:runID,graphHash:graphHash,graph:graph(),sampleCapture:nil)
        XCTAssertEqual(decoded,event(outputs)["previews"])
        XCTAssertThrowsError(try WorkflowActivationArtifact.decode(bytes,metadata:metadata,runID:UUID().uuidString,graphHash:graphHash,graph:graph(),sampleCapture:nil))
        XCTAssertThrowsError(try WorkflowActivationArtifact.decode(bytes+Data([32]),metadata:metadata,runID:runID,graphHash:graphHash,graph:graph(),sampleCapture:nil))
        XCTAssertThrowsError(try WorkflowActivationArtifact.decode(bytes,metadata:metadata,runID:runID,graphHash:graphHash,graph:graph(false),sampleCapture:nil))
        XCTAssertThrowsError(try WorkflowActivationArtifact.decode(bytes,metadata:metadata,runID:runID,graphHash:graphHash,graph:graph(),sampleCapture:.init(activationValues:false,predictionRows:false)))
    }

}
