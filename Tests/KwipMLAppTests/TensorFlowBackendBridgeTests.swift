import XCTest
import Foundation
import Darwin
import KwipMLCore
@testable import KwipMLApp

final class TensorFlowBackendBridgeTests:XCTestCase {
    func testRequestRoundTripAndPrivateGraphProjection()throws {
        var graph=ExampleFlows.parallelClassifier;graph.execution = .init(datasetCSV:"/private/source.csv")
        let selection=PreparedTrainingSelection(manifestPath:"/tmp/preparation/manifest.json",manifestSHA256:String(repeating:"a",count:64),graphDigest:String(repeating:"b",count:64),trainingNode:"training",trainingPort:"table",validationNode:"heldout",validationPort:"table",featureColumns:["first","second"],targetColumn:"label")
        let request=try TensorFlowPreparedWorkerRequest(graph:graph,settings:TensorFlowEstimatorConfiguration.defaultSettings,selection:selection)
        let data=try JSONEncoder().encode(request)
        XCTAssertFalse(String(decoding:data,as:UTF8.self).contains("/private/source.csv"))
        XCTAssertEqual(try TensorFlowPreparedWorkerRequest.decode(data),request)
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any]);object["unknown"]=true
        XCTAssertThrowsError(try TensorFlowPreparedWorkerRequest.decode(JSONSerialization.data(withJSONObject:object)))
    }
    func testRuntimeDescriptorMustMatchExactly()throws {
        let data=try JSONEncoder().encode(TensorFlowRuntimeCapabilities.expected)
        XCTAssertEqual(try TensorFlowRuntimeCapabilities.decode(data).runtimeVersion,"2.21.0")
        var object=try XCTUnwrap(JSONSerialization.jsonObject(with:data) as? [String:Any]);object["devices"]=["cpu","gpu"]
        XCTAssertThrowsError(try TensorFlowRuntimeCapabilities.decode(JSONSerialization.data(withJSONObject:object)))
        object["devices"]=["cpu"];object["runtimeVersion"]="unknown"
        XCTAssertThrowsError(try TensorFlowRuntimeCapabilities.decode(JSONSerialization.data(withJSONObject:object)))
    }
    func testActualTensorFlowArtifactFixture()throws {
        guard let fixture=ProcessInfo.processInfo.environment["KWIPML_TENSORFLOW_FIXTURE"] else {throw XCTSkip("Opt in with an actual prepared TensorFlow run directory.")}
        let directory=URL(fileURLWithPath:fixture)
        let raw=try Data(contentsOf:directory.appendingPathComponent("manifest.json"))
        let root=try XCTUnwrap(JSONSerialization.jsonObject(with:raw) as? [String:Any])
        let config=try JSONDecoder().decode(TensorFlowPreparedWorkerRequest.Configuration.self,from:JSONSerialization.data(withJSONObject:try XCTUnwrap(root["configuration"])))
        let provenance=try XCTUnwrap(root["provenance"] as? [String:Any]);let prep=try XCTUnwrap(provenance["preparation"] as? [String:Any])
        let selection=PreparedTrainingSelection(manifestPath:directory.appendingPathComponent("preparation/manifest.json").path,manifestSHA256:try XCTUnwrap(prep["manifestSHA256"] as? String),graphDigest:try XCTUnwrap(prep["graphDigest"] as? String),trainingNode:"training",trainingPort:"table",validationNode:"heldout",validationPort:"table",featureColumns:["feature","second"],targetColumn:"group")
        let request=try TensorFlowPreparedWorkerRequest(graph:config.graph,settings:config.settings,selection:selection)
        var requestedSamples=request;requestedSamples.capturePredictions=true
        XCTAssertThrowsError(try TensorFlowWorkflowResult.verify(directory:directory,request:requestedSamples))
        let result=try TensorFlowWorkflowResult.verify(directory:directory,request:request)
        XCTAssertEqual(result.evaluation.samples,16)
        var event:[String:Any]=["event":"backendCompleted","backendID":"tensorflow","manifest":directory.appendingPathComponent("manifest.json").path,"evaluation":try XCTUnwrap(root["evaluation"])]
        XCTAssertNoThrow(try TensorFlowWorkflowResult.validateCompletion(JSONSerialization.data(withJSONObject:event),directory:directory,request:request))
        event["backendID"]="sklearn"
        XCTAssertThrowsError(try TensorFlowWorkflowResult.validateCompletion(JSONSerialization.data(withJSONObject:event),directory:directory,request:request))
    }
}

extension TensorFlowBackendBridgeTests {
    /// Exercise JSON numeric representation, not a claim of re-training this fixture.
    func testWholeNumberLearningRateMatchesCanonicalJSON()throws {
        guard let fixture=ProcessInfo.processInfo.environment["KWIPML_TENSORFLOW_FIXTURE"] else {throw XCTSkip("Requires TensorFlow artifact fixture.")}
        let directory=try canonicalBackendTemporaryDirectory()
        try FileManager.default.copyItem(at:URL(fileURLWithPath:fixture),to:directory)
        defer {try? FileManager.default.removeItem(at:directory)}
        let file=directory.appendingPathComponent("manifest.json")
        var root=try XCTUnwrap(JSONSerialization.jsonObject(with:Data(contentsOf:file)) as? [String:Any])
        var config=try XCTUnwrap(root["configuration"] as? [String:Any]);var settings=try XCTUnwrap(config["settings"] as? [String:Any]);settings["learningRate"]=1;config["settings"]=settings;root["configuration"]=config
        try JSONSerialization.data(withJSONObject:root,options:.sortedKeys).write(to:file)
        let typed=try JSONDecoder().decode(TensorFlowPreparedWorkerRequest.Configuration.self,from:JSONSerialization.data(withJSONObject:config))
        let provenance=try XCTUnwrap(root["provenance"] as? [String:Any]);let prep=try XCTUnwrap(provenance["preparation"] as? [String:Any])
        let selection=PreparedTrainingSelection(manifestPath:directory.appendingPathComponent("preparation/manifest.json").path,manifestSHA256:try XCTUnwrap(prep["manifestSHA256"] as? String),graphDigest:try XCTUnwrap(prep["graphDigest"] as? String),trainingNode:"training",trainingPort:"table",validationNode:"heldout",validationPort:"table",featureColumns:["feature","second"],targetColumn:"group")
        var requested=typed.settings;requested["learningRate"] = .number(1.0)
        let request=try TensorFlowPreparedWorkerRequest(graph:typed.graph,settings:requested,selection:selection)
        XCTAssertNoThrow(try TensorFlowWorkflowResult.verify(directory:directory,request:request))
    }
}
