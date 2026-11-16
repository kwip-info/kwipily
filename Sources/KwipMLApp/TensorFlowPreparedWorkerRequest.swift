import Foundation
import KwipMLCore

struct TensorFlowBridgeError:Error,LocalizedError {
    let message:String
    init(_ message:String){self.message=message}
    var errorDescription:String? {message}
}

struct TensorFlowPreparedWorkerRequest:Codable,Equatable {
    struct Configuration:Codable,Equatable {let graph:GraphDocument;let settings:[String:JSONValue]}
    struct Prepared:Codable,Equatable {
        let manifest:String;let manifestSHA256:String;let graphDigest:String
        let trainingNode:String;let trainingPort:String;let validationNode:String;let validationPort:String
        let featureColumns:[String];let targetColumn:String
        var selection:PreparedTrainingSelection {.init(manifestPath:manifest,manifestSHA256:manifestSHA256,graphDigest:graphDigest,trainingNode:trainingNode,trainingPort:trainingPort,validationNode:validationNode,validationPort:validationPort,featureColumns:featureColumns,targetColumn:targetColumn)}
    }
    let schemaVersion:Int
    let backendID:String
    let configuration:Configuration
    let prepared:Prepared
    var capturePredictions:Bool? = nil
    init(graph:GraphDocument,settings:[String:JSONValue],selection:PreparedTrainingSelection,capturePredictions:Bool?=nil)throws {
        var graph=graph;graph.execution=nil
        schemaVersion=1;backendID="tensorflow";configuration = .init(graph:graph,settings:settings)
        prepared = .init(manifest:selection.manifestPath,manifestSHA256:selection.manifestSHA256,graphDigest:selection.graphDigest,trainingNode:selection.trainingNode,trainingPort:selection.trainingPort,validationNode:selection.validationNode,validationPort:selection.validationPort,featureColumns:selection.featureColumns,targetColumn:selection.targetColumn)
        self.capturePredictions=capturePredictions
        try validate()
    }
    func validate()throws {
        guard schemaVersion==1,backendID=="tensorflow",configuration.graph.execution==nil else {throw TensorFlowBridgeError("Invalid TensorFlow request identity.")}
        try TensorFlowEstimatorConfiguration.validate(configuration.settings)
        try TensorFlowEstimatorConfiguration.validateGraph(configuration.graph,registry:ComponentRegistry.builtIn())
        try prepared.selection.validate()
        guard WorkflowArchiveCodec.isHash(prepared.manifestSHA256),WorkflowArchiveCodec.isHash(prepared.graphDigest) else {throw TensorFlowBridgeError("TensorFlow preparation requires lowercase SHA-256 pins.")}
    }
    static func decode(_ data:Data)throws->Self {
        guard data.count<=2_000_000,let root=try JSONSerialization.jsonObject(with:data) as? [String:Any],Set(root.keys).isSubset(of:["schemaVersion","backendID","configuration","prepared","capturePredictions"]) && Set(["schemaVersion","backendID","configuration","prepared"]).isSubset(of:Set(root.keys)),let config=root["configuration"] as? [String:Any],Set(config.keys)==["graph","settings"],let selection=root["prepared"] as? [String:Any],Set(selection.keys)==["manifest","manifestSHA256","graphDigest","trainingNode","trainingPort","validationNode","validationPort","featureColumns","targetColumn"] else {throw TensorFlowBridgeError("Unknown or incomplete TensorFlow request fields.")}
        if root.keys.contains("capturePredictions"),root["capturePredictions"] is NSNull {throw TensorFlowBridgeError("Prediction capture must be an explicit boolean when present.")}
        let value=try JSONDecoder().decode(Self.self,from:data);try value.validate();return value
    }
}
