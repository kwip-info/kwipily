import Foundation
import KwipMLCore

/// Availability comes from the actual worker response, never from a menu selection.
struct TensorFlowRuntimeCapabilities:Equatable {
    let runtimeVersion:String
    static func decode(_ data:Data)throws->Self {
        guard data.count<=8192,case .object(let value)=try JSONDecoder().decode(JSONValue.self,from:data),
              .object(value)==expected else {throw TensorFlowBridgeError("TensorFlow runtime does not match the reviewed CPU foundation contract.")}
        return .init(runtimeVersion:"2.21.0")
    }
    static var expected:JSONValue {
        .object(["schemaVersion":.integer(1),"backendID":.string("tensorflow"),"runtimeVersion":.string("2.21.0"),
                 "devices":.array([.string("cpu")]),"supportedOperations":.array(["construct","fit","predict","evaluate","weightsSave","weightsLoad","export"].map(JSONValue.string)),
                 "unsupportedOperations":.array(["gpu","resumeOptimizer","customCode","arbitraryLayers"].map(JSONValue.string)),
                 "components":.array(["input","linear","relu","tanh","add","concat","output"].map(JSONValue.string)),
                 "stateFormat":.string("keras-weights-h5"),"task":.string("binaryClassification")])
    }
}
