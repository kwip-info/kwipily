import Foundation

public struct TrainingControlPolicy:Codable,Equatable {
    public struct EarlyStopping:Codable,Equatable {
        public var enabled:Bool
        public var patience:Int
        public var minimumImprovement:Double
        public init(enabled:Bool=false,patience:Int=5,minimumImprovement:Double=0.001) {self.enabled=enabled;self.patience=patience;self.minimumImprovement=minimumImprovement}
    }
    public let schemaVersion:Int
    public var modelSelection:String
    public var monitor:String
    public var earlyStopping:EarlyStopping
    public static let factory=TrainingControlPolicy()
    /// Shared discovery for clients editing the same inline Train configuration.
    public static var configurationSchema:JSONValue {
        .object(["type":.string("object"),"additionalProperties":false,"required":.array(["schemaVersion","modelSelection","monitor","earlyStopping"].map(JSONValue.string)),"properties":.object([
            "schemaVersion":.object(["const":1]),
            "modelSelection":.object(["enum":.array(["finalEpoch","bestValidation"].map(JSONValue.string))]),
            "monitor":.object(["const":.string("validationLoss")]),
            "earlyStopping":.object(["type":.string("object"),"additionalProperties":false,"required":.array(["enabled","patience","minimumImprovement"].map(JSONValue.string)),"properties":.object([
                "enabled":.object(["type":.string("boolean")]),"patience":.object(["type":.string("integer"),"minimum":1,"maximum":10000]),
                "minimumImprovement":.object(["type":.string("number"),"minimum":0,"maximum":1_000_000])
            ])])
        ])])
    }

    public init(modelSelection:String="finalEpoch",monitor:String="validationLoss",earlyStopping:EarlyStopping = .init()) {schemaVersion=1;self.modelSelection=modelSelection;self.monitor=monitor;self.earlyStopping=earlyStopping}
    public func validate()throws {
        guard schemaVersion==1,["finalEpoch","bestValidation"].contains(modelSelection),monitor=="validationLoss",(1...10000).contains(earlyStopping.patience),earlyStopping.minimumImprovement.isFinite,(0...1_000_000).contains(earlyStopping.minimumImprovement) else {throw GraphConfigurationError("Choose a supported model-selection policy, positive patience and finite nonnegative minimum improvement.")}
    }
    public init(from decoder:Decoder)throws {
        let value=try JSONValue(from:decoder)
        guard case .object(let raw)=value,Set(raw.keys)==["schemaVersion","modelSelection","monitor","earlyStopping"],raw["schemaVersion"] == .integer(1),
              case .string(let selected)=raw["modelSelection"],case .string(let monitor)=raw["monitor"],case .object(let stop)=raw["earlyStopping"],Set(stop.keys)==["enabled","patience","minimumImprovement"],case .bool(let enabled)=stop["enabled"],case .integer(let patience)=stop["patience"],patience>=1,patience<=10000,let delta=stop["minimumImprovement"]?.doubleValue else {throw GraphConfigurationError("Unknown or malformed training-control policy.")}
        self.init(modelSelection:selected,monitor:monitor,earlyStopping:.init(enabled:enabled,patience:Int(patience),minimumImprovement:delta));try validate()
    }
}
