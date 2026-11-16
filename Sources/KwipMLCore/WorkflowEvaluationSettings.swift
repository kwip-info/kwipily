import Foundation

/// Evaluation processes the full protected population. Preview controls display
/// only; training, optimization and experiment-search fields are not accepted.
public struct WorkflowEvaluationSettings:Codable,Equatable {
    public let partitionRole:String
    public var partitionTitle:String { partitionRole == "validation" ? "Validation" : "Test" }
    public let device:String
    public let batchSize:Int
    public let previewLimit:Int
    public init(device:String="auto",batchSize:Int=256,previewLimit:Int=8,partitionRole:String="test") {
        self.partitionRole=partitionRole;self.device=device;self.batchSize=batchSize;self.previewLimit=previewLimit
    }
    public func validate()throws {
        guard ["test","validation"].contains(partitionRole),["auto","cpu","mps"].contains(device),(1...100000).contains(batchSize),(0...100).contains(previewLimit) else {
            throw DataFlowContractError("Evaluation requires device auto, cpu or mps, batch size 1–100000, and preview limit 0–100.")
        }
    }
    public static func decode(_ config:[String:JSONValue])throws->Self {
        guard Set(config.keys).isSubset(of:["device","batchSize","previewLimit","settingsVersion","partitionRole"]) else {
            throw DataFlowContractError("Unknown evaluation setting; training and experiment settings do not apply to evaluation.")
        }
        let role:String
        if config["settingsVersion"] == nil && config["partitionRole"] == nil { role="test" }
        else {
            guard config["settingsVersion"] == .integer(2),config["partitionRole"] == .string("validation") else {throw DataFlowContractError("Evaluation role requires explicit version 2 Validation settings; omitted role means Test.")}
            role="validation"
        }
        var device="auto",batch=256,preview=8
        if let value=config["device"] {
            guard case .string(let choice)=value else {throw DataFlowContractError("Evaluation device must be a string.")}
            device=choice
        }
        if let value=config["batchSize"] {
            guard case .integer(let number)=value,let exact=Int(exactly:number) else {throw DataFlowContractError("Evaluation batch size must be an integer.")}
            batch=exact
        }
        if let value=config["previewLimit"] {
            guard case .integer(let number)=value,let exact=Int(exactly:number) else {throw DataFlowContractError("Evaluation preview limit must be an integer.")}
            preview=exact
        }
        let settings=Self(device:device,batchSize:batch,previewLimit:preview,partitionRole:role)
        try settings.validate();return settings
    }
    public func encode(to encoder:Encoder)throws {
        try validate()
        var values:[String:JSONValue] = ["device":.string(device),"batchSize":.integer(Int64(batchSize)),"previewLimit":.integer(Int64(previewLimit))]
        if partitionRole == "validation" { values["settingsVersion"] = .integer(2); values["partitionRole"] = .string(partitionRole) }
        try values.encode(to:encoder)
    }
    public init(from decoder:Decoder)throws {
        let values=try [String:JSONValue](from:decoder)
        self=try Self.decode(values)
    }
}
