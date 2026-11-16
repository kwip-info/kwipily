import Foundation

/// Versioned inference settings. No model or local filesystem authority lives here.
public struct TextGenerationConfiguration:Equatable {
    public let values:[String:JSONValue]
    public static let defaults:[String:JSONValue] = ["method":.string("greedy"),"temperature":.number(1),"topK":.integer(0),"topP":.number(1),"seed":.integer(0),"maxNewTokens":.integer(64),"maxPrompts":.integer(128),"maxTotalTokens":.integer(8192),"longPrompt":.string("refuse"),"allowUnknown":.bool(false),"stopAtEOS":.bool(true),"device":.string("cpu"),"promptMode":.string("one"),"prompt":.string(""),"promptColumn":.string(""),"textColumn":.string("generated_text"),"idsColumn":.string("generated_ids"),"countColumn":.string("new_tokens"),"stopColumn":.string("stop_reason")]
    public init(_ config:[String:JSONValue])throws {
        guard Set(config.keys).isSubset(of:Set(Self.defaults.keys)) else {throw DataFlowContractError("Unknown Generate text setting.")}
        let c=Self.defaults.merging(config){_,new in new}
        for (key,lo,hi) in [("topK",Int64(0),Int64(512)),("seed",0,Int64.max),("maxNewTokens",1,1024),("maxPrompts",1,128),("maxTotalTokens",1,8192)] {
            guard case .integer(let n)=c[key],(lo...hi).contains(n) else {throw DataFlowContractError("\(key) must be a whole number within \(lo)–\(hi).")}
        }
        for (key,hi) in [("temperature",100.0),("topP",1.0)] {
            let n:Double
            switch c[key] {case .number(let x):n=x;case .integer(let x):n=Double(x);default:throw DataFlowContractError("\(key) must be a number.")}
            guard n.isFinite,n>0,n<=hi else {throw DataFlowContractError("\(key) must be positive and at most \(hi).")}
        }
        for (key,choices) in [("method",["greedy","sample"]),("longPrompt",["refuse","latest"]),("device",["cpu","mps"]),("promptMode",["one","table"])] {
            guard case .string(let x)=c[key],choices.contains(x) else {throw DataFlowContractError("Choose a supported \(key).")}
        }
        for key in ["allowUnknown","stopAtEOS"] {guard case .bool=c[key] else {throw DataFlowContractError("\(key) must be Boolean.")}}
        for key in ["prompt","promptColumn","textColumn","idsColumn","countColumn","stopColumn"] {guard case .string(let x)=c[key],!x.contains("\0"),x.utf8.count <= (key == "prompt" ? 1_048_576:200) else {throw DataFlowContractError("\(key) exceeds its text allowance.")}}
        let names=["textColumn","idsColumn","countColumn","stopColumn"].compactMap{c[$0]?.stringValue}
        guard names.allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty}),Set(names).count==4 else {throw DataFlowContractError("Give each output column a distinct nonempty name.")}
        values=c
    }
    public var tableMode:Bool {values["promptMode"] == .string("table")}
    public var workerSettings:[String:JSONValue] {values.filter{!["promptMode","prompt","promptColumn","textColumn","idsColumn","countColumn","stopColumn"].contains($0.key)}}
}
