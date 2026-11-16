import Foundation

/// Configuration contains no captured file authority or model weights.
public struct GameSessionConfiguration:Equatable {
    public static let typeID="gameSession"
    public static let lessonID="game-episodes"
    public static let defaults:[String:JSONValue]=["version":1,"episodes":8,"seats":.string("alternate"),"aSeat":0,"seed":4100,"maxMoves":128,"maximumTotalMoves":32768,"episodeSeconds":5,"sessionSeconds":60,"captureDecisions":.bool(false),"device":.string("cpu"),"policies":.object([:])]
    public static var configurationSchema:JSONValue {
        func integer(_ lo:Int,_ hi:Int)->JSONValue{.object(["type":.string("integer"),"minimum":.integer(Int64(lo)),"maximum":.integer(Int64(hi))])}
        func choice(_ values:[String])->JSONValue{.object(["type":.string("string"),"enum":.array(values.map(JSONValue.string))])}
        let policy:JSONValue = .object(["type":.string("object"),"additionalProperties":.bool(false),"properties":.object(["source":choice(["canvas","saved","random","sameA"]),"initializationSeed":integer(0,2147483647),"inputName":.object(["type":.string("string"),"maxLength":128]),"selection":choice(["greedy","sampling"]),"temperature":.object(["type":.string("number"),"minimum":0.01,"maximum":100])])])
        return .object(["type":.string("object"),"additionalProperties":.bool(false),"properties":.object(["version":integer(1,1),"episodes":integer(1,256),"seats":choice(["fixed","alternate"]),"aSeat":integer(0,1),"seed":integer(0,2147483647),"maxMoves":integer(1,4096),"maximumTotalMoves":integer(1,32768),"episodeSeconds":.object(["type":.string("number"),"exclusiveMinimum":0,"maximum":3600]),"sessionSeconds":.object(["type":.string("number"),"exclusiveMinimum":0,"maximum":3600]),"device":choice(["cpu","mps"]),"captureDecisions":.object(["type":.string("boolean")]),"policies":.object(["type":.string("object"),"additionalProperties":.bool(false),"properties":.object(["A":policy,"B":policy])])]),"description":.string("Frozen-policy episodes, not training. Alternate seats requires even episodes. sameA is B-only and requires a model in A. Exactly one numeric float32 [1,N] input and [1,A] action logits. Saved policies require private verified canvas checkpoints. Source changes preserve incompatible wires until explicitly disconnected.")])
    }
    public let values:[String:JSONValue]
    public init(_ raw:[String:JSONValue])throws {
        func fail(_ message:String)throws->Never {throw DataFlowContractError(message)}
        guard Set(raw.keys).isSubset(of:Set(Self.defaults.keys)) else {try fail("Unknown Game session setting.")}
        var c=Self.defaults.merging(raw){_,new in new}
        for (key,lo,hi) in [("version",Int64(1),Int64(1)),("episodes",1,256),("aSeat",0,1),("seed",0,2147483647),("maxMoves",1,4096),("maximumTotalMoves",1,32768)] {
            guard case .integer(let v)=c[key],(lo...hi).contains(v) else {try fail("\(key) must be a whole number within \(lo)…\(hi).")}
        }
        guard ["fixed","alternate"].contains(c["seats"]?.stringValue ?? ""),c["seats"] != .string("alternate") || Int(c["episodes"]?.doubleValue ?? 1)%2==0 else {try fail("Alternate seats requires an even episode count.")}
        for key in ["episodeSeconds","sessionSeconds"] {guard let v=c[key]?.doubleValue,v.isFinite,v>0,v<=3600 else {try fail("Choose a positive \(key), at most 3600 seconds.")}}
        guard case .bool=c["captureDecisions"],["cpu","mps"].contains(c["device"]?.stringValue ?? "") else {try fail("Choose a supported device and capture setting.")}
        guard case .object(let supplied)=c["policies"],Set(supplied.keys).isSubset(of:["A","B"]) else {try fail("Define only Policy A and B.")}
        var policies:[String:JSONValue]=[:]
        for role in ["A","B"] {
            var p:[String:JSONValue]=["source":.string("canvas"),"initializationSeed":.integer(role=="A" ? 17:18),"selection":.string("greedy"),"temperature":1,"inputName":.string("input")]
            if let value=supplied[role] {guard case .object(let fields)=value,Set(fields.keys).isSubset(of:Set(p.keys)) else {try fail("Unknown policy fields.")};p.merge(fields){_,new in new}}
            guard let source=p["source"]?.stringValue,["canvas","saved","random","sameA"].contains(source),role=="B" || source != "sameA" else {try fail("Choose a supported policy source.")}
            guard case .integer(let seed)=p["initializationSeed"],(0...2147483647).contains(seed),["greedy","sampling"].contains(p["selection"]?.stringValue ?? ""),let temp=p["temperature"]?.doubleValue,temp.isFinite,(0.01...100).contains(temp),let name=p["inputName"]?.stringValue,!name.isEmpty,name.utf8.count<=128 else {try fail("Choose a valid policy seed, selection, temperature and input name.")}
            policies[role] = .object(p)
        }
        if case .object(let a)=policies["A"],case .object(let b)=policies["B"],b["source"] == .string("sameA"),a["source"] == .string("random") {try fail("Same snapshot requires a model in Policy A.")}
        c["policies"] = .object(policies);values=c
    }
    public func policy(_ role:String)->[String:JSONValue] {guard case .object(let p)=values["policies"],case .object(let v)=p[role] else{return [:]};return v}
    public func source(_ role:String)->String {policy(role)["source"]?.stringValue ?? "canvas"}
    public var inputPorts:[WorkflowResolvedPort] {
        var result:[WorkflowResolvedPort]=[.init(id:"definition",kind:.table,required:true,maximumConnections:1)]
        for role in ["A","B"] where ["canvas","saved"].contains(source(role)) {result.append(.init(id:"policy"+role,kind:source(role)=="canvas" ? .modelDefinition:.frozenModel,required:true,maximumConnections:1))}
        return result
    }
}
