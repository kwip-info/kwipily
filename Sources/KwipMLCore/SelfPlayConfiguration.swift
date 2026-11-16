import Foundation

/// Self-play owns its internal feedback loop; no cyclic canvas connections or hidden supervision.
public struct SelfPlayConfiguration:Equatable {
    public static let defaults:[String:JSONValue]=["iterations":8,"games":8,"updates":16,"batch":64,"maxMoves":128,"seed":4100,"initializationSeed":17,"learningRate":0.001,"optimizer":.string("adam"),"scheduler":.string("constant"),"finalLearningRate":0.0001,"momentum":0,"beta1":0.9,"beta2":0.999,"epsilon":1e-8,"weightDecay":0,"policyWeight":1,"valueWeight":1,"gradientClip":1,"temperature":1,"temperatureMoves":12,"replayPositions":25000,"replayBytes":134217728,"replayPolicy":.string("stop"),"checkpoints":3,"evaluationPairs":4,"evaluationEvery":1,"seconds":600,"device":.string("cpu"),"search":.object(["simulations":32,"exploration":1.5,"maximumDepth":128,"maximumNodes":2048,"seconds":3,"noiseAlpha":0.3,"noiseMix":0.25])]
    public static let baseTablePorts=["iterations","episodes","moves","evaluation"]
    public static let allTablePorts=baseTablePorts+["campaign","opponents","performance"]
    public static func tablePorts(_ values:[String:JSONValue])->[String] {if case .object(let c)=values["campaign"],c["enabled"] == .bool(true){return allTablePorts};return baseTablePorts}
    public static let advancedDefaults:[String:JSONValue]=[
        "campaign":.object(["enabled":false,"sliceIterations":1,"maximumSeconds":28800,"maximumBytes":1073741824]),
        "compute":.object(["activeGames":1,"inferenceDevice":.string("cpu"),"batchSize":16,"batchWait":0.002,"cacheBytes":16777216]),
        "evaluation":.object(["opponents":.array([.string("initial")]),"openings":.array([.array([])]),"seed":8301,"checkpoints":.array([])])]
    public static var configurationSchema:JSONValue {
        func schema(_ value:JSONValue)->JSONValue {
            if case .object(let fields)=value {return .object(["type":.string("object"),"additionalProperties":false,"properties":.object(fields.mapValues(schema))])}
            if case .bool=value {return .object(["type":.string("boolean"),"default":value])}
            if case .array=value {return .object(["type":.string("array"),"default":value])}
            if case .string(let text)=value {return .object(["type":.string("string"),"default":.string(text)])}
            if case .integer=value {return .object(["type":.string("integer"),"default":value])}
            return .object(["type":.string("number"),"default":value])
        }
        var advertised=defaults.merging(advancedDefaults){_,v in v}
        if case .object(var search)=advertised["search"]{search["reuseTree"]=false;advertised["search"] = .object(search)}
        return .object(["type":.string("object"),"additionalProperties":false,"required":.array([.string("selfPlay")]),"properties":.object(["selfPlay":schema(.object(advertised))]),"description":.string("Train Self-play replaces supervised configuration. CPU reference / qualified Apple GPU, explicit gamePolicyValue output task v1, typed Game definition input; model plus iterations/episodes/moves/evaluation outputs. Runtime validators enforce finite search/resource/replay bounds and exact resume pins.")])
    }
    public let values:[String:JSONValue]
    public init(_ raw:[String:JSONValue])throws {
        guard Set(raw.keys)==["selfPlay"],case .object(let fields)=raw["selfPlay"],Set(fields.keys).isSubset(of:Set(Self.defaults.keys).union(Self.advancedDefaults.keys)) else {throw DataFlowContractError("Self-play requires its own versioned Train configuration; supervised mappings are separate.")}
        var c=Self.defaults.merging(fields){_,new in new}
        for (key,defaultValue) in Self.advancedDefaults where fields[key] != nil {
            guard case .object(let object)=fields[key],case .object(let template)=defaultValue,Set(object.keys).isSubset(of:Set(template.keys))else{throw DataFlowContractError("Invalid Self-play \(key) settings.")}
            c[key] = .object(template.merging(object){_,v in v})
        }
        let campaign={ ()->[String:JSONValue] in if case .object(let value)=c["campaign"] {return value};return [:] }()
        let enabled=campaign["enabled"] == .bool(true)
        let slice=enabled ? Int(campaign["sliceIterations"]?.integerValue ?? 1):Int(c["iterations"]?.integerValue ?? 0)
        func fail(_ key:String)throws->Never {throw DataFlowContractError("Choose a supported bounded Self-play \(key).")}
        let integerBounds:[(String,Int,Int)]=[("iterations",1,enabled ? 1000000:99),("games",2,256),("updates",1,4096),("batch",1,1024),("maxMoves",1,4096),("seed",0,2147483647),("initializationSeed",0,2147483647),("temperatureMoves",0,4096),("evaluationPairs",1,128),("evaluationEvery",1,99),("replayPositions",1,100000),("replayBytes",65536,enabled ? 2147483648:536870912),("checkpoints",1,10)]
        for (key,lo,hi) in integerBounds {
            guard case .integer(let v)=c[key],(Int64(lo)...Int64(hi)).contains(v) else {try fail(key)}
        }
        for (key,lo,hi) in [("learningRate",1e-7,1.0),("weightDecay",0,1),("finalLearningRate",1e-7,1),("momentum",0,0.999),("beta1",0,0.9999),("beta2",0,0.999999),("epsilon",1e-12,1),("policyWeight",0,100),("valueWeight",0,100),("gradientClip",0.001,1000),("temperature",0.01,10),("seconds",0.01,3600)] {
            guard let v=c[key]?.doubleValue,v.isFinite,(lo...hi).contains(v) else {try fail(key)}
        }
        guard slice*(c["games"]?.integerValue ?? 0)*(c["maxMoves"]?.integerValue ?? 0)<=32768 else {try fail("planned moves (maximum 32768 per run)")}
        guard slice*(c["updates"]?.integerValue ?? 0)<=4096 else {try fail("update retention (maximum 4096 measurements)")}
        guard (c["games"]?.integerValue ?? 1)%2==0,(c["policyWeight"]?.doubleValue ?? 0)+(c["valueWeight"]?.doubleValue ?? 0)>0,["constant","linear","cosine"].contains(c["scheduler"]?.stringValue ?? ""),["adam","adamw","sgd"].contains(c["optimizer"]?.stringValue ?? ""),["stop","rolling"].contains(c["replayPolicy"]?.stringValue ?? ""),["cpu","mps","auto"].contains(c["device"]?.stringValue ?? "") else {try fail("task settings (exact continuation is currently CPU-qualified)")}
        guard case .object(let search)=c["search"],case .object(let defaults)=Self.defaults["search"],Set(search.keys).isSubset(of:Set(defaults.keys).union(["reuseTree"])) else {try fail("search")}
        let s=defaults.merging(search){_,new in new}
        for (key,lo,hi) in [("simulations",1,4096),("maximumDepth",1,512),("maximumNodes",2,10000)] {guard case .integer(let v)=s[key],(Int64(lo)...Int64(hi)).contains(v) else {try fail(key)}}
        for (key,lo,hi) in [("exploration",0.0,100.0),("seconds",0.001,60),("noiseAlpha",0.001,10),("noiseMix",0,1)] {guard let v=s[key]?.doubleValue,v.isFinite,(lo...hi).contains(v) else {try fail(key)}}
        if let value=s["reuseTree"],case .bool=value {}else if s["reuseTree"] != nil {try fail("tree reuse")}
        func object(_ key:String)->[String:JSONValue] {if case .object(let value)=c[key]{return value};return [:]}
        func bounded(_ object:[String:JSONValue],_ key:String,_ low:Int64,_ high:Int64)throws {guard let n=object[key]?.integerValue,(low...high).contains(Int64(n))else{try fail(key)}}
        if c["campaign"] != nil {
            guard case .bool=campaign["enabled"]else{try fail("campaign mode")}
            try bounded(campaign,"sliceIterations",1,99);try bounded(campaign,"maximumSeconds",1,604800);try bounded(campaign,"maximumBytes",67108864,8589934592)
        }
        if c["compute"] != nil {
            let x=object("compute");try bounded(x,"activeGames",1,8);try bounded(x,"batchSize",1,256);try bounded(x,"cacheBytes",0,268435456)
            guard ["cpu","mps","auto"].contains(x["inferenceDevice"]?.stringValue ?? ""),let w=x["batchWait"]?.doubleValue,w.isFinite,(0...0.05).contains(w)else{try fail("compute")}
        }
        if c["evaluation"] != nil {
            let e=object("evaluation");try bounded(e,"seed",0,2147483647)
            guard case .array(let opponents)=e["opponents"],(1...8).contains(opponents.count),case .array(let openings)=e["openings"],(1...32).contains(openings.count),case .array(let checkpoints)=e["checkpoints"],checkpoints.count<=5 else{try fail("evaluation suite")}
            for name in opponents {guard let n=name.stringValue,["initial","random","uniform-search"].contains(n) || n.hasPrefix("checkpoint:") && UUID(uuidString:String(n.dropFirst(11))) != nil else{try fail("opponent")}}
            for opening in openings {guard case .array(let actions)=opening,actions.count<=128,actions.allSatisfy({if case .object=$0{return true};return false})else{try fail("openings")}}
        }
        var normalized=c;normalized["search"] = .object(s);values=normalized
    }
    public static func isSelfPlay(_ node:WorkflowNode)->Bool {if case .train(let c)=node.payload{return c["selfPlay"] != nil};return false}
}

public struct SelfPlayPlan {
    public let model:ModelExecutionSnapshot
    public let definition:[String:JSONValue]
    public let configuration:SelfPlayConfiguration
    public static func compile(_ document:WorkflowDocument,address:WorkflowNodeAddress,modelRegistry:ComponentRegistry,dataRegistry:DataFlowOperatorRegistry)throws->Self {
        try WorkflowCodec.validate(document)
        guard document.drafts.isEmpty,let node=document.nodes.first(where:{$0.address==address}),case .train(let c)=node.payload else {throw DataFlowContractError("Select Train Self-play with no unresolved drafts.")}
        let config=try SelfPlayConfiguration(c)
        _=try WorkflowCompiler.dependencies(document,terminal:address,modelRegistry:modelRegistry,dataRegistry:dataRegistry)
        func source(_ port:String)throws->WorkflowNode {
            let links=document.edges.filter{$0.to.address==address && $0.to.port==port}
            guard links.count==1,let edge=links.first,let n=document.nodes.first(where:{$0.address==edge.from.address}) else {throw DataFlowContractError("Connect one \(port) input.")};return n
        }
        let rules=try source("definition")
        guard case .dataOperator(GameRuleConfiguration.operatorID,let fields)=rules.payload,case .object(let definition)=fields["definition"] else {throw DataFlowContractError("Self-play requires the typed Game definition resource.")}
        try GameRuleConfiguration.validate(fields)
        let model=try WorkflowCompiler.modelSnapshot(document,definition:source("model").address,modelRegistry:modelRegistry,dataRegistry:dataRegistry,context:.policyValue)
        guard model.graph.nodes.contains(where:{$0.type=="output" && $0.config["task"] == .string("gamePolicyValue")})else{throw DataFlowContractError("Select the versioned Game policy/value Model Output task.")}
        return .init(model:model,definition:definition,configuration:config)
    }
}
