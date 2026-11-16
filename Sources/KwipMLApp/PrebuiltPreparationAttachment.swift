import Foundation
import KwipMLCore

/// Bounded inert attachment admission. Exact bytes are pinned; the shared worker
/// validates learned rules and their canonical digest again before any application.
struct PrebuiltPreparationAttachment {
    static let maximumBytes = 20_000_000
    let contentDigest:String
    let modelSHA256:String
    let inputSchema:[JSONValue]
    let outputSchema:[JSONValue]
    let parameters:[[String:JSONValue]]
    var summary:[String:JSONValue] {
        let rules=parameters.flatMap { parameter -> [JSONValue] in
            guard case .array(let entries)=parameter["rules"] else{return []}
            return entries.compactMap { entry in
                guard case .object(let rule)=entry else{return nil}
                return .object(rule.filter {["column","missing","encoding","scaling","unknownPolicy"].contains($0.key)})
            }
        }
        return ["contentDigest":.string(contentDigest),"modelSHA256":.string(modelSHA256),"ruleCount":.integer(Int64(parameters.count)),"inputSchema":.array(inputSchema),"rules":.array(rules)]
    }
    static func decode(_ bytes:Data,modelBytes:Data)throws->Self {
        func require(_ condition:Bool)throws {guard condition else {throw WorkflowArchiveError("Invalid model-bound learned preparation attachment.")}}
        try require(!bytes.isEmpty && bytes.count<=maximumBytes)
        let envelope=try JSONDecoder().decode([String:JSONValue].self,from:bytes)
        try require(Set(envelope.keys)==["schemaVersion","kind","modelSHA256","featureColumns","inputSchema","outputSchema","parameters","contentDigest"])
        guard envelope["schemaVersion"] == 1,envelope["kind"] == .string("kwipml.prebuilt-preparation.v1"),let modelPin=envelope["modelSHA256"]?.stringValue,modelPin==WorkflowArchiveCodec.hash(modelBytes),let digest=envelope["contentDigest"]?.stringValue,WorkflowArchiveCodec.isHash(digest),case .array(let parameters)=envelope["parameters"],(1...16).contains(parameters.count),case .array(let input)=envelope["inputSchema"],case .array(let output)=envelope["outputSchema"] else {throw WorkflowArchiveError("Learned preparation is missing or belongs to a different model.")}
        let state=try JSONDecoder().decode([String:JSONValue].self,from:modelBytes)
        try require(state["stateFormat"] == .string("kwipml.prebuilt-model.v1") && state["featureColumns"]==envelope["featureColumns"])
        func schema(_ raw:[JSONValue])throws {
            try require((1...128).contains(raw.count))
            var names=Set<String>()
            for item in raw {
                guard case .object(let c)=item,let name=c["name"]?.stringValue,!name.isEmpty,name.count<=200,!name.contains("\0"),let dtype=c["dtype"]?.stringValue,["string","bool","int64","float64","image","audio","tensor","int64Sequence","date","datetime","time"].contains(dtype),case .bool=c["nullable"] else {throw WorkflowArchiveError("Invalid learned preparation schema.")}
                try require(names.insert(name).inserted && Set(c.keys).isSubset(of:["name","dtype","nullable","displayLabel","temporal"]))
            }
        }
        try schema(input);try schema(output)
        var previous=input,parsed:[[String:JSONValue]]=[],digests=Set<String>()
        for entry in parameters {
            guard case .object(let p)=entry,p["schemaVersion"]==1,p["kind"] == .string("portableFittedParameters"),p["operator"] == .string("kwip.data.preprocess"),let source=p["sourceContentDigest"]?.stringValue,WorkflowArchiveCodec.isHash(source),case .object(let config)=p["configuration"],case .array(let rules)=p["rules"],case .array(let next)=p["outputSchema"] else {throw WorkflowArchiveError("Unsupported learned preparation chain.")}
            try require(Set(p.keys)==["schemaVersion","kind","operator","columns","inputSchema","outputSchema","sourceContentDigest","configuration","rules"] && digests.insert(source).inserted && p["inputSchema"] == .array(previous))
            let validated=try PreprocessConfiguration(values:config)
            try require(validated.mode=="partitions" && rules.count==validated.columns.count && p["columns"] == .array(validated.columns.map{.string($0.column)}))
            for (index,entry) in rules.enumerated() {
                guard case .object(let rule)=entry,case .array(let vocabulary)=rule["vocabulary"],let mean=rule["mean"]?.doubleValue,mean.isFinite,let scale=rule["scale"]?.doubleValue,scale.isFinite,scale>0 else {throw WorkflowArchiveError("Invalid captured preprocessing values.")}
                try require(vocabulary.count<=validated.columns[index].maxCategories && vocabulary.allSatisfy{$0.stringValue != nil} && Set(vocabulary).count==vocabulary.count)
                guard case .array(let originals)=config["columns"],case .object(let original)=originals[index] else {throw WorkflowArchiveError("Invalid captured preprocessing configuration.")}
                try require(Set(rule.keys)==Set(original.keys).union(["fill","vocabulary","mean","scale"]) && original.allSatisfy{rule[$0.key]==$0.value})
            }
            try schema(next);previous=next;parsed.append(p)
        }
        try require(previous==output)
        return .init(contentDigest:digest,modelSHA256:modelPin,inputSchema:input,outputSchema:output,parameters:parsed)
    }
}
