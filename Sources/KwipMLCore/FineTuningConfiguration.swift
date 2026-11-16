import Foundation

/// Imported-model policy lives on Train. Values describe a task adapter, never
/// executable repository code or a second model-definition graph.
public struct FineTuningConfiguration:Equatable {
    public static let defaults:[String:JSONValue] = ["mode":.string("full"),"groups":.array([]),"targets":.array([]),"rank":8,"alpha":16,"dropout":0,"bias":.string("none"),"additionalModules":.array([]),"textColumn":.string("text"),"maximumTokens":128,"stride":127,"appendEOS":true,"accumulation":1]
    public static let expansionDefaults:[String:JSONValue] = ["exampleFormat":.string("text"),"promptColumn":.string("prompt"),"responseColumn":.string("response"),"separator":.string("\n"),"lossCoverage":.string("response"),"activationCheckpointing":false,"checkpointCount":0,"checkpointBytes":1_000_000_000,"startMode":.string("base"),"resumeRecord":.string(""),"resumeEpoch":0]
    public let values:[String:JSONValue]
    public init(_ supplied:[String:JSONValue])throws {
        if supplied["task"] == .string("typedDecisions") {values=try TypedDecisionConfiguration.validateTraining(supplied);return}
        guard Set(supplied.keys).isSubset(of:Set(Self.defaults.keys).union(Self.expansionDefaults.keys)) else {throw DataFlowContractError("Unknown imported-model training setting.")}
        var c=Self.defaults.merging(supplied){_,new in new}
        for (key,choices) in [("mode",["full","selected","lora"]),("bias",["none","lora_only","all"])] {
            guard let value=c[key]?.stringValue,choices.contains(value) else {throw DataFlowContractError("Choose a supported fine-tuning \(key).")}
        }
        for (key,range) in [("rank",1...1024),("maximumTokens",2...131072),("stride",1...131071),("accumulation",1...1024)] {
            guard case .integer(let n)=c[key],range.contains(Int(n)) else {throw DataFlowContractError("Fine-tuning \(key) is outside its supported range.")}
        }
        guard c["stride"]!.integerValue! < c["maximumTokens"]!.integerValue! else {throw DataFlowContractError("Stride must be smaller than context.")}
        for (key,range) in [("alpha",0.00001...1_000_000.0),("dropout",0.0...0.999999)] {
            guard let n=c[key]?.doubleValue,n.isFinite,range.contains(n) else {throw DataFlowContractError("Invalid LoRA \(key).")};c[key] = n.rounded() == n ? .integer(Int64(n)):.number(n)
        }
        for key in ["groups","targets","additionalModules"] {
            guard case .array(let a)=c[key],a.count<=512 else {throw DataFlowContractError("Choose known layer groups and LoRA targets.")}
            let names=a.compactMap(\.stringValue)
            guard names.count==a.count,Set(names).count==a.count,names.allSatisfy({!$0.isEmpty && $0.count<=256}) else {throw DataFlowContractError("Layer selections must be unique names.")}
        }
        guard case .bool=c["appendEOS"],let column=c["textColumn"]?.stringValue,!column.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,column.utf8.count<=200 else {throw DataFlowContractError("Choose one text column and an EOS policy.")}
        for key in ["exampleFormat","lossCoverage","startMode"] {
            if let value=c[key] {let options=key=="exampleFormat" ? ["text","promptResponse"]:key=="lossCoverage" ? ["response","all"]:["base","weights","resume"];guard let name=value.stringValue,options.contains(name) else{throw DataFlowContractError("Choose a supported fine-tuning \(key).")}}
        }
        for key in ["promptColumn","responseColumn"] {if let value=c[key] {guard let name=value.stringValue,!name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.utf8.count<=200 else{throw DataFlowContractError("Choose a bounded prompt/response column.")}}}
        if let value=c["separator"] {guard let text=value.stringValue,text.utf8.count<=256 else{throw DataFlowContractError("Use a separator within 256 UTF-8 bytes.")}}
        if let value=c["activationCheckpointing"] {guard case .bool=value else{throw DataFlowContractError("Activation checkpointing needs a Boolean.")}}
        for (key,maximum) in [("checkpointCount",Int64(10)),("checkpointBytes",8_000_000_000),("resumeEpoch",10_000)] {if let value=c[key] {guard case .integer(let number)=value,number >= (key=="checkpointBytes" ? 1:0),number<=maximum else{throw DataFlowContractError("Invalid \(key) allowance.")}}}
        if let value=c["resumeRecord"] {guard let name=value.stringValue,name.isEmpty || UUID(uuidString:name)?.uuidString==name else{throw DataFlowContractError("Choose a recorded same-project training boundary.")}}
        if c["exampleFormat"] == .string("promptResponse") {guard (c["promptColumn"] ?? Self.expansionDefaults["promptColumn"]) != (c["responseColumn"] ?? Self.expansionDefaults["responseColumn"]) else{throw DataFlowContractError("Prompt and response need different columns.")}}
        values=c
    }
    public static var configurationSchema:JSONValue {
        var properties:[String:JSONValue]=[:]
        for (key,value) in defaults.merging(expansionDefaults,uniquingKeysWith:{old,_ in old}) {
            let type:String
            switch value {case .string:type="string";case .integer:type="integer";case .number:type="number";case .bool:type="boolean";case .array:type="array";default:type="object"}
            properties[key] = .object(["type":.string(type),"default":value])
        }
        for key in ["alpha","dropout"] { properties[key] = .object(["type":.string("number"),"default":defaults[key]!]) }
        properties["mode"] = .object(["enum":.array([.string("full"),.string("selected"),.string("lora")])])
        properties["bias"] = .object(["enum":.array([.string("none"),.string("lora_only"),.string("all")])])
        for (key,options) in [("exampleFormat",["text","promptResponse"]),("lossCoverage",["response","all"]),("startMode",["base","weights","resume"])] {properties[key] = .object(["type":.string("string"),"enum":.array(options.map(JSONValue.string)),"default":expansionDefaults[key]!])}
        for (key,minimum,maximum) in [("checkpointCount",Int64(0),Int64(10)),("checkpointBytes",1,8_000_000_000),("resumeEpoch",0,10_000)] {properties[key] = .object(["type":.string("integer"),"minimum":.integer(minimum),"maximum":.integer(maximum),"default":expansionDefaults[key]!])}
        for key in ["groups","targets","additionalModules"]{properties[key] = .object(["type":.string("array"),"maxItems":512,"uniqueItems":true,"items":.object(["type":.string("string"),"minLength":1,"maxLength":256]),"default":.array([])])}
        return .object(["oneOf":.array([.object(["type":.string("object"),"additionalProperties":false,"properties":.object(properties)]),TypedDecisionConfiguration.trainingSchema])])
    }
    public static func decode(_ train:[String:JSONValue])throws->Self? {
        guard let value=train["fineTuning"] else {return nil}
        guard case .object(let c)=value else {throw DataFlowContractError("Fine-tuning settings must be an object.")};return try .init(c)
    }
    public static func isExternal(_ node:WorkflowNode)->Bool {if case .train(let c)=node.payload {return c["fineTuning"] != nil};return false}
}

public struct WorkflowFineTuningPlan:Equatable {
    public let train:WorkflowNodeAddress
    public let model:WorkflowNodeAddress
    public let training:WorkflowEndpoint
    public let validation:WorkflowEndpoint
    public let configuration:FineTuningConfiguration
    public func preparationDocument(_ document:WorkflowDocument)throws->WorkflowDocument {
        var needed:Set<WorkflowNodeAddress>=[],pending=[training.address,validation.address]
        while let address=pending.popLast() {
            guard needed.insert(address).inserted else {continue}
            guard address.domain == .data else {throw workflowFailure(.configuration,"Prepare text from ordinary data sources before imported-model training.",train)}
            pending += document.edges.filter{$0.to.address==address}.map(\.from.address)
        }
        var result=document
        result.nodes.removeAll{!needed.contains($0.address)};result.edges.removeAll{!needed.contains($0.to.address) || !needed.contains($0.from.address)}
        result.definitions=[];result.drafts=[];result.modelInputs=nil;result.namedModelInputs=nil;result.modelSourceVersions=nil;result.guidedBuild=nil
        return result
    }
    public static func compile(_ document:WorkflowDocument,train:WorkflowNodeAddress,dataRegistry:DataFlowOperatorRegistry)throws->Self {
        try WorkflowCodec.validate(document)
        guard document.schemaVersion>=21,document.drafts.isEmpty,let node=document.nodes.first(where:{$0.address==train}),case .train(let c)=node.payload,let configuration=try FineTuningConfiguration.decode(c) else {throw workflowFailure(.configuration,"Choose an imported-model Train and apply its pending edits.",train)}
        let edges=document.edges.filter{$0.to.address==train}
        guard Set(edges.map(\.to.port))==["model","training","validation"],edges.count==3,let model=edges.first(where:{$0.to.port=="model"}),model.kind == .frozenModel,model.from.port=="model",let provider=document.nodes.first(where:{$0.address==model.from.address}),case .frozenModel(let source)=provider.payload,try LoadModelConfiguration.decode(source).source != .savedRun else {throw workflowFailure(.requiredInput,"Connect a verified external Load model and both text table partitions.",train)}
        func table(_ role:String)throws->WorkflowEndpoint {
            guard let edge=edges.first(where:{$0.to.port==role}),edge.kind == .table,edge.from.address.domain == .data,let source=document.nodes.first(where:{$0.address==edge.from.address}),case .dataOperator(let operatorID,let config)=source.payload,let definition=dataRegistry.operatorDefinition(id:operatorID)?.configured(config),definition.outputs.contains(where:{$0.id==edge.from.port && ["table","dataset"].contains($0.kind)}) else {throw workflowFailure(.requiredInput,"Connect a prepared \(role) table.",train,role)}
            return edge.from
        }
        let training=try table("training"),validation=try table("validation")
        guard training != validation else {throw workflowFailure(.lineage,"Training and Validation require disjoint observations.",train)}
        let value=Self(train:train,model:model.from.address,training:training,validation:validation,configuration:configuration)
        _=try WorkflowDataPreparationPlan.compile(value.preparationDocument(document),registry:dataRegistry)
        return value
    }
}
