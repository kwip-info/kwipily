import Foundation

/// Public structural measurements after the checker assays. Contains no fixture values or learned arrays.
public struct ModuleCompositionLearningMetadata: Codable, Equatable {
    public struct Owner: Codable, Equatable {
        public var node: String
        public var parameter: String
        public var name: String
        public var shape: [Int]
        public var dtype: String
        public var requiresGrad: Bool
        public init(node:String,parameter:String,name:String,shape:[Int],dtype:String="float32",requiresGrad:Bool) {
            self.node=node;self.parameter=parameter;self.name=name;self.shape=shape;self.dtype=dtype;self.requiresGrad=requiresGrad
        }
    }
    public struct Container: Codable, Equatable {
        public var path:String
        public var kind:CompositionPolicy.Kind
        public var keys:[String]
        public var matches:Bool
        public init(path:String,kind:CompositionPolicy.Kind,keys:[String],matches:Bool) {self.path=path;self.kind=kind;self.keys=keys;self.matches=matches}
    }
    public struct Invocation: Codable, Equatable {
        public var phase:String
        public var node:String
        public init(phase:String,node:String) {self.phase=phase;self.node=node}
        public init(from decoder:Decoder)throws {var c=try decoder.unkeyedContainer();phase=try c.decode(String.self);node=try c.decode(String.self);guard c.isAtEnd else {throw LearningContractError("Invocation requires phase and node only.")}}
        public func encode(to encoder:Encoder)throws {var c=encoder.unkeyedContainer();try c.encode(phase);try c.encode(node)}
    }
    public static let maximumBytes=131072
    public var schemaVersion:Int
    public var nativeOwners:[Owner]
    public var nativeContainers:[Container]
    public var invocations:[Invocation]
    public var savedStateKeys:[String]
    public var independentStorage:Bool
    public init(schemaVersion:Int=1,nativeOwners:[Owner],nativeContainers:[Container],invocations:[Invocation],savedStateKeys:[String],independentStorage:Bool) {
        self.schemaVersion=schemaVersion;self.nativeOwners=nativeOwners;self.nativeContainers=nativeContainers;self.invocations=invocations;self.savedStateKeys=savedStateKeys;self.independentStorage=independentStorage
    }
    private static func require(_ valid:Bool,_ message:String)throws {
        guard valid else {throw LearningContractError(message)}
    }
    private static func object(_ value:JSONValue,_ keys:Set<String>)throws->[String:JSONValue] {
        guard case .object(let fields)=value,Set(fields.keys)==keys else {throw LearningContractError("Composition measurements contain unknown or missing fields.")}
        return fields
    }
    private static func array(_ value:JSONValue?,maximum:Int)throws->[JSONValue] {
        guard case .array(let values)=value,values.count<=maximum else {throw LearningContractError("Composition measurement collection exceeds its bounded schema.")}
        return values
    }
    private static func text(_ value:JSONValue?,maximum:Int=128)throws->String {
        guard case .string(let text)=value,!text.isEmpty,text.utf8.count<=maximum,!text.contains("\0") else {throw LearningContractError("Composition measurement identity is missing or too long.")}
        return text
    }
    private static func boolean(_ value:JSONValue?)throws->Bool {
        guard case .bool(let flag)=value else {throw LearningContractError("Composition measurement flags must be actual booleans.")}
        return flag
    }
    public init(from decoder:Decoder)throws {
        let fields=try Self.object(JSONValue(from:decoder),["schemaVersion","nativeOwners","nativeContainers","invocations","savedStateKeys","independentStorage"])
        guard case .integer(1)=fields["schemaVersion"] else {throw LearningContractError("Composition measurements require schema version 1.")}
        let owners=try Self.array(fields["nativeOwners"],maximum:256).map { value -> Owner in
            let f=try Self.object(value,["node","parameter","name","shape","dtype","requiresGrad"])
            let shape=try Self.array(f["shape"],maximum:7).map { value -> Int in
                guard case .integer(let n)=value,(1...4096).contains(n) else {throw LearningContractError("Native parameter dimensions must be bounded integers.")}
                return Int(n)
            }
            return try Owner(node:Self.text(f["node"]),parameter:Self.text(f["parameter"]),name:Self.text(f["name"],maximum:4096),shape:shape,dtype:Self.text(f["dtype"]),requiresGrad:Self.boolean(f["requiresGrad"]))
        }
        let containers=try Self.array(fields["nativeContainers"],maximum:129).map { value -> Container in
            let f=try Self.object(value,["path","kind","keys","matches"])
            guard let kind=CompositionPolicy.Kind(rawValue:try Self.text(f["kind"])) else {throw LearningContractError("Unknown native container kind.")}
            return try Container(path:Self.text(f["path"],maximum:4096),kind:kind,keys:Self.array(f["keys"],maximum:128).map {try Self.text($0)},matches:Self.boolean(f["matches"]))
        }
        let invocations=try Self.array(fields["invocations"],maximum:256).map { value -> Invocation in
            let pair=try Self.array(value,maximum:2)
            try Self.require(pair.count==2,"Invocation requires exactly phase and node.")
            return try Invocation(phase:Self.text(pair[0]),node:Self.text(pair[1]))
        }
        self.init(nativeOwners:owners,nativeContainers:containers,invocations:invocations,savedStateKeys:try Self.array(fields["savedStateKeys"],maximum:512).map {try Self.text($0,maximum:4096)},independentStorage:try Self.boolean(fields["independentStorage"]))
        try validateBounds()
    }
    public static func decode(_ data:Data)throws->Self {
        try require(data.count<=maximumBytes,"Keep composition measurements within 131072 UTF8 bytes.")
        return try JSONDecoder().decode(Self.self,from:data)
    }
    private func validateBounds()throws {
        try Self.require(schemaVersion==1 && nativeOwners.count<=256 && (1...129).contains(nativeContainers.count) && invocations.count<=256 && savedStateKeys.count<=512,"Composition measurements exceed their schema or collection bounds.")
        func validText(_ value:String,_ limit:Int)->Bool {!value.isEmpty && value.utf8.count<=limit && !value.contains("\0")}
        for owner in nativeOwners {
            try Self.require(validText(owner.node,128) && validText(owner.parameter,128) && validText(owner.name,4096) && owner.dtype=="float32" && (1...7).contains(owner.shape.count),"Use bounded float32 native parameter records.")
            var product=1
            for n in owner.shape {try Self.require((1...4096).contains(n) && product<=200000/n,"Native parameter shape exceeds the element limit.");product*=n}
        }
        try Self.require(Set(nativeOwners.map(\.name)).count==nativeOwners.count, "Native parameter owners cannot alias or repeat.")
        for container in nativeContainers {
            try Self.require(validText(container.path,4096) && (1...128).contains(container.keys.count) && Set(container.keys).count==container.keys.count && container.keys.allSatisfy {validText($0,128)},"Native child keys must be bounded, complete and unique.")
        }
        try Self.require(Set(nativeContainers.map(\.path)).count==nativeContainers.count,"Native container paths must be unique.")
        try Self.require(savedStateKeys.allSatisfy {validText($0,4096)} && Set(savedStateKeys).count==savedStateKeys.count,"Saved state keys must be bounded and unique.")
        try Self.require(invocations.allSatisfy { ["started","completed"].contains($0.phase) && validText($0.node,128)},"Invocation records require supported phases and bounded node identities.")
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys,.withoutEscapingSlashes]
        try Self.require(try encoder.encode(self).count<=Self.maximumBytes,"Keep composition measurements within 131072 UTF8 bytes.")
    }
    /// Validates measurements after checks, including the checker assay's temporary left-weight freeze.
    /// False storage/match measurements remain evidence of failure, never an authorization assertion.
    public func validate(source:CompositeProjectDocument,task:ModuleCompositionLearningTask,registry:ComponentRegistry)throws {
        try validateBounds()
        _ = try ModuleCompositionLearningSubmission.validatedSource(source,task:task,registry:registry,complete:true)
        let expansion=try CompositeExpander.expand(source,registry:registry,context:.explicitObjective)
        guard let plan=expansion.graph.compositionPlan else {throw LearningContractError("Submitted source has no native registration plan.")}
        try plan.validate(graph:expansion.graph,registry:registry)
        let containers=Dictionary(uniqueKeysWithValues:plan.containers.map {($0.id,$0)})
        let nodes=Dictionary(uniqueKeysWithValues:expansion.graph.nodes.map {($0.id,$0)})
        var expectedContainers:[Container]=[],expectedOwners:[Owner]=[]
        func visit(_ id:String,_ path:String)throws {
            guard let container=containers[id] else {throw LearningContractError("Submitted container is missing.")}
            expectedContainers.append(.init(path:path,kind:container.kind,keys:container.children.map(\.key),matches:true))
            for child in container.children {
                let childPath=path+(container.kind == .graph ? ".blocks.":".")+child.key
                if child.kind == .container {try visit(child.id,childPath);continue}
                guard let node=nodes[child.id] else {throw LearningContractError("Submitted native child is missing.")}
                guard node.type == "linear" else {
                    try Self.require(["add","concat","dropout"].contains(node.type),"Unsupported parameter or buffer geometry in this composition lesson.")
                    continue
                }
                let config=try GraphMigration.effectiveConfiguration(of:node,schemaVersion:expansion.graph.schemaVersion,registry:registry)
                guard let input=config["in_features"]?.integerValue,let output=config["out_features"]?.integerValue,let origin=expansion.nodeOrigins[node.id] else {throw LearningContractError("Native affine parameter geometry is unresolved.")}
                var parameters=[("weight",[output,input])]
                if config["bias"]?.boolValue == true {parameters.append(("bias",[output]))}
                for (parameter,shape) in parameters {
                    let fixed=["negate","difference"].contains(origin.localNodeID) || (task == .buildReuse && origin.instancePath == ["block","left"] && origin.localNodeID == "a" && parameter == "weight")
                    expectedOwners.append(.init(node:node.id,parameter:parameter,name:childPath+".operation."+parameter,shape:shape,requiresGrad:!fixed))
                }
            }
        }
        try visit(plan.root,"composition_root")
        try Self.require(nativeOwners==expectedOwners,"Measured native owners, ordered names, shapes or after-check flags differ from submitted source.")
        try Self.require(nativeContainers.count==expectedContainers.count && zip(nativeContainers,expectedContainers).allSatisfy {$0.0.path==$0.1.path && $0.0.kind==$0.1.kind && $0.0.keys==$0.1.keys},"Measured native containers differ from submitted child registration order.")
        try Self.require(savedStateKeys==expectedOwners.map(\.name),"Saved state keys differ from actual submitted native ownership order.")
        try Self.require(invocations.count==nodes.count*2,"Invocation log must cover every submitted primitive once.")
        var completed=Set<String>(),order:[String]=[]
        for index in stride(from:0,to:invocations.count,by:2) {
            let started=invocations[index],finished=invocations[index+1]
            try Self.require(started.phase=="started" && finished.phase=="completed" && started.node==finished.node && nodes[started.node] != nil && !completed.contains(started.node),"Invocation phases, node pairs or unique ownership are invalid.")
            let dependencies=expansion.graph.edges.filter {$0.target==started.node}.map(\.source)
            try Self.require(dependencies.allSatisfy {completed.contains($0)},"Invocation ran before its connected inputs were available.")
            completed.insert(started.node);order.append(started.node)
        }
        try Self.require(Set(nodes.keys)==completed && nodes[order.first ?? ""]?.type=="input" && nodes[order.last ?? ""]?.type=="output","Invocation boundaries do not match the submitted model.")
    }
    public static func project(plan:LearningAssessmentPlan,evidence:LearningEvidence,currentSubmission:Data,registry:ComponentRegistry,mode:LearningMode,attemptMode:LearningMode)->[Self]? {
        guard let projected=ModuleCompositionFormativeProjection.project(plan:plan,evidence:evidence,currentSubmission:currentSubmission,registry:registry,mode:mode,attemptMode:attemptMode),
              let task=ModuleCompositionLearningTask.allCases.first(where:{plan.id=="assessment."+$0.rawValue}),
              let submission=try? ModuleCompositionLearningSubmission.decode(currentSubmission,task:task,registry:registry) else {return nil}
        var result:[Self]=[]
        for fixture in projected.fixtures {
            guard let metadata=fixture.composition,(try? metadata.validate(source:submission.source,task:task,registry:registry)) != nil else {return nil}
            result.append(metadata)
        }
        return result
    }
}
