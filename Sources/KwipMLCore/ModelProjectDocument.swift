import Foundation

/// Authoring ownership is distinct from the primitive graph submitted to the worker.
public enum ModelProjectDocument:Equatable {
    case legacyFlat(GraphDocument)
    case blockModel(ModelAuthoringDocument)
}
public struct ModelAuthoringDocument:Codable,Equatable {
    public var schemaVersion=1
    public var kind="modelProject"
    public var projectID:String
    public var revision:UInt64
    public var source:CompositeProjectDocument
    public var drafts:[ModelBlockDraft]
    public init(projectID:String,revision:UInt64=0,source:CompositeProjectDocument,drafts:[ModelBlockDraft]=[],schemaVersion:Int=1) {
        self.schemaVersion=schemaVersion
        self.projectID=projectID;self.revision=revision;self.source=source;self.drafts=drafts
    }
}
/// The owner envelope before an explicit selected-policy migration, retained only in its draft.
public enum CompositeDraftMigrationOrigin: Codable, Equatable {
    case model(modelSchemaVersion: Int, sourceSchemaVersion: Int)
    case workflow(schemaVersion: Int)
    private enum Keys: String, CodingKey { case kind, modelSchemaVersion, sourceSchemaVersion, workflowSchemaVersion }
    private struct AnyKey: CodingKey {
        let stringValue: String; let intValue: Int? = nil
        init?(stringValue: String) { self.stringValue = stringValue }
        init?(intValue: Int) { return nil }
    }
    private static func failure() -> CompositeContractError { .init(.immutableRevision, "Draft migration origin does not match its original owner envelope.") }
    public init(from decoder: Decoder) throws {
        let keys = Set(try decoder.container(keyedBy: AnyKey.self).allKeys.map(\.stringValue))
        let c = try decoder.container(keyedBy: Keys.self)
        switch try c.decode(String.self, forKey: .kind) {
        case "model":
            guard keys == ["kind", "modelSchemaVersion", "sourceSchemaVersion"] else { throw Self.failure() }
            self = .model(modelSchemaVersion: try c.decode(Int.self, forKey: .modelSchemaVersion), sourceSchemaVersion: try c.decode(Int.self, forKey: .sourceSchemaVersion))
            try validate(modelSchemaVersion: 2, sourceSchemaVersion: 3)
        case "workflow":
            guard keys == ["kind", "workflowSchemaVersion"] else { throw Self.failure() }
            self = .workflow(schemaVersion: try c.decode(Int.self, forKey: .workflowSchemaVersion))
            try validate(workflowSchemaVersion: 11)
        default: throw Self.failure()
        }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: Keys.self)
        switch self {
        case .model(let model, let source):
            try c.encode("model", forKey: .kind); try c.encode(model, forKey: .modelSchemaVersion); try c.encode(source, forKey: .sourceSchemaVersion)
        case .workflow(let version):
            try c.encode("workflow", forKey: .kind); try c.encode(version, forKey: .workflowSchemaVersion)
        }
    }
    public func validate(modelSchemaVersion: Int, sourceSchemaVersion: Int) throws {
        guard case .model(let model, let source) = self, model == 1, source == 2, modelSchemaVersion == 2, sourceSchemaVersion == 3 else { throw Self.failure() }
    }
    public func validate(workflowSchemaVersion: Int) throws {
        guard case .workflow(let version) = self, (1...10).contains(version), [11,12,13,14,15,16,17].contains(workflowSchemaVersion) else { throw Self.failure() }
    }
    public static func validateWire(_ value: JSONValue) throws {
        guard case .object(let raw) = value, let kind = raw["kind"]?.stringValue else { throw failure() }
        if kind == "model" {
            guard Set(raw.keys) == ["kind", "modelSchemaVersion", "sourceSchemaVersion"],
                  case .integer(let m) = raw["modelSchemaVersion"], case .integer(let s) = raw["sourceSchemaVersion"], m == 1, s == 2 else { throw failure() }
        } else if kind == "workflow" {
            guard Set(raw.keys) == ["kind", "workflowSchemaVersion"], case .integer(let v) = raw["workflowSchemaVersion"], (1...10).contains(v) else { throw failure() }
        } else { throw failure() }
    }
}
/// Unpublished child construction, captured against one parent revision and instance path.
public struct ModelBlockDraft:Codable,Equatable {
    public var projectID:String
    public var instancePath:[String]
    public var baseReference:CompositeDefinitionReference
    public var baseRevision:UInt64
    public var proposed:CompositeDefinition
    public var migrationOrigin:CompositeDraftMigrationOrigin?
    public init(projectID:String,instancePath:[String],baseReference:CompositeDefinitionReference,baseRevision:UInt64,proposed:CompositeDefinition,migrationOrigin:CompositeDraftMigrationOrigin?=nil) {
        self.projectID=projectID;self.instancePath=instancePath;self.baseReference=baseReference;self.baseRevision=baseRevision;self.proposed=proposed;self.migrationOrigin=migrationOrigin
    }
}
public enum ModelProjectCodec {
    public static let maximumBytes=2_000_000
    public static func decode(_ data:Data)throws->ModelProjectDocument {
        try check(data.count<=maximumBytes,"Model document exceeds 2 MB.")
        guard case .object(let root)=try JSONDecoder().decode(JSONValue.self,from:data) else {throw error("Model document must be an object.")}
        let result:ModelProjectDocument
        if root["kind"] != nil {
            try check(root["kind"] == .string("modelProject"),"Unknown model document kind; no flat fallback is allowed.")
            try exact(root,required:["kind","schemaVersion","projectID","revision","source","drafts"])
            guard case .array(let drafts)=root["drafts"] else {throw error("Model drafts must be an array.")}
            for draft in drafts {
                guard case .object(let value)=draft else {throw error("Malformed Block draft.")}
                try exact(value,required:["projectID","instancePath","baseReference","baseRevision","proposed"],optional:["migrationOrigin"])
                if let origin=value["migrationOrigin"] {try CompositeDraftMigrationOrigin.validateWire(origin)}
            }
            result = .blockModel(try JSONDecoder().decode(ModelAuthoringDocument.self,from:data))
        } else {
            // Historical flat decoding keeps its existing optional/unknown execution-field behavior.
            try check(Set(root.keys).isDisjoint(with:["source","drafts","projectID","revision"]),"Untagged model authoring fields cannot be read as a flat graph.")
            result = .legacyFlat(try JSONDecoder().decode(GraphDocument.self,from:data))
        }
        try validate(result);return result
    }
    public static func encode(_ document:ModelProjectDocument)throws->Data {
        try validate(document)
        let encoder=JSONEncoder();encoder.outputFormatting=[.prettyPrinted,.sortedKeys]
        let data:Data
        switch document {case .legacyFlat(let graph):data=try encoder.encode(graph);case .blockModel(let model):data=try encoder.encode(model)}
        try check(data.count<=maximumBytes,"Model document exceeds 2 MB.");return data
    }
    public static func enablingComposition(_ model:ModelAuthoringDocument)throws->ModelAuthoringDocument {
        try validate(.blockModel(model))
        var copy = model; copy.schemaVersion = 2; copy.source.schemaVersion = 3
        try validate(.blockModel(copy)); return copy
    }
    public static func promote(_ graph:GraphDocument,projectID:String)throws->ModelAuthoringDocument {
        try validate(.legacyFlat(graph))
        let source=CompositeProjectDocument(name:graph.name,root:CompositeGraph(nodes:graph.nodes,edges:graph.edges,configurationVersion:graph.schemaVersion),definitions:[],execution:graph.execution)
        let value=ModelAuthoringDocument(projectID:projectID,source:source)
        try validate(.blockModel(value));return value
    }
    public static func validate(_ document:ModelProjectDocument)throws {
        switch document {
        case .legacyFlat(let graph):
            // Match historical managed-flat storage validation; execution validation is separate.
            try check([1,2].contains(graph.schemaVersion) && graph.nodes.count<=128 && graph.edges.count<=256,"Unsupported or oversized flat model.")
            try check(graph.compositionPlan == nil, "Managed native composition requires its source-qualified Block model, not a source-less flat authoring document.")
            try check(graph.nodes.allSatisfy{$0.x.isFinite && $0.y.isFinite && $0.config.values.allSatisfy(\.isFiniteJSON)},"Flat model contains nonfinite values.")
        case .blockModel(let model):
            try check([1,2].contains(model.schemaVersion) && model.kind == "modelProject" && UUID(uuidString:model.projectID) != nil,"Invalid model authoring identity or version.")
            try check(model.source.schemaVersion == (model.schemaVersion == 2 ? 3 : 2) && model.source.definitions.count<=64 && model.drafts.count<=1,"Model source requires composite schema 2, at most 64 definitions and one child draft.")
            try check(model.schemaVersion == 2 || (model.source.definitions.allSatisfy { $0.composition == nil } && model.drafts.allSatisfy { $0.proposed.composition == nil }),"Native container policies require the composition authoring and execution contracts; legacy model files cannot preserve their ownership.")
            try body(model.source.root)
            var definitions:[CompositeDefinitionReference:CompositeDefinition]=[:]
            for value in model.source.definitions {
                try definition(value)
                try check(definitions[value.reference] == nil,"Duplicate pinned Block definition.")
                definitions[value.reference]=value
            }
            // Unresolved references and cycles remain editable; execution resolution is authoritative.
            // Bounded traversal checks depth for all resolvable, nonrecursive authoring paths.
            var visited=Set<DepthVisit>()
            try depth(model.source.root,definitions:definitions,path:[],active:[],visited:&visited)
            for value in model.source.definitions {try depth(value.body,definitions:definitions,path:[],active:[value.reference],visited:&visited)}
            for draft in model.drafts {
                try draft.migrationOrigin?.validate(modelSchemaVersion:model.schemaVersion,sourceSchemaVersion:model.source.schemaVersion)
                try check(draft.projectID == model.projectID && draft.baseRevision == model.revision && (1...8).contains(draft.instancePath.count) && draft.instancePath.allSatisfy(named),"Child draft does not match its parent identity, revision or bounded instance path.")
                try reference(draft.baseReference);try definition(draft.proposed)
                try check(draft.proposed.reference == draft.baseReference,"A child draft must retain its base reference until Apply creates a revision.")
                var scope=model.source.root
                var resolved:CompositeDefinitionReference?
                for id in draft.instancePath {
                    guard let instance=scope.instances.first(where:{$0.id == id}),let value=definitions[instance.definition] else {throw error("Child draft path does not resolve in its parent model.")}
                    resolved=instance.definition;scope=value.body
                }
                try check(resolved == draft.baseReference,"Child draft base differs from its selected instance.")
                // Include the path depth already consumed before proposed nested contents.
                try depth(draft.proposed.body,definitions:definitions,path:draft.instancePath,active:[draft.baseReference],visited:&visited)
            }
        }
    }
    private static func body(_ value:CompositeGraph)throws {
        try check(value.configurationVersion == nil || [1,2].contains(value.configurationVersion!),"Unsupported Block configuration version.")
        try check(value.nodes.count+value.instances.count<=128 && value.edges.count<=256,"Block body exceeds component or connection bounds.")
        let ids=value.nodes.map(\.id)+value.instances.map(\.id)
        try check(ids.allSatisfy(named) && Set(ids).count == ids.count,"Block component identities must be bounded and unique.")
        for node in value.nodes {
            if let settings=node.parameterSettings {try LayerParameterSettings.validate(settings)}
            try check(named(node.type) && node.x.isFinite && node.y.isFinite && node.config.count<=128 && node.config.keys.allSatisfy(named) && node.config.values.allSatisfy(\.isFiniteJSON),"Invalid Block primitive metadata.")
        }
        for instance in value.instances {
            try reference(instance.definition)
            try check(instance.x.isFinite && instance.y.isFinite && instance.overrides.count<=128 && instance.overrides.keys.allSatisfy(named) && instance.overrides.values.allSatisfy(\.isFiniteJSON),"Invalid Block instance metadata.")
        }
        try check(Set(value.edges.map(\.id)).count == value.edges.count,"Duplicate Block connection identity.")
        for edge in value.edges {try check([edge.id,edge.source,edge.sourcePort,edge.target,edge.targetPort].allSatisfy(named),"Invalid Block connection metadata.")}
    }
    private static func definition(_ value:CompositeDefinition)throws {
        try reference(value.reference);try body(value.body)
        try check(value.inputs.count<=128 && value.outputs.count<=128 && value.parameters.count<=128,"Block interface exceeds bounds.")
        try check(Set(value.inputs.map(\.id)).count == value.inputs.count && Set(value.outputs.map(\.id)).count == value.outputs.count && Set(value.parameters.map{$0.field.key}).count == value.parameters.count,"Duplicate Block interface identity.")
        try check(value.inputs.reduce(0){$0+$1.targets.count}<=256 && value.parameters.reduce(0){$0+$1.targets.count}<=256,"Block interface bindings exceed bounds.")
        for input in value.inputs {
            try check(named(input.id) && input.targets.allSatisfy{named($0.componentID) && named($0.portID)},"Invalid Block input metadata.")
        }
        for output in value.outputs {try check(named(output.id) && named(output.source.componentID) && named(output.source.portID),"Invalid Block output metadata.")}
        for parameter in value.parameters {
            let field=parameter.field
            try check(named(field.key) && field.defaultValue.isFiniteJSON && (field.min?.isFinite ?? true) && (field.max?.isFinite ?? true) && parameter.targets.allSatisfy{named($0.componentID) && named($0.field)},"Invalid Block parameter metadata.")
        }
    }
    private struct DepthVisit:Hashable {let reference:CompositeDefinitionReference;let depth:Int}
    private static func depth(_ body:CompositeGraph,definitions:[CompositeDefinitionReference:CompositeDefinition],path:[String],active:Set<CompositeDefinitionReference>,visited:inout Set<DepthVisit>)throws {
        for instance in body.instances {
            try check(path.count<8,"Block authoring exceeds nesting depth 8.")
            guard let child=definitions[instance.definition],!active.contains(instance.definition) else {continue}
            guard visited.insert(DepthVisit(reference:instance.definition,depth:path.count+1)).inserted else {continue}
            try depth(child.body,definitions:definitions,path:path+[instance.id],active:active.union([instance.definition]),visited:&visited)
        }
    }
    private static func reference(_ value:CompositeDefinitionReference)throws {
        try check(named(value.id) && [value.version.major,value.version.minor,value.version.patch].allSatisfy{(0...Int(Int32.max)).contains($0)},"Invalid Block definition reference.")
    }
    private static func named(_ value:String)->Bool {!value.isEmpty && value.unicodeScalars.count<=128}
    private static func exact(_ value:[String:JSONValue],required:Set<String>,optional:Set<String>=[])throws {try check(required.isSubset(of:Set(value.keys)) && Set(value.keys).isSubset(of:required.union(optional)),"Unknown or missing model authoring fields.")}
    private static func check(_ condition:Bool,_ message:String)throws {if !condition {throw error(message)}}
    private static func error(_ message:String)->CompositeContractError {CompositeContractError(.invalidPrimitive,message)}
}
