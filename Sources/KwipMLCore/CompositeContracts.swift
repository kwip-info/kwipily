import Foundation

public struct CompositeDefinitionReference:Codable,Equatable,Hashable {
    public var id:String;public var version:ContractVersion
    public init(_ id:String,version:ContractVersion=ContractVersion()) {self.id=id;self.version=version}
}
public struct CompositePortReference:Codable,Equatable,Hashable {
    public var componentID:String;public var portID:String
    public init(_ componentID:String,_ portID:String) {self.componentID=componentID;self.portID=portID}
}
public struct CompositeInput:Codable,Equatable {
    public var id:String;public var targets:[CompositePortReference]
    public init(_ id:String,targets:[CompositePortReference]) {self.id=id;self.targets=targets}
}
public struct CompositeOutput:Codable,Equatable {
    public var id:String;public var source:CompositePortReference
    public init(_ id:String,source:CompositePortReference) {self.id=id;self.source=source}
}
public struct CompositeParameterTarget:Codable,Equatable,Hashable {
    public var componentID:String;public var field:String
    public init(_ componentID:String,_ field:String) {self.componentID=componentID;self.field=field}
}
public struct CompositeParameter:Codable,Equatable {
    public var field:ComponentField;public var targets:[CompositeParameterTarget]
    public init(_ field:ComponentField,targets:[CompositeParameterTarget]) {self.field=field;self.targets=targets}
}
public struct CompositeInstance:Codable,Equatable,Identifiable {
    public var id:String;public var definition:CompositeDefinitionReference;public var name:String;public var x:Double;public var y:Double;public var overrides:[String:JSONValue]
    public init(id:String,definition:CompositeDefinitionReference,name:String="",x:Double=0,y:Double=0,overrides:[String:JSONValue]=[:]) {self.id=id;self.definition=definition;self.name=name;self.x=x;self.y=y;self.overrides=overrides}
}
public struct CompositeGraph:Codable,Equatable {
    public var configurationVersion:Int?;public var nodes:[GraphNode];public var instances:[CompositeInstance];public var edges:[GraphEdge]
    public init(nodes:[GraphNode]=[],instances:[CompositeInstance]=[],edges:[GraphEdge]=[],configurationVersion:Int?=2) {self.configurationVersion=configurationVersion;self.nodes=nodes;self.instances=instances;self.edges=edges}
}
public struct CompositeIntegerProductBinding: Codable, Equatable {
    public var lhs: String
    public var rhs: String
    public var targets: [CompositeParameterTarget]
    public init(lhs: String, rhs: String, targets: [CompositeParameterTarget]) {
        self.lhs = lhs; self.rhs = rhs; self.targets = targets
    }
}
public enum CompositeAppearance:String,Codable,Equatable {case parallelFeatures="parallel_features",separable="separable_conv2d",transformer="transformer"}
public struct CompositeDefinition:Codable,Equatable {
    public var appearance:CompositeAppearance?
    public var resolvedAppearance:CompositeAppearance? {
        if let appearance {return appearance}
        if reference==SeparableConv2dBlock.reference {return .separable}
        if reference.id=="kwip.parallel-features" || reference.id=="kwip.parallel-features.container" || title=="Parallel features" {return .parallelFeatures}
        if reference==TransformerBlock.reference {return .transformer}
        return nil
    }
    public var reference:CompositeDefinitionReference;public var title:String;public var body:CompositeGraph;public var inputs:[CompositeInput];public var outputs:[CompositeOutput];public var parameters:[CompositeParameter]
    public var composition:CompositionPolicy?
    public var integerProducts:[CompositeIntegerProductBinding]?
    private enum CodingKeys:String,CodingKey {case reference,title,body,inputs,outputs,parameters,integerProducts,composition,appearance}
    public init(from decoder:Decoder) throws {
        let c = try decoder.container(keyedBy:CodingKeys.self)
        appearance = try c.decodeIfPresent(CompositeAppearance.self,forKey:.appearance)
        reference = try c.decode(CompositeDefinitionReference.self,forKey:.reference)
        title = try c.decode(String.self,forKey:.title)
        body = try c.decode(CompositeGraph.self,forKey:.body)
        inputs = try c.decode([CompositeInput].self,forKey:.inputs)
        outputs = try c.decode([CompositeOutput].self,forKey:.outputs)
        parameters = try c.decode([CompositeParameter].self,forKey:.parameters)
        integerProducts = try c.decodeIfPresent([CompositeIntegerProductBinding].self,forKey:.integerProducts)
        composition = c.contains(.composition) ? try c.decode(CompositionPolicy.self,forKey:.composition) : nil
    }

    public init(reference:CompositeDefinitionReference,title:String,body:CompositeGraph,inputs:[CompositeInput],outputs:[CompositeOutput],parameters:[CompositeParameter]=[],integerProducts:[CompositeIntegerProductBinding]?=nil,composition:CompositionPolicy?=nil,appearance:CompositeAppearance?=nil) {self.appearance=appearance;self.composition=composition;self.reference=reference;self.title=title;self.body=body;self.inputs=inputs;self.outputs=outputs;self.parameters=parameters;self.integerProducts=integerProducts}
}
/// A separate envelope; this does not reinterpret GraphDocument schemaVersion 1.
public struct CompositeProjectDocument:Codable,Equatable {
    public var namedInputs:[String:String]? = nil
    public var schemaVersion:Int;public var name:String;public var root:CompositeGraph;public var definitions:[CompositeDefinition];public var execution:ExecutionSettings?
    public init(schemaVersion:Int=2,name:String="Composite flow",root:CompositeGraph,definitions:[CompositeDefinition],execution:ExecutionSettings?=nil,namedInputs:[String:String]?=nil) {self.namedInputs=namedInputs;self.schemaVersion=schemaVersion;self.name=name;self.root=root;self.definitions=definitions;self.execution=execution}
}
public struct CompositeNodeOrigin:Codable,Equatable {
    public var instancePath:[String];public var localNodeID:String;public var definition:CompositeDefinitionReference?
    public init(instancePath:[String],localNodeID:String,definition:CompositeDefinitionReference?) {self.instancePath=instancePath;self.localNodeID=localNodeID;self.definition=definition}
}
public struct CompositeExpansion:Equatable {
    public var graph:GraphDocument;public var nodeOrigins:[String:CompositeNodeOrigin];public var consumedDefinitions:[CompositeDefinitionReference]
    public init(graph:GraphDocument,nodeOrigins:[String:CompositeNodeOrigin],consumedDefinitions:[CompositeDefinitionReference]) {self.graph=graph;self.nodeOrigins=nodeOrigins;self.consumedDefinitions=consumedDefinitions}
}
public struct CompositeContractError:Error,Equatable,LocalizedError {
    public enum Code:String {case unsupportedSchema,duplicateIdentity,missingDefinition,recursion,resourceLimit,invalidBoundary,invalidParameter,invalidPrimitive,invalidExpandedGraph,immutableRevision}
    public var code:Code;public var message:String
    public var validationIssues: [ValidationIssue]
    public var nodeOrigins: [String: CompositeNodeOrigin]
    public init(_ code:Code,_ message:String, validationIssues:[ValidationIssue] = [], nodeOrigins:[String:CompositeNodeOrigin] = [:]) {
        self.code=code;self.message=message;self.validationIssues=validationIssues;self.nodeOrigins=nodeOrigins
    }
    public var errorDescription:String? {message}
}
