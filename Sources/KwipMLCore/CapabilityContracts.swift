import Foundation
import CryptoKit

public struct ContractVersion:Codable,Equatable,Hashable {
    public var major:Int;public var minor:Int;public var patch:Int
    public init(_ major:Int=1,_ minor:Int=0,_ patch:Int=0) {self.major=major;self.minor=minor;self.patch=patch}
}
public enum ContractBackend:String,Codable {case portable,pytorch,scikitLearn,tensorflow}
public enum ContractDevice:String,Codable {case runtimeSelected,host,cpu,mps,cuda}
public enum ContractDType:String,Codable {case float32,float64,int32,int64,bool,string}
public enum ContractPhase:String,Codable {case initialize,fit,transform,forward,backward,train,evaluate,predict,checkpoint,resume,export,reset}
public enum PortDirection:String,Codable {case input,output}
public enum SchemaKind:String,Codable {case tensor,table,dataset,estimator,state,artifact}
public enum PartitionRole:String,Codable {case training,validation,test,unassigned}
public enum StateKind:String,Codable,CaseIterable {case parameters,buffers,recurrent,inferenceCache,fittedTransform,optimizer,scheduler,objective,initialization,gradientPolicy,trainability,randomGenerator,dataCursor}
public enum StateSharing:String,Codable {case privateInstance,explicitShared}
public enum FitBoundary:String,Codable {case notApplicable,trainingPartitionOnly}
public enum ArtifactKind:String,Codable {case modelDefinition,weights,trainingCheckpoint,fittedEstimator,tensorCache,datasetPartition,assessmentEvidence}
public enum ResumeGuarantee:String,Codable {case none,weightsOnly,fullTraining}
public enum CapabilityAvailability:String,Codable {case contractOnly,existingPOCAdapter}

/// Tagged dimensions remain readable JSON; parameter names resolve against node config.
public struct DimensionContract:Codable,Equatable {
    public enum Kind:String,Codable {case fixed,symbol,parameter}
    public var kind:Kind;public var value:Int?;public var name:String?
    public init(kind:Kind,value:Int?=nil,name:String?=nil) {self.kind=kind;self.value=value;self.name=name}
    public static func fixed(_ value:Int)->Self {Self(kind:.fixed,value:value)}
    public static func symbol(_ name:String)->Self {Self(kind:.symbol,name:name)}
    public static func parameter(_ name:String)->Self {Self(kind:.parameter,name:name)}
}
public struct TensorLeadingAxes:Codable,Equatable {
    public var minimum:Int
    public var maximum:Int
    public init(minimum:Int,maximum:Int) {self.minimum=minimum;self.maximum=maximum}
}
public struct TensorContract:Codable,Equatable {
    /// Optional variable leading axes, followed by the fixed-length dimensions suffix.
    /// Concrete connection validation requires specialization by the declared shape rule.
    public var leadingAxes:TensorLeadingAxes?
    public var dimensions:[DimensionContract];public var dtype:ContractDType;public var backend:ContractBackend;public var device:ContractDevice;public var gradients:Bool
    public init(_ dimensions:[DimensionContract],dtype:ContractDType = .float32,backend:ContractBackend = .pytorch,device:ContractDevice = .runtimeSelected,gradients:Bool=false,leadingAxes:TensorLeadingAxes?=nil) {self.dimensions=dimensions;self.dtype=dtype;self.backend=backend;self.device=device;self.gradients=gradients;self.leadingAxes=leadingAxes}
}
public struct ColumnContract:Codable,Equatable {
    public var name:String;public var dtype:ContractDType;public var nullable:Bool
    public init(_ name:String,dtype:ContractDType,nullable:Bool=false) {self.name=name;self.dtype=dtype;self.nullable=nullable}
}
public struct TableContract:Codable,Equatable {
    public var columns:[ColumnContract];public var allowsAdditionalColumns:Bool
    public init(_ columns:[ColumnContract],allowsAdditionalColumns:Bool=false) {self.columns=columns;self.allowsAdditionalColumns=allowsAdditionalColumns}
}
public struct DatasetContract:Codable,Equatable {
    public var table:TableContract?;public var tensor:TensorContract?;public var partition:PartitionRole
    public init(table:TableContract?=nil,tensor:TensorContract?=nil,partition:PartitionRole = .unassigned) {self.table=table;self.tensor=tensor;self.partition=partition}
}
public struct EstimatorContract:Codable,Equatable {
    public var backend:ContractBackend;public var family:String;public var requiresFitted:Bool
    public init(backend:ContractBackend,family:String,requiresFitted:Bool=true) {self.backend=backend;self.family=family;self.requiresFitted=requiresFitted}
}
public struct StatePortContract:Codable,Equatable {
    public var key:String;public var kind:StateKind;public var sharing:StateSharing
    public init(key:String,kind:StateKind,sharing:StateSharing = .privateInstance) {self.key=key;self.kind=kind;self.sharing=sharing}
}
public struct ArtifactContract:Codable,Equatable {
    public var kind:ArtifactKind;public var backend:ContractBackend;public var format:String;public var resume:ResumeGuarantee
    public init(kind:ArtifactKind,backend:ContractBackend,format:String,resume:ResumeGuarantee = .none) {self.kind=kind;self.backend=backend;self.format=format;self.resume=resume}
}
/// Exactly one matching payload is required. Malformed decoded schemas are rejected.
public struct PortSchema:Codable,Equatable {
    public var kind:SchemaKind
    public var tensor:TensorContract?;public var table:TableContract?;public var dataset:DatasetContract?;public var estimator:EstimatorContract?;public var state:StatePortContract?;public var artifact:ArtifactContract?
    public init(kind:SchemaKind,tensor:TensorContract?=nil,table:TableContract?=nil,dataset:DatasetContract?=nil,estimator:EstimatorContract?=nil,state:StatePortContract?=nil,artifact:ArtifactContract?=nil) {self.kind=kind;self.tensor=tensor;self.table=table;self.dataset=dataset;self.estimator=estimator;self.state=state;self.artifact=artifact}
    public static func tensor(_ value:TensorContract)->Self {Self(kind:.tensor,tensor:value)}
    public static func table(_ value:TableContract)->Self {Self(kind:.table,table:value)}
    public static func dataset(_ value:DatasetContract)->Self {Self(kind:.dataset,dataset:value)}
    public static func estimator(_ value:EstimatorContract)->Self {Self(kind:.estimator,estimator:value)}
    public static func state(_ value:StatePortContract)->Self {Self(kind:.state,state:value)}
    public static func artifact(_ value:ArtifactContract)->Self {Self(kind:.artifact,artifact:value)}
}
public struct PortContract:Codable,Equatable,Identifiable {
    public var id:String;public var direction:PortDirection;public var schema:PortSchema;public var phases:[ContractPhase];public var required:Bool;public var maximumConnections:Int?
    public init(_ id:String,direction:PortDirection,schema:PortSchema,phases:[ContractPhase],required:Bool=true,maximumConnections:Int?=1) {self.id=id;self.direction=direction;self.schema=schema;self.phases=phases;self.required=required;self.maximumConnections=maximumConnections}
}
public struct StateDeclaration:Codable,Equatable {
    public var key:String;public var kind:StateKind;public var sharing:StateSharing;public var resetPhase:ContractPhase;public var serializable:Bool
    public init(_ key:String,kind:StateKind,sharing:StateSharing = .privateInstance,resetPhase:ContractPhase = .initialize,serializable:Bool=true) {self.key=key;self.kind=kind;self.sharing=sharing;self.resetPhase=resetPhase;self.serializable=serializable}
}
public struct LifecycleContract:Codable,Equatable {
    public var phases:[ContractPhase];public var fitBoundary:FitBoundary;public var state:[StateDeclaration];public var resume:ResumeGuarantee
    public init(phases:[ContractPhase],fitBoundary:FitBoundary = .notApplicable,state:[StateDeclaration]=[],resume:ResumeGuarantee = .none) {self.phases=phases;self.fitBoundary=fitBoundary;self.state=state;self.resume=resume}
}
public struct BackendBoundary:Codable,Equatable {
    public var source:ContractBackend;public var destination:ContractBackend;public var detached:Bool;public var adapterID:String
    public init(source:ContractBackend,destination:ContractBackend,detached:Bool=true,adapterID:String) {self.source=source;self.destination=destination;self.detached=detached;self.adapterID=adapterID}
}
public struct LearningContract:Codable,Equatable {
    public var conceptIDs:[String];public var lessonIDs:[String];public var exampleIDs:[String];public var challengeIDs:[String];public var assessedOutputs:[String]
    public init(conceptIDs:[String]=[],lessonIDs:[String]=[],exampleIDs:[String]=[],challengeIDs:[String]=[],assessedOutputs:[String]=[]) {self.conceptIDs=conceptIDs;self.lessonIDs=lessonIDs;self.exampleIDs=exampleIDs;self.challengeIDs=challengeIDs;self.assessedOutputs=assessedOutputs}
}
public struct CapabilityDescriptor:Codable,Equatable,Identifiable {
    public var id:String;public var version:ContractVersion;public var packageID:String;public var backend:ContractBackend;public var ports:[PortContract];public var fields:[ComponentField];public var lifecycle:LifecycleContract;public var devices:[ContractDevice];public var boundary:BackendBoundary?;public var availability:CapabilityAvailability;public var adapterID:String?;public var learning:LearningContract;public var shapeRuleID:String?
    public init(id:String,version:ContractVersion = ContractVersion(),packageID:String,backend:ContractBackend,ports:[PortContract],fields:[ComponentField]=[],lifecycle:LifecycleContract,devices:[ContractDevice],boundary:BackendBoundary?=nil,availability:CapabilityAvailability = .contractOnly,adapterID:String?=nil,learning:LearningContract = LearningContract(),shapeRuleID:String?=nil) {self.id=id;self.version=version;self.packageID=packageID;self.backend=backend;self.ports=ports;self.fields=fields;self.lifecycle=lifecycle;self.devices=devices;self.boundary=boundary;self.availability=availability;self.adapterID=adapterID;self.learning=learning;self.shapeRuleID=shapeRuleID}
}
public struct CapabilityPackageDescriptor:Codable,Equatable {
    public var schemaVersion:Int;public var id:String;public var version:ContractVersion;public var bundled:Bool;public var backendVersions:[String:String];public var capabilities:[CapabilityDescriptor]
    public init(schemaVersion:Int=1,id:String,version:ContractVersion = ContractVersion(),bundled:Bool=true,backendVersions:[String:String],capabilities:[CapabilityDescriptor]) {self.schemaVersion=schemaVersion;self.id=id;self.version=version;self.bundled=bundled;self.backendVersions=backendVersions;self.capabilities=capabilities}
}
public struct ContractEndpoint {
    public var capability:CapabilityDescriptor;public var instanceID:String;public var portID:String;public var configuration:[String:JSONValue];public var stateOwnerID:String?
    public init(_ capability:CapabilityDescriptor,instanceID:String,portID:String,configuration:[String:JSONValue]=[:],stateOwnerID:String?=nil) {self.capability=capability;self.instanceID=instanceID;self.portID=portID;self.configuration=configuration;self.stateOwnerID=stateOwnerID}
}
public struct PartitionEvidence:Codable,Equatable {
    public var role:PartitionRole;public var sourceDigest:String;public var membershipDigest:String;public var observationIDs:[String]
    public init(role:PartitionRole,sourceDigest:String,membershipDigest:String,observationIDs:[String]) {self.role=role;self.sourceDigest=sourceDigest;self.membershipDigest=membershipDigest;self.observationIDs=observationIDs}
}
public struct ArtifactIdentity:Codable,Equatable {
    public var id:String;public var schemaVersion:Int;public var contract:ArtifactContract;public var contentDigest:String;public var producerCapabilityID:String;public var producerVersion:ContractVersion;public var graphDigest:String;public var includedState:[StateKind]
    public init(id:String,schemaVersion:Int=1,contract:ArtifactContract,contentDigest:String,producerCapabilityID:String,producerVersion:ContractVersion,graphDigest:String,includedState:[StateKind]=[]) {self.id=id;self.schemaVersion=schemaVersion;self.contract=contract;self.contentDigest=contentDigest;self.producerCapabilityID=producerCapabilityID;self.producerVersion=producerVersion;self.graphDigest=graphDigest;self.includedState=includedState}
}
public struct ContractIssue:Equatable,Identifiable {
    public enum Severity:String {case error,deferred}
    public var id:String {path+":"+code};public let code:String;public let path:String;public let message:String;public let severity:Severity
    public init(_ code:String,path:String,message:String,severity:Severity = .error) {self.code=code;self.path=path;self.message=message;self.severity=severity}
}
public struct ContractValidation:Equatable {
    public var issues:[ContractIssue]
    public var isValid:Bool {!issues.contains {$0.severity == .error}}
    public var isFullyResolved:Bool {issues.isEmpty}
    public init(_ issues:[ContractIssue]=[]) {self.issues=issues}
}

public extension PartitionEvidence {
    /// SHA-256 of domain prefix and sorted length-prefixed UTF-8 row identities.
    /// Independent of membership order, locale and JSON encoder escaping choices.
    static func membershipDigest(for observationIDs:[String])->String {
        var data=Data("kwip.membership.v1\n".utf8)
        let rows=observationIDs.map {Array($0.utf8)}.sorted {$0.lexicographicallyPrecedes($1)}
        for row in rows {data.append(contentsOf:"\(row.count):".utf8);data.append(contentsOf:row)}
        return SHA256.hash(data:data).map {String(format:"%02x",$0)}.joined()
    }
}
