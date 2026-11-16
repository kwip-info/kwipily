import Foundation

/// Extends the existing frozen-model node. Empty configuration retains the
/// original Saved checkpoint interpretation and its private History authority.
public struct LoadModelConfiguration:Equatable,Codable {
    public enum Source:String,Codable,CaseIterable {case savedRun,localFolder,huggingFace,documentPackage}
    public var source:Source
    public var repository:String
    public var revision:String
    public var subfolder:String
    public var maximumTokens:Int
    public var truncation:String
    public var padding:String
    public var paddingSide:String
    public init(source:Source = .savedRun,repository:String="",revision:String="main",subfolder:String="",maximumTokens:Int=512,truncation:String="reject",padding:String="longest",paddingSide:String="right") {
        self.source=source;self.repository=repository;self.revision=revision;self.subfolder=subfolder
        self.maximumTokens=maximumTokens;self.truncation=truncation;self.padding=padding;self.paddingSide=paddingSide
    }
    public static func decode(_ values:[String:JSONValue])throws->Self {
        if values.isEmpty {return .init()}
        let fields:Set<String>=["schemaVersion","source","repository","revision","subfolder","maximumTokens","truncation","padding","paddingSide"]
        guard Set(values.keys)==fields,values["schemaVersion"] == .integer(1),
              let raw=values["source"]?.stringValue,let source=Source(rawValue:raw),source != .savedRun,
              let repository=values["repository"]?.stringValue,let revision=values["revision"]?.stringValue,
              let subfolder=values["subfolder"]?.stringValue,let max=values["maximumTokens"]?.integerValue,
              let truncation=values["truncation"]?.stringValue,let padding=values["padding"]?.stringValue,
              let side=values["paddingSide"]?.stringValue else {throw DataFlowContractError("Invalid versioned Load model settings.")}
        let result=Self(source:source,repository:repository,revision:revision,subfolder:subfolder,maximumTokens:Int(max),truncation:truncation,padding:padding,paddingSide:side)
        try result.validate();return result
    }
    public func validate()throws {
        guard (1...131072).contains(maximumTokens),["reject","truncate"].contains(truncation),["none","longest","maximum"].contains(padding),["left","right"].contains(paddingSide),revision.utf8.count<=128,subfolder.utf8.count<=128,repository.utf8.count<=200 else {throw DataFlowContractError("Load model token settings exceed supported limits.")}
        if source == .huggingFace {
            guard Self.validRepository(repository),Self.validRevision(revision),subfolder.isEmpty || Self.validSegment(subfolder) else {throw DataFlowContractError("Use organization/model, a branch/tag/commit and an optional single model subfolder.")}
        }
        guard ![repository,revision,subfolder].contains(where:{$0.contains("\0")}) else {throw DataFlowContractError("Invalid Load model identity.")}
    }
    public var values:[String:JSONValue] {
        if source == .savedRun{return [:]}
        return ["schemaVersion":1,"source":.string(source.rawValue),"repository":.string(repository),"revision":.string(revision),"subfolder":.string(subfolder),"maximumTokens":.integer(Int64(maximumTokens)),"truncation":.string(truncation),"padding":.string(padding),"paddingSide":.string(paddingSide)]
    }
    public static var configurationSchema:JSONValue {
        let text:JSONValue = .object(["type":.string("string")])
        return .object(["oneOf":.array([
            .object(["type":.string("object"),"maxProperties":0,"description":.string("Saved run; the selected run is a private project binding.")]),
            .object(["type":.string("object"),"additionalProperties":false,"required":.array(["schemaVersion","source","repository","revision","subfolder","maximumTokens","truncation","padding","paddingSide"].map(JSONValue.string)),"properties":.object([
                "schemaVersion":.object(["const":1]),"source":.object(["enum":.array(["localFolder","huggingFace","documentPackage"].map(JSONValue.string))]),"repository":text,"revision":text,"subfolder":text,
                "maximumTokens":.object(["type":.string("integer"),"minimum":1,"maximum":131072]),
                "truncation":.object(["enum":.array(["reject","truncate"].map(JSONValue.string))]),"padding":.object(["enum":.array(["none","longest","maximum"].map(JSONValue.string))]),"paddingSide":.object(["enum":.array(["left","right"].map(JSONValue.string))])
            ])])
        ])])
    }
    public static func validSegment(_ text:String)->Bool {
        !text.isEmpty && text != "." && text != ".." && !text.contains("..") && text.utf8.count<=100 && text.unicodeScalars.allSatisfy {CharacterSet(charactersIn:"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.").contains($0)}
    }
    public static func validRepository(_ text:String)->Bool {let parts=text.split(separator:"/",omittingEmptySubsequences:false);return parts.count==2 && parts.allSatisfy {validSegment(String($0))}}
    public static func validRevision(_ text:String)->Bool {validSegment(text)}
}
