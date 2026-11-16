import Foundation
import CryptoKit

/// One application-owned capability catalog, copied verbatim into the isolated worker.
public enum PrebuiltModelConfiguration {
    public struct Field:Decodable,Equatable {public let key:String;public let title:String;public let kind:String;public let `default`:JSONValue;public let minimum:Double?;public let maximum:Double?}
    public struct Algorithm:Decodable,Equatable {public let title:String;public let task:String;public let predict:Bool;public let fields:[Field]}
    public static let algorithms:[String:Algorithm] = {
        guard let url=Bundle.module.url(forResource:"prebuilt-models",withExtension:"json"),let data=try? Data(contentsOf:url),let result=try? JSONDecoder().decode([String:Algorithm].self,from:data) else {return [:]};return result
    }()
    public static let catalogSHA256:String = {guard let url=Bundle.module.url(forResource:"prebuilt-models",withExtension:"json"),let data=try? Data(contentsOf:url) else{return ""};return SHA256.hash(data:data).map{String(format:"%02x",$0)}.joined()}()
    public static var catalogJSON:JSONValue {guard let url=Bundle.module.url(forResource:"prebuilt-models",withExtension:"json"),let data=try? Data(contentsOf:url),let value=try? JSONDecoder().decode(JSONValue.self,from:data) else{return .object([:])};return value}
    public static func isPrebuilt(_ values:[String:JSONValue])->Bool {values["task"] == .string("prebuilt")}
    public static func isCluster(_ values:[String:JSONValue])->Bool {isPrebuilt(values) && algorithms[values["algorithm"]?.stringValue ?? ""]?.task == "clustering"}
    public static func hasTables(_ values:[String:JSONValue])->Bool {isPrebuilt(values) || DocumentClassificationConfiguration.isDocument(values)}
    public static func hasModel(_ values:[String:JSONValue])->Bool {hasTables(values) && values["algorithm"] != .string("dbscan") && (values["operation"] == nil || values["operation"] == .string("fit"))}
    public static func defaults(_ algorithm:String="logistic")->[String:JSONValue] {
        guard let entry=algorithms[algorithm] else {return [:]}
        return ["backendID":.string("sklearn"),"task":.string("prebuilt"),"algorithm":.string(algorithm),"operation":.string("fit"),"featureColumns":.array([.string("feature")]),"targetColumn":.string(entry.task == "clustering" ? "":"target"),"settings":.object(Dictionary(uniqueKeysWithValues:entry.fields.map{($0.key,$0.default)}))]
    }
    public static func validateSettings(algorithm:String,settings:[String:JSONValue])throws {
        guard let entry=algorithms[algorithm],Set(settings.keys)==Set(entry.fields.map(\.key)) else {throw DataFlowContractError("Choose a supported Prebuilt model and its declared options.")}
        for field in entry.fields {
            let value=settings[field.key]
            if field.kind=="bool" {guard case .bool=value else {throw DataFlowContractError("\(field.title) must be Boolean.")}}
            else {guard let number=value?.doubleValue,number.isFinite,number >= (field.minimum ?? 0),number <= (field.maximum ?? 0),field.kind != "integer" || value?.integerValue != nil else {throw DataFlowContractError("\(field.title) is outside its supported range.")}}
        }
    }
    public static func decode(_ values:[String:JSONValue])throws->WorkflowEstimatorConfiguration {
        guard Set(values.keys)==["backendID","task","algorithm","operation","featureColumns","targetColumn","settings"],values["backendID"] == .string("sklearn"),isPrebuilt(values),let algorithm=values["algorithm"]?.stringValue,let entry=algorithms[algorithm],let operation=values["operation"]?.stringValue,["fit","predict","evaluate"].contains(operation),case .array(let raw)=values["featureColumns"],let target=values["targetColumn"]?.stringValue,case .object(let settings)=values["settings"] else {throw DataFlowContractError("Invalid Prebuilt models configuration.")}
        let names=raw.compactMap(\.stringValue)
        guard names.count==raw.count,(1...128).contains(names.count),Set(names).count==names.count,names.allSatisfy({!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count<=200}),!names.contains(target) else {throw DataFlowContractError("Select 1–128 distinct ordered numeric feature columns.")}
        guard entry.predict || operation=="fit",entry.task != "clustering" || operation != "evaluate" else {throw DataFlowContractError("This algorithm does not support the selected operation; DBSCAN is fit-only analysis.")}
        guard entry.task != "clustering" || target.isEmpty else {throw DataFlowContractError("Clustering does not use a target column.")}
        if entry.task != "clustering" && operation != "predict" {try TrainingColumnMapping.validate(features:names,target:target)}
        try validateSettings(algorithm:algorithm,settings:settings)
        var result=WorkflowEstimatorConfiguration(backendID:"sklearn",featureColumns:names,targetColumn:target,settings:settings)
        result.task="prebuilt";result.algorithm=algorithm;result.operation=operation;return result
    }
}
