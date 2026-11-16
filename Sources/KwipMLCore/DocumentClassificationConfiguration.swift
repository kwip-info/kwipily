import Foundation

/// Additive task discriminator. Missing task retains historical two-feature logistic behavior.
public enum DocumentClassificationConfiguration {
    public static let task="documentClassification"
    public static let featureDefaults:[String:JSONValue]=[
        "analyzer":.string("word"),"ngramMin":1,"ngramMax":2,"lowercase":true,"stripAccents":.string("none"),
        "minimumDF":1,"maximumDF":1,"vocabularyLimit":20000,"binary":false,"sublinearTF":false,
        "useIDF":true,"smoothIDF":true,"normalization":.string("l2")]
    public static let estimatorDefaults:[String:JSONValue]=[
        "C":1,"maxIterations":1000,"tolerance":.number(0.0001),"fitIntercept":true,"classWeight":.string("none"),"solver":.string("lbfgs")]
    public static func isDocument(_ values:[String:JSONValue])->Bool {values["task"] == .string(task)}
    public static var defaults:[String:JSONValue] {
        ["backendID":.string("sklearn"),"task":.string(task),"featureColumns":.array([.string("text")]),"targetColumn":.string("category"),"settings":.object(estimatorDefaults)]
    }
    public static func validateFeatures(_ c:[String:JSONValue])throws {
        guard Set(c.keys)==Set(featureDefaults.keys),["word","char"].contains(c["analyzer"]?.stringValue ?? ""),
              ["none","ascii","unicode"].contains(c["stripAccents"]?.stringValue ?? ""),["none","l1","l2"].contains(c["normalization"]?.stringValue ?? ""),
              let low=c["ngramMin"]?.integerValue,let high=c["ngramMax"]?.integerValue,1<=low,low<=high,high<=5,
              let minDF=c["minimumDF"]?.integerValue,(1...1000000).contains(minDF),
              let maxDF=c["maximumDF"]?.doubleValue,maxDF.isFinite,maxDF>0,maxDF<=1,
              let cap=c["vocabularyLimit"]?.integerValue,(1...65536).contains(cap),
              ["lowercase","binary","sublinearTF","useIDF","smoothIDF"].allSatisfy({if case .bool=c[$0] {return true};return false}) else {
            throw DataFlowContractError("Choose valid document-feature settings: word/character n-grams 1–5, document frequency, bounded vocabulary and normalization.")
        }
    }
    public static func validateEstimator(_ c:[String:JSONValue])throws {
        guard Set(c.keys)==Set(estimatorDefaults.keys),c["solver"] == .string("lbfgs"),["none","balanced"].contains(c["classWeight"]?.stringValue ?? "") else {throw DataFlowContractError("Document classification supports L2/lbfgs and explicit none/balanced class weights.")}
        let legacy=c.filter {SklearnEstimatorConfiguration.defaultSettings[$0.key] != nil}
        try SklearnEstimatorConfiguration.validate(legacy)
    }
}
