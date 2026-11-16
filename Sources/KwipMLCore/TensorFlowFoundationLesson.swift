import Foundation

/// Versioned starter content, shared by every native presentation of this lesson.
public struct TensorFlowFoundationLesson:Decodable {
    public struct Step:Decodable {public let title:String;public let body:String;public let formula:String?}
    public struct Check:Decodable {public let question:String;public let expected:String}
    public let schemaVersion:Int
    public let id:String
    public let contentVersion:String
    public let reviewedAt:String
    public let title:String
    public let summary:String
    public let steps:[Step]
    public let checks:[Check]
    public let limitations:String
    public static func bundled()throws->Self {
        guard let url=Bundle.module.url(forResource:"tensorflow-foundation-learning",withExtension:"json") else {throw DataFlowContractError("The TensorFlow starter lesson is missing from this build.")}
        let data=try Data(contentsOf:url)
        guard data.count<=64_000 else {throw DataFlowContractError("The TensorFlow lesson exceeds its resource bound.")}
        let lesson=try JSONDecoder().decode(Self.self,from:data)
        guard lesson.schemaVersion==1,lesson.id=="tensorflow.dense-foundation",!lesson.contentVersion.isEmpty,!lesson.reviewedAt.isEmpty,!lesson.steps.isEmpty,!lesson.checks.isEmpty else {throw DataFlowContractError("The TensorFlow starter lesson is invalid.")}
        return lesson
    }
}
