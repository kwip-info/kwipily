import Foundation

/// Canonical versioned starter content shared by the native lesson and example.
public struct SklearnFoundationLesson: Decodable {
    public let schemaVersion: Int
    public let id: String
    public let title: String
    public let contentVersion: String
    public let reviewedAt: String
    public let objective: String
    public let formula: String
    public let steps: [String]
    public let exercise: String
    public let assessment: String
    public let limits: String
    public static func bundled() throws -> Self {
        guard let url = Bundle.module.url(forResource: "sklearn-foundation-learning", withExtension: "json") else { throw DataFlowContractError("The scikit-learn starter lesson is missing from this build.") }
        let data = try Data(contentsOf: url)
        guard data.count <= 64_000 else { throw DataFlowContractError("The scikit-learn lesson exceeds its resource bound.") }
        let lesson = try JSONDecoder().decode(Self.self, from: data)
        guard lesson.schemaVersion == 1, lesson.id == "sklearn.logistic-foundation", !lesson.contentVersion.isEmpty, !lesson.reviewedAt.isEmpty, !lesson.steps.isEmpty else { throw DataFlowContractError("The scikit-learn starter lesson is invalid.") }
        return lesson
    }
}
