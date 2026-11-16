import Foundation

extension WorkflowExamples {
    public static let sklearnEstimatorAddress = WorkflowNodeAddress(domain: .control, localID: "estimator")
    public static func preparedSklearnClassifier(projectID: String = UUID().uuidString) throws -> WorkflowDocument {
        var document = try WorkflowCodec.migrateToVersion5(preparedParallelClassifier(projectID: projectID, device: "cpu"))
        let removed = Set(document.nodes.filter { $0.address.domain != .data }.map(\.address))
        document.nodes.removeAll { removed.contains($0.address) }
        document.edges.removeAll { removed.contains($0.from.address) || removed.contains($0.to.address) }
        document.name = "Logistic regression · prepared observations"
        document.nodes.append(.init(address: sklearnEstimatorAddress, name: "Learn a linear boundary", x: 1400, y: 220, payload: .estimator(config: [
            "backendID": .string("sklearn"), "featureColumns": .array([.string("signal"), .string("context")]), "targetColumn": .string("label"), "settings": .object(SklearnEstimatorConfiguration.defaultSettings)])))
        for name in ["training", "validation"] {
            document.edges.append(.init(id: "sklearn-\(name)", from: .init(address: .init(domain: .data, localID: "fit"), port: name), to: .init(address: sklearnEstimatorAddress, port: name), kind: .table))
        }
        try WorkflowCodec.validate(document); return document
    }
}
