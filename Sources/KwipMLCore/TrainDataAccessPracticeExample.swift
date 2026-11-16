import Foundation

/// Ordinary Train controls over the existing verified prepared-classifier source.
/// No extra source authority or recipe nodes are introduced.
public enum TrainDataAccessPracticeExample {
    public static let exampleID = "train-data-access"
    public static let lessonID = "train-data-access"
    public static var datasetAccess: [String: JSONValue] { DatasetConfiguration.defaults }
    public static var batching: [String: JSONValue] {
        var result = DataLoaderConfiguration.defaults
        result["batch_size"] = .integer(7)
        result["collation"] = .string("fixed_stack")
        return result
    }
    public static func document(projectID: String = UUID().uuidString, device: String = "cpu") throws -> WorkflowDocument {
        var document = try WorkflowExamples.preparedParallelClassifier(projectID: projectID, device: device)
        document.schemaVersion = max(document.schemaVersion, 13)
        document.name = "Choose training observations and batches in Train"
        guard let index = document.nodes.firstIndex(where: { $0.address == WorkflowExamples.trainAddress }),
              case .train(var config) = document.nodes[index].payload else {
            throw DataFlowContractError("The training practice needs its authored Train component.")
        }
        config["datasetAccess"] = .object(datasetAccess)
        config["batching"] = .object(batching)
        document.nodes[index].payload = .train(config: config)
        try WorkflowCodec.validate(document)
        return document
    }
}
