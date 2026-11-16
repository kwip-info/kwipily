import Foundation

extension WorkflowExamples {
    public static func preparedOptimizerClassifier(componentID: String, projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument {
        var document = try preparedParallelClassifier(projectID: projectID, device: device)
        let configuration = try OptimizerConfiguration(componentID: componentID, values: ["lr": 0.03])
        try CurrentPracticeConfiguration.setTrain("optimizer",value:configuration.recipeValue,in:&document)
        try WorkflowCodec.validate(document)
        return document
    }
    public static func preparedSGDClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedOptimizerClassifier(componentID: "optimizer_sgd", projectID: projectID, device: device) }
    public static func preparedAdamClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedOptimizerClassifier(componentID: "optimizer_adam", projectID: projectID, device: device) }
    public static func preparedAdamWClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedOptimizerClassifier(componentID: "optimizer_adamw", projectID: projectID, device: device) }
    public static func preparedRMSpropClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedOptimizerClassifier(componentID: "optimizer_rmsprop", projectID: projectID, device: device) }
    public static func preparedAdagradClassifier(projectID: String = UUID().uuidString, device: String = "auto") throws -> WorkflowDocument { try preparedOptimizerClassifier(componentID: "optimizer_adagrad", projectID: projectID, device: device) }
}
