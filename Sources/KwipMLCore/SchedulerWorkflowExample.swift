import Foundation

extension WorkflowExamples {
    public static func preparedSchedulerClassifier(componentID: String, device: String = "auto") throws -> WorkflowDocument {
        var document = try preparedSGDClassifier(device: device)
        let configuration = try SchedulerConfiguration(componentID: componentID, values: [:])
        document.name = "Prepare and train with " + componentID.replacingOccurrences(of: "scheduler_", with: "").replacingOccurrences(of: "_", with: " ")
        try CurrentPracticeConfiguration.setTrain("scheduler",value:configuration.recipeValue,in:&document)
        try WorkflowCodec.validate(document)
        return document
    }
    public static func preparedStepLRClassifier(device: String = "auto") throws -> WorkflowDocument { try preparedSchedulerClassifier(componentID: "scheduler_step_lr", device: device) }
    public static func preparedCosineAnnealingLRClassifier(device: String = "auto") throws -> WorkflowDocument { try preparedSchedulerClassifier(componentID: "scheduler_cosine_annealing_lr", device: device) }
    public static func preparedReduceLROnPlateauClassifier(device: String = "auto") throws -> WorkflowDocument { try preparedSchedulerClassifier(componentID: "scheduler_reduce_on_plateau", device: device) }
    public static func preparedOneCycleLRClassifier(device: String = "auto") throws -> WorkflowDocument { try preparedSchedulerClassifier(componentID: "scheduler_one_cycle_lr", device: device) }
    public static func preparedLinearLRClassifier(device: String = "auto") throws -> WorkflowDocument { try preparedSchedulerClassifier(componentID: "scheduler_linear_lr", device: device) }
}
