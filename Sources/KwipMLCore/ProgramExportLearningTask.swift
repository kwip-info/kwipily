import Foundation

public enum ProgramExportLearningTask:String,CaseIterable,Codable {
    case computation="program-export-computation",sizeRange="program-export-size-range"
    public var title:String {self == .computation ? "Save the computation":"Allow a safe input-size range"}
    public var controlInstructions:String {self == .computation ? "Choose Static batch and set example, minimum and maximum batch sizes to 2. Keep CPU, Strict on and Deferred runtime checks off.":"Choose Bounded batch. Set Example batch size to 2, Minimum to 1 and Maximum to 4. Keep CPU, Strict on and Deferred runtime checks off."}
    public static let captureControlExplanation="Strict capture checks assumptions while tracing the model’s calculation. Turning it off does not make every unsupported calculation exportable. The Deferred runtime checks option prefers checking eligible size rules inside the saved calculation when it runs. Off keeps the exporter’s normal checking policy, including any later input checks."
    public var instructions:String {
        let meaning=self == .computation ? "An executable program saves the calculation together with the learned settings it uses. Saving weights alone does not save that calculation." : "A batch is a group of observations. Bounded export accepts a declared range of group sizes while keeping two input values per observation."
        return "**Goal**\n"+meaning+" The checker supplies the selected saved model and real example rows.\n\n**Capture controls explained**\n"+Self.captureControlExplanation+" This exercise keeps Strict on and Deferred runtime checks off for the checked capture path and normal size-check policy.\n\n**Configure**\n"+controlInstructions+" Apply your changes.\n\n**Check**\nChoose Check construction. Compare the reloaded results, requested capture policy and accepted or rejected group sizes. A check value of 1 means the actual operation passed. Zero parent difference means the saved model remained unchanged. Private assessment keeps its test rows and reference results hidden."
    }
    public var values:[String:JSONValue] {
        var values=ProgramExportConfiguration.defaults
        if self == .sizeRange {values["batch_mode"] = .string("bounded");values["minimum_batch_size"]=1;values["maximum_batch_size"]=4}
        return values
    }
    public var componentID:String {"program_export"}
}
public enum ProgramExportLearningExamples {
    public static func submission(task:ProgramExportLearningTask)throws->ProgramExportLearningSubmission {try .init(task:task,values:task.values)}
}
