import KwipMLCore

/// Runnable worked examples open their own managed lesson file. The real checker creates
/// and reopens the supplied parent; no fake checkpoint or developer path is bundled.
struct CheckpointLearningExample:Identifiable {
    let id:String
    let task:CheckpointLearningTask
    var title:String {task.title+" · worked example"}
    var configuration:CheckpointLearningConfiguration {task.configuration}
}
enum CheckpointLearningExamples {
    static let all:[CheckpointLearningExample]=[
        .init(id:"prepared-checkpoint-restore-weights",task:.restoreWeights),
        .init(id:"prepared-checkpoint-continue-training",task:.continueTraining)
    ]
    static func example(id:String)->CheckpointLearningExample? {all.first {$0.id == id}}
}
