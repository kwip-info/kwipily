import Foundation

enum WorkflowEvaluationPresentation {
    static func explanations(_ metric:WorkflowEvaluationClassMetric,partitionTitle:String="Test")->[String] {
        var result:[String]=[]
        if metric.precisionUnavailableReason == "noPredictedObservations" {
            result.append("Precision is unavailable because no observations were predicted as class \(metric.classIndex).")
        }
        if metric.recallUnavailableReason == "noActualObservations" {
            result.append("Recall is unavailable because the \(partitionTitle.lowercased()) data contains no class \(metric.classIndex) observations.")
        }
        return result
    }
    static func confusionDescription(_ counts:[[Int]],partitionTitle:String="Test")->String {
        guard counts.count==2,counts.allSatisfy({$0.count==2}) else {return "\(partitionTitle) confusion counts unavailable."}
        return "\(partitionTitle) confusion counts. " + (0..<2).map { row in
            "Actual class \(row): predicted class 0, \(counts[row][0]); predicted class 1, \(counts[row][1])."
        }.joined(separator:" ")
    }
}
