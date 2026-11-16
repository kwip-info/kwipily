import Foundation
import KwipMLCore

/// Required model-selection provenance, retained even when optional run detail is off.
struct TrainingControlReport:Codable,Equatable {
    let schemaVersion:Int
    let policy:TrainingControlPolicy
    let completedEpoch:Int
    let selectedEpoch:Int
    let validationLoss:Double
    let stopReason:String
    static func verify(_ raw:JSONValue?,policy:TrainingControlPolicy?,epochLimit:Int,losses:[Double]?=nil)throws->Self? {
        guard let policy else {guard raw==nil else {throw WorkflowPhaseError("Unselected training controls cannot acquire provenance.")};return nil}
        guard let raw,case .object(let fields)=raw,Set(fields.keys)==["schemaVersion","policy","completedEpoch","selectedEpoch","validationLoss","stopReason"] else {throw WorkflowPhaseError("Missing or malformed model-selection report.")}
        let report=try JSONDecoder().decode(Self.self,from:JSONEncoder().encode(raw))
        guard report.schemaVersion==1,report.policy==policy,(1...epochLimit).contains(report.completedEpoch),(1...report.completedEpoch).contains(report.selectedEpoch),report.validationLoss.isFinite,["epochLimit","earlyStopping"].contains(report.stopReason),report.stopReason != "earlyStopping" || policy.earlyStopping.enabled,report.stopReason != "epochLimit" || report.completedEpoch==epochLimit,policy.modelSelection != "finalEpoch" || report.selectedEpoch==report.completedEpoch else {throw WorkflowPhaseError("Training outcome differs from its accepted selection/stopping policy.")}
        if let losses {
            guard losses.count==report.completedEpoch,losses.allSatisfy(\.isFinite) else {throw WorkflowPhaseError("Training policy does not match the completed validation ledger.")}
            var best=Double.infinity,bestEpoch=0,reference=Double.infinity,waiting=0,stopEpoch:Int?
            for (index,value) in losses.enumerated() {
                if value<best {best=value;bestEpoch=index+1}
                if value<reference-policy.earlyStopping.minimumImprovement {reference=value;waiting=0} else {waiting+=1}
                if policy.earlyStopping.enabled,waiting>=policy.earlyStopping.patience,stopEpoch==nil {stopEpoch=index+1}
            }
            let selected=policy.modelSelection=="finalEpoch" ? losses.count:bestEpoch
            guard report.selectedEpoch==selected,report.validationLoss==losses[selected-1],report.stopReason=="earlyStopping" ? stopEpoch==report.completedEpoch:stopEpoch==nil else {throw WorkflowPhaseError("Selected model or stopping epoch contradicts the measured validation history.")}
        }
        return report
    }
}
