import Foundation
import KwipMLCore

/// Ordered column identities remain stable while a mapping is edited.
struct LossTargetMappingDraft:Equatable {
    struct Role:Identifiable,Equatable {
        var name:String
        var columns:TrainingMappingDraft
        var sampleShape:[Int]
        var dtype:ContractDType
        var id:String {name}
    }
    let componentID:String
    var inputs:TrainingMappingDraft
    var roles:[Role]
    var groups:[TrainingMappingDraft]
    init(mapping:LossTargetMapping) {
        componentID=mapping.componentID
        inputs=TrainingMappingDraft(features:mapping.inputColumns,target:"")
        roles=mapping.roles.map {Role(name:$0.role,columns:TrainingMappingDraft(features:$0.columns,target:""),sampleShape:$0.sampleShape,dtype:$0.dtype)}
        groups=mapping.tripletSourceGroups.map {TrainingMappingDraft(features:$0,target:"")}
    }
    var mapping:LossTargetMapping {
        .init(componentID:componentID,inputColumns:groups.isEmpty ? inputs.features:groups.flatMap(\.features),
              roles:roles.map {.init(role:$0.name,columns:$0.columns.features,sampleShape:$0.sampleShape,dtype:$0.dtype)},tripletSourceGroups:groups.map(\.features))
    }
}
