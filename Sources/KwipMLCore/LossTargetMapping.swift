import Foundation

/// Structural table bindings; objective-specific value checks belong to execution.
public struct LossTargetRoleMapping: Codable, Equatable {
    public var role: String
    public var columns: [String]
    public var sampleShape: [Int]
    public var dtype: ContractDType
    public init(role: String, columns: [String], sampleShape: [Int], dtype: ContractDType) {
        self.role=role; self.columns=columns; self.sampleShape=sampleShape; self.dtype=dtype
    }
}

public struct LossTargetMapping: Codable, Equatable {
    public var schemaVersion: Int
    public var componentID: String
    public var inputColumns: [String]
    public var roles: [LossTargetRoleMapping]
    public var tripletSourceGroups: [[String]]
    public init(componentID: String, inputColumns: [String], roles: [LossTargetRoleMapping], tripletSourceGroups: [[String]] = []) {
        schemaVersion=1; self.componentID=componentID; self.inputColumns=inputColumns
        self.roles=roles; self.tripletSourceGroups=tripletSourceGroups
    }
    public func validate() throws {
        let supported:Set<String>=["loss_cross_entropy","loss_bce_with_logits","loss_mse","loss_l1","loss_nll","loss_kl_div","loss_ctc","loss_triplet_margin"]
        guard [1,2].contains(schemaVersion),supported.contains(componentID) else {throw GraphConfigurationError("Unknown loss target mapping version or objective.")}
        func validColumns(_ columns:[String])->Bool {
            (1...TrainingColumnMapping.maximumColumns).contains(columns.count)
                && Set(columns).count == columns.count
                && columns.allSatisfy {!$0.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty && $0.count<=200}
        }
        guard validColumns(inputColumns) else {throw GraphConfigurationError("Select distinct, ordered input columns with nonblank names.")}
        if schemaVersion==2 {
            guard componentID=="loss_cross_entropy",roles.count==1,roles[0].role=="target",roles[0].dtype == .int64,roles[0].columns.count==1,validColumns(roles[0].columns),Set(inputColumns).isDisjoint(with:roles[0].columns),roles[0].sampleShape.count==1,(1...4096).contains(roles[0].sampleShape[0]),tripletSourceGroups.isEmpty else {throw GraphConfigurationError("Sequence targets require one integer sequence column and exact context length.")}
            return
        }
        if componentID == "loss_triplet_margin" {
            guard roles.isEmpty,tripletSourceGroups.count == 3,
                  tripletSourceGroups.allSatisfy({validColumns($0) && $0.count == tripletSourceGroups[0].count}),
                  tripletSourceGroups.flatMap({$0}) == inputColumns else {
                throw GraphConfigurationError("Triplet loss needs three equally sized source groups in anchor, positive, negative order, without a dummy target column.")
            }
            return
        }
        guard tripletSourceGroups.isEmpty,Set(roles.map(\.role)).count == roles.count else {throw GraphConfigurationError("Target roles must be unique; triplet source groups belong only to Triplet loss.")}
        let required:Set<String> = componentID == "loss_ctc" ? ["target","input_lengths","target_lengths"]:["target"]
        let optional:Set<String> = componentID == "loss_bce_with_logits" ? ["weight","pos_weight"]:[]
        let names=Set(roles.map(\.role))
        guard required.isSubset(of:names),names.isSubset(of:required.union(optional)) else {throw GraphConfigurationError("Connect the objective's exact target and optional weight roles.")}
        var used=Set(inputColumns)
        for role in roles {
            guard validColumns(role.columns),used.isDisjoint(with:role.columns),role.sampleShape.count<=8 else {throw GraphConfigurationError("Each target role needs separate ordered columns and a bounded sample shape.")}
            var count=1
            for dimension in role.sampleShape {
                guard dimension>0,dimension<=TrainingColumnMapping.maximumColumns,count<=TrainingColumnMapping.maximumColumns/dimension else {throw GraphConfigurationError("Target shape is invalid or exceeds the table column limit.")}
                count *= dimension
            }
            guard count == role.columns.count else {throw GraphConfigurationError("Target columns must fill the declared sample shape in row-major order.")}
            if componentID == "loss_ctc" {
                guard role.dtype == .int64,(role.role == "target" ? role.sampleShape.count == 1:role.sampleShape.isEmpty) else {throw GraphConfigurationError("Sequence targets are integer rows; source and target lengths are integer scalars.")}
            } else if componentID == "loss_nll" {
                guard role.dtype == .int64 else {throw GraphConfigurationError("Negative log likelihood requires integer class targets.")}
            } else if componentID == "loss_cross_entropy" {
                guard role.dtype == .int64 || (role.dtype == .float32 && !role.sampleShape.isEmpty) else {throw GraphConfigurationError("Cross entropy targets must be integer classes or explicitly shaped floating-point class shares.")}
            } else {
                guard role.dtype == .float32 else {throw GraphConfigurationError("This objective requires explicit floating-point target or weight columns.")}
            }
            used.formUnion(role.columns)
        }
    }
}
