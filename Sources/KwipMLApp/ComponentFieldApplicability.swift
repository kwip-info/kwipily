import KwipMLCore

/// Inactive controls keep their authored values so toggling a dependency never erases a draft.
enum ComponentFieldApplicability {
    static func inactiveReason(componentID: String, field: String, values: [String: JSONValue]) -> String? {
        guard StatefulNormalizationConfiguration.componentIDs.contains(componentID), let configuration = try? StatefulNormalizationConfiguration(componentID: componentID, values: values) else { return nil }
        if field == "bias", !configuration.affine { return "Inactive while learned scale and shift are off. Your shift setting is retained." }
        if field == "averaging", !configuration.tracksRunningStatistics { return "Inactive while running statistics are off. Your averaging setting is retained." }
        if field == "momentum" {
            if !configuration.tracksRunningStatistics { return "Inactive while running statistics are off. Your update weight is retained." }
            if configuration.averaging == "cumulative" { return "Cumulative averaging gives each training group equal weight. Your moving update weight is retained." }
        }
        return nil
    }
}
