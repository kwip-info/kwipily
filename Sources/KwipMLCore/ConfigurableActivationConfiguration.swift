import Foundation

/// Graph values have independent connector ownership. Storage-mutating APIs are
/// not part of these activation contracts; numerical coefficients remain editable.
public struct ConfigurableActivationConfiguration: Equatable {
    public static let componentIDs: Set<String> = ["silu", "leaky_relu", "elu"]
    public static let coefficientMaximum = 3.4028234663852886e38
    public let values: [String: JSONValue]

    public init(componentID: String, values: [String: JSONValue]) throws {
        guard Self.componentIDs.contains(componentID) else {
            throw GraphConfigurationError("Unknown configurable activation: \(componentID).")
        }
        let key: String? = componentID == "silu" ? nil : componentID == "leaky_relu" ? "negative_slope" : "alpha"
        let allowed: Set<String> = key.map { [$0] } ?? []
        guard Set(values.keys).isSubset(of: allowed) else {
            throw GraphConfigurationError("\(componentID) accepts only its declared coefficient. Connector inputs are preserved; in-place mutation is unsupported.")
        }
        guard let key else { self.values = [:]; return }
        let value = values[key] ?? .number(componentID == "leaky_relu" ? 0.01 : 1)
        guard let number = value.doubleValue, abs(number) <= Self.coefficientMaximum else {
            throw GraphConfigurationError("\(key) must be a finite signed float32 number; booleans and overflow are not coefficients.")
        }
        self.values = [key: .number(number)]
    }
}
