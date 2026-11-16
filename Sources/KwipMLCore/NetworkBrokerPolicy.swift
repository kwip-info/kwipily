import Foundation
import CoreFoundation

/// The free app has no commerce operations. Keep the bounded health probe for
/// packaged network isolation diagnostics; provider access uses its separate API.
public enum NetworkBrokerPolicy {
    public static let maximumRequestBytes = 16_384
    public static let maximumResponseBytes = 65_536
    public enum Refusal: Error { case invalidMessage }
    public static func prepare(_ data: Data) throws -> URLRequest {
        guard !data.isEmpty, data.count <= maximumRequestBytes,
              let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              Set(object.keys) == ["schemaVersion", "operation"],
              let version = object["schemaVersion"] as? NSNumber,
              CFGetTypeID(version) != CFBooleanGetTypeID(), version == 1,
              object["operation"] as? String == "health" else { throw Refusal.invalidMessage }
        var request = URLRequest(url: URL(string: DesktopDeployment.origin + "/api/v1/health")!, timeoutInterval: 20)
        request.httpMethod = "GET"
        request.httpShouldHandleCookies = false
        request.setValue("application/json", forHTTPHeaderField: "Accept")
        return request
    }
}
