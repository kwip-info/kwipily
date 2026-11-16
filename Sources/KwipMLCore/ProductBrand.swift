import Foundation

/// Human-facing product naming only. Storage, signing and wire identities belong to
/// DesktopDeployment and their versioned contracts; branding never migrates them.
public enum ProductBrand {
    private struct Definition: Decodable {
        let name: String
        let formerName: String
        let publisher: String
        let tagline: String
        let businessTagline: String
    }
    private static let definition: Definition = {
        guard let url = Bundle.module.url(forResource: "ProductBrand", withExtension: "json"),
              let data = try? Data(contentsOf: url),
              let value = try? JSONDecoder().decode(Definition.self, from: data) else {
            preconditionFailure("Missing product presentation resource")
        }
        return value
    }()
    /// Isolated helpers do not have access to the enclosing app's resource bundle.
    /// Their signed Info.plist is populated from the same presentation resource.
    public static func serviceName(in info: [String: Any]? = Bundle.main.infoDictionary) -> String {
        guard let value=info?["KwipMLProductName"] as? String,
              !value.isEmpty, value.utf8.count <= 120, !value.contains("\n") else { return "Local workflow service" }
        return value
    }
    public static var name: String { definition.name }
    public static var formerName: String { definition.formerName }
    public static var publisher: String { definition.publisher }
    public static var tagline: String { definition.tagline }
    public static var businessTagline: String { definition.businessTagline }
    public static var transitionMessage: String {
        "\(formerName) is now \(name). Your projects and existing v1 license stay with you."
    }
}
