import Foundation

/// Build-time trust boundary. No preferences, process environment or broker message
/// can change these values in an installed app. Staging uses the existing POC identity.
public enum DesktopDeployment {
    #if KWIPML_COMMERCE_STAGING
    public static let environment = "staging"
    public static let origin = "http://127.0.0.1:8768"
    public static let callbackScheme = "kwipml-staging"
    public static let bundleIdentifier = "info.kwip.KwipML.POC"
    public static let isStaging = true
    #else
    public static let environment = "production"
    public static let origin = "https://apps.kwip.info"
    public static let callbackScheme = "kwipml"
    public static let bundleIdentifier = "info.kwip.KwipML"
    public static let isStaging = false
    #endif
    public static let applicationSupportDirectoryName = isStaging ? "KwipML Staging" : "KwipML"
    public static let networkServiceIdentifier = bundleIdentifier + ".NetworkService"
    public static func keychainService(_ purpose: String) -> String {
        "info.kwip.KwipML." + purpose + "." + environment + ".v1"
    }
    public static func isCommercialEdition(bundle: String?) -> Bool { bundle == bundleIdentifier }
}
