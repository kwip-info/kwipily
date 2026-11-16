import Foundation
import Security
import OSLog
import KwipMLCore

@objc public protocol KwipMLDatabaseSourceProtocol {
    /// The descriptor must identify a newly created private capture, never a user destination.
    func capture(_ request: Data, credentials: Data, output: FileHandle, reply: @escaping (Data?, String?) -> Void)
    func databaseOutput(_ request:Data, credentials:Data, table:FileHandle?, directoryBookmark:Data?, reply:@escaping(Data?,String?)->Void)
    func cancelCapture(reply: @escaping (Bool) -> Void)
    #if DEBUG
    func debugProcessIDs(reply: @escaping (Int32, Int32) -> Void)
    func authorizeDevelopmentRoot(_ bookmark: Data, reply: @escaping (Bool) -> Void)
    #endif
}

public enum DatabaseSourceServiceIdentity {
    public static var identifier: String {
        #if DEBUG
        if let app = appBundleURL, let id = Bundle(url: app)?.bundleIdentifier { return id + ".DatabaseSourceService" }
        #endif
        return DesktopDeployment.bundleIdentifier + ".DatabaseSourceService"
    }
    private static var appBundleURL: URL? {
        var value = Bundle.main.bundleURL
        while value.pathExtension != "app", value.path != "/" { value.deleteLastPathComponent() }
        return value.pathExtension == "app" ? value : nil
    }
    public static func clientRequirement() throws -> String {
        #if DEBUG
        if let app = appBundleURL {
            let parentExecutable = Bundle.main.object(forInfoDictionaryKey: "KwipMLParentExecutable") as? String
            return try designatedRequirement(app, executableName: parentExecutable)
        }
        #endif
        return requirement(identifier: DesktopDeployment.bundleIdentifier)
    }
    public static func serviceRequirement() throws -> String {
        #if DEBUG
        if let app = appBundleURL { return try designatedRequirement(app.appendingPathComponent("Contents/XPCServices/DatabaseSourceService.xpc")) }
        #endif
        return requirement(identifier: identifier)
    }
    #if DEBUG
    /// Exact designated signature, never identifier-only, for a locally built embedded peer.
    public static func designatedRequirement(_ bundle: URL, executableName: String? = nil) throws -> String {
        let log = Logger(subsystem: "info.kwip.KwipML.DatabaseSource", category: "developmentTrust")
        log.debug("Reading designated requirement at \(bundle.path, privacy: .public)")
        let name: String?
        if let executableName { name = executableName }
        else {
            let infoURL = bundle.appendingPathComponent("Contents/Info.plist")
            if let data = try? Data(contentsOf: infoURL), data.count <= 65_536,
               let info = try? PropertyListSerialization.propertyList(from: data, format: nil) as? [String: Any] {
                name = info["CFBundleExecutable"] as? String
            } else { name = nil }
        }
        guard let name, !name.isEmpty,
              !name.contains("/"), !name.contains("\\"), name != ".", name != ".." else {
            log.error("Expected peer executable declaration unavailable at \(bundle.path, privacy: .public)")
            throw DatabaseSourceBrokerFailure.unavailable
        }
        let executable = bundle.appendingPathComponent("Contents/MacOS").appendingPathComponent(name)
        var code: SecStaticCode?
        // Inspect the peer's Mach-O signature directly, rather than asking the
        // nested sandbox to discover/validate the enclosing bundle envelope.
        let createStatus = SecStaticCodeCreateWithPath(executable as CFURL, SecCSFlags(), &code)
        guard createStatus == errSecSuccess, let code else {
            log.error("Static code creation failed status \(createStatus, privacy: .public)")
            throw DatabaseSourceBrokerFailure.unavailable
        }
        // Reading the designated requirement is not an authorization decision.
        // The sandboxed nested helper cannot validate the enclosing development
        // bundle's complete resource envelope. NSXPC verifies the actual peer
        // against the exact cryptographic requirement below before delivery.
        var requirement: SecRequirement?
        let copyStatus = SecCodeCopyDesignatedRequirement(code, SecCSFlags(), &requirement)
        guard copyStatus == errSecSuccess, let requirement else {
            log.error("Designated requirement extraction failed status \(copyStatus, privacy: .public)")
            throw DatabaseSourceBrokerFailure.unavailable
        }
        var value: CFString?
        guard SecRequirementCopyString(requirement, SecCSFlags(), &value) == errSecSuccess, let value else { throw DatabaseSourceBrokerFailure.unavailable }
        let text = value as String
        // Ad-hoc peers must carry the exact cdhash; other signatures must carry
        // their cryptographic anchor/certificate. A bare identifier cannot pass.
        guard text.contains("cdhash") || (text.contains("anchor") && text.contains("certificate")) else { throw DatabaseSourceBrokerFailure.unavailable }
        return text
    }
    #endif
    public static func requirement(identifier: String) -> String {
        "anchor apple generic and identifier \"\(identifier)\" and certificate leaf[subject.OU] = \"MT68X9Z6NQ\""
    }
    #if DEBUG
    /// One-use transferred bookmarks are never persisted or refreshed. Their
    /// stale bit is advisory; exact physical code-root identity is mandatory.
    /// The recipient must still separately prove read access and deny writes.
    public static func matchesDevelopmentRoot(_ resolved: URL, expected: URL, bookmarkIsStale _: Bool) -> Bool {
        resolved.standardizedFileURL.resolvingSymlinksInPath().path == expected.standardizedFileURL.resolvingSymlinksInPath().path
    }
    public static func developmentRootBookmark() throws -> Data? {
        guard let app = appBundleURL else { return nil }
        let pointer = app.appendingPathComponent("Contents/Resources/ProjectRoot.txt")
        guard let text = try? String(contentsOf: pointer, encoding: .utf8) else { return nil }
        let root = URL(fileURLWithPath: text.trimmingCharacters(in: .whitespacesAndNewlines), isDirectory: true)
        // Identity only: no implicit or persistent sandbox extension crosses
        // XPC. Debug assembly provides finite read-only runtime-code grants.
        return try root.bookmarkData(options: [.withoutImplicitSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil)
    }
    #endif
    public static func request(options: [String: JSONValue], preview: Bool, memoryLimit:Int?=nil, operationSeconds:Int?=nil) throws -> Data {
        struct Request: Codable { let schemaVersion: Int; let options: [String: JSONValue]; let preview: Bool }
        let normalized = try options["kind"] == .string("api") ? APITableSettings.normalized(options):DatabaseSourceSettings.normalized(options)
        guard normalized["backend"] != .string("sqlite") else { throw DatabaseSourceBrokerFailure.invalidRequest }
        var object:[String:JSONValue]=["schemaVersion":1,"options":.object(normalized),"preview":.bool(preview)]
        if normalized["kind"] == .string("api") {guard normalized["direction"] == .string("input") else {throw DatabaseSourceBrokerFailure.invalidRequest};object["memoryLimit"] = memoryLimit.map { .integer(Int64($0)) } ?? .null;object["operationSeconds"] = .integer(Int64(operationSeconds ?? 300))}
        let data = try JSONEncoder().encode(object)
        guard data.count <= 32_768 else { throw DatabaseSourceBrokerFailure.invalidRequest }
        return data
    }
    public static func validate(_ message: Data) throws -> ([String: JSONValue], Bool) {
        guard message.count <= 32_768, case .object(let value) = try JSONDecoder().decode(JSONValue.self, from: message),
              (Set(value.keys) == ["schemaVersion", "options", "preview"] || Set(value.keys) == ["schemaVersion", "options", "preview","memoryLimit","operationSeconds"]), value["schemaVersion"] == .integer(1),
              case .object(let config) = value["options"], case .bool(let preview) = value["preview"] else { throw DatabaseSourceBrokerFailure.invalidRequest }
        let normalized = try config["kind"] == .string("api") ? APITableSettings.normalized(config):DatabaseSourceSettings.normalized(config)
        guard normalized["backend"] != .string("sqlite") else { throw DatabaseSourceBrokerFailure.invalidRequest }
        return (normalized, preview)
    }
}
public enum DatabaseSourceBrokerFailure: Error { case invalidRequest, unavailable, captureFailed, cancelled }
