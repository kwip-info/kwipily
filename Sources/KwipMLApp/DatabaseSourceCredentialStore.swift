import Foundation
import Security
import KwipMLCore

/// Private credentials are stored per project/node, never in portable source settings or worker artifacts.
struct DatabaseSourceCredentials: Codable, Equatable {
    var username: String
    var password: String
    func encoded() throws -> Data {
        guard [username, password].allSatisfy({ !$0.contains("\0") && $0.utf8.count <= 4096 }) else { throw DatabaseSourceCredentialFailure.invalidCredential }
        return try JSONEncoder().encode(self)
    }
}
enum DatabaseSourceCredentialFailure: Error { case invalidIdentity, invalidCredential, unavailable }
protocol DatabaseSourceCredentialStorage {
    func read(projectID: String, nodeID: String, connectionID: String) throws -> DatabaseSourceCredentials?
    func write(_ value: DatabaseSourceCredentials, projectID: String, nodeID: String, connectionID: String) throws
    func clear(projectID: String, nodeID: String, connectionID: String) throws
}
struct DatabaseSourceKeychain: DatabaseSourceCredentialStorage {
    /// The ad-hoc development app has no provisioning-profile Keychain group.
    /// Its login-Keychain item retains the default creator application ACL.
    static func storagePolicy(bundleIdentifier: String?) -> (service: String, dataProtection: Bool) {
        #if DEBUG
        if bundleIdentifier == "info.kwip.KwipML.POC" {
            return ("info.kwip.KwipML.POC.database-source", false)
        }
        #endif
        return (DesktopDeployment.keychainService("database-source"), true)
    }

    private func query(projectID: String, nodeID: String, connectionID: String) throws -> [String: Any] {
        guard UUID(uuidString: projectID)?.uuidString == projectID, !nodeID.isEmpty, nodeID.utf8.count <= 128, !nodeID.contains("\0"), connectionID.utf8.count == 64, connectionID.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }) else { throw DatabaseSourceCredentialFailure.invalidIdentity }
        let policy = Self.storagePolicy(bundleIdentifier: Bundle.main.bundleIdentifier)
        return [kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: policy.service,
            kSecAttrAccount as String: projectID + ":" + nodeID + ":" + connectionID,
            kSecUseDataProtectionKeychain as String: policy.dataProtection, kSecAttrSynchronizable as String: false]
    }
    func read(projectID: String, nodeID: String, connectionID: String) throws -> DatabaseSourceCredentials? {
        var q = try query(projectID: projectID, nodeID: nodeID, connectionID: connectionID)
        q[kSecReturnData as String] = true; q[kSecMatchLimit as String] = kSecMatchLimitOne
        var result: CFTypeRef?
        let status = SecItemCopyMatching(q as CFDictionary, &result)
        if status == errSecItemNotFound { return nil }
        guard status == errSecSuccess, let data = result as? Data, data.count <= 65_536,
              let value = try? JSONDecoder().decode(DatabaseSourceCredentials.self, from: data), (try? value.encoded()) != nil else { throw DatabaseSourceCredentialFailure.unavailable }
        return value
    }
    func write(_ value: DatabaseSourceCredentials, projectID: String, nodeID: String, connectionID: String) throws {
        let data = try value.encoded(), q = try query(projectID: projectID, nodeID: nodeID, connectionID: connectionID)
        let status = SecItemUpdate(q as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecItemNotFound {
            var add = q; add[kSecValueData as String] = data
            if Self.storagePolicy(bundleIdentifier: Bundle.main.bundleIdentifier).dataProtection { add[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly }
            guard SecItemAdd(add as CFDictionary, nil) == errSecSuccess else { throw DatabaseSourceCredentialFailure.unavailable }
        } else if status != errSecSuccess { throw DatabaseSourceCredentialFailure.unavailable }
    }
    func clear(projectID: String, nodeID: String, connectionID: String) throws {
        let status = SecItemDelete(try query(projectID: projectID, nodeID: nodeID, connectionID: connectionID) as CFDictionary)
        guard status == errSecSuccess || status == errSecItemNotFound else { throw DatabaseSourceCredentialFailure.unavailable }
    }
}
