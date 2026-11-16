import Foundation
import Security
import KwipMLCore

/// Secrets never enter graph, project sidecars, logs, History or MCP catalog data.
struct ModelHubKeychain {
    var copy:(CFDictionary,UnsafeMutablePointer<CFTypeRef?>?)->OSStatus = SecItemCopyMatching
    var add:(CFDictionary,UnsafeMutablePointer<CFTypeRef?>?)->OSStatus = SecItemAdd
    var update:(CFDictionary,CFDictionary)->OSStatus = SecItemUpdate
    var delete:(CFDictionary)->OSStatus = SecItemDelete
}

enum ModelHubCredentials {
    private static let service=DesktopDeployment.keychainService("hugging-face")
    static func read(_ name:String)throws->String {
        guard !name.isEmpty,name.utf8.count<=100 else {throw WorkflowArchiveError("Choose a saved Hub credential.")}
        let query:[String:Any]=[kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:name,kSecReturnData as String:true,kSecMatchLimit as String:kSecMatchLimitOne,kSecUseDataProtectionKeychain as String:true]
        var result:CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary,&result)==errSecSuccess,let data=result as? Data,let value=String(data:data,encoding:.utf8),value.hasPrefix("hf_") else {throw WorkflowArchiveError("Hub credential is unavailable. Add it using Manage credential.")}
        return value
    }
    static func save(name:String,token:String,using keychain:ModelHubKeychain = .init())throws {
        guard !name.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,name.utf8.count<=100,token.hasPrefix("hf_"),token.utf8.count<=4096,token.unicodeScalars.allSatisfy({$0.value>=33 && $0.value<=126}) else {throw WorkflowArchiveError("Enter a credential name and valid Hugging Face access token.")}
        let query:[String:Any]=[kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:name,kSecUseDataProtectionKeychain as String:true]
        var item=query;item[kSecValueData as String]=Data(token.utf8);item[kSecAttrAccessible as String]=kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly
        let status=keychain.add(item as CFDictionary,nil)
        guard status==errSecSuccess || (status==errSecDuplicateItem && keychain.update(query as CFDictionary,[kSecValueData as String:Data(token.utf8)] as CFDictionary)==errSecSuccess) else {throw WorkflowArchiveError("Could not store the Hub credential in Keychain.")}
    }
    /// Attribute-only enumeration: no credential data is requested or returned.
    static func names(using keychain:ModelHubKeychain = .init())throws->[String] {
        let query:[String:Any]=[kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecReturnAttributes as String:true,kSecMatchLimit as String:kSecMatchLimitAll,kSecUseDataProtectionKeychain as String:true]
        var result:CFTypeRef?
        let status=keychain.copy(query as CFDictionary,&result)
        if status==errSecItemNotFound {return []}
        guard status==errSecSuccess,let rows=result as? [[String:Any]],rows.count<=1000 else {throw WorkflowArchiveError("Could not list this app’s Hub credentials.")}
        return rows.compactMap{$0[kSecAttrAccount as String] as? String}.sorted()
    }
    static func remove(name:String,using keychain:ModelHubKeychain = .init())throws {
        guard !name.isEmpty,name.utf8.count<=100 else {throw WorkflowArchiveError("Choose one saved Hub credential.")}
        let query:[String:Any]=[kSecClass as String:kSecClassGenericPassword,kSecAttrService as String:service,kSecAttrAccount as String:name,kSecUseDataProtectionKeychain as String:true]
        let status=keychain.delete(query as CFDictionary)
        guard status==errSecSuccess || status==errSecItemNotFound else {throw WorkflowArchiveError("Could not remove that Hub credential.")}
    }

}
