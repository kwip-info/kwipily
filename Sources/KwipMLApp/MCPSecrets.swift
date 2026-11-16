import Foundation
import Security
import KwipMLMCP

protocol MCPSecretStorage {
    func read(clientID:String)throws->String?
    func write(_ token:String,clientID:String)throws
    func remove(clientID:String)throws
}
struct MCPSecretKeychain:MCPSecretStorage {
    let namespace:String
    private func query(_ id:String)->[String:Any]{[
        kSecClass as String:kSecClassGenericPassword,
        kSecAttrService as String:"info.kwip.KwipML.mcp."+namespace+".v1",
        kSecAttrAccount as String:id,kSecAttrSynchronizable as String:false]}
    func read(clientID:String)throws->String? {
        var q=query(clientID);q[kSecReturnData as String]=true;q[kSecMatchLimit as String]=kSecMatchLimitOne
        var found:CFTypeRef?;let status=SecItemCopyMatching(q as CFDictionary,&found)
        if status==errSecItemNotFound{return nil}
        guard status==errSecSuccess,let bytes=found as? Data,bytes.count<=128,let token=String(data:bytes,encoding:.utf8)else{throw MCPSecretFailure.unavailable};return token
    }
    func write(_ token:String,clientID:String)throws {
        guard (32...128).contains(token.utf8.count)else{throw MCPSecretFailure.unavailable}
        let attributes=[kSecValueData as String:Data(token.utf8)]
        let status=SecItemUpdate(query(clientID) as CFDictionary,attributes as CFDictionary)
        if status==errSecItemNotFound {
            var add=query(clientID);add[kSecValueData as String]=Data(token.utf8);add[kSecAttrAccessible as String]=kSecAttrAccessibleWhenUnlockedThisDeviceOnly
            guard SecItemAdd(add as CFDictionary,nil)==errSecSuccess else{throw MCPSecretFailure.unavailable}
        }else if status != errSecSuccess{throw MCPSecretFailure.unavailable}
    }
    func remove(clientID:String)throws {let status=SecItemDelete(query(clientID) as CFDictionary);guard status==errSecSuccess || status==errSecItemNotFound else{throw MCPSecretFailure.unavailable}}
    static func makeToken()throws->String {
        var bytes=[UInt8](repeating:0,count:32)
        guard SecRandomCopyBytes(kSecRandomDefault,bytes.count,&bytes)==errSecSuccess else{throw MCPSecretFailure.unavailable}
        return Data(bytes).base64EncodedString().replacingOccurrences(of:"+",with:"-").replacingOccurrences(of:"/",with:"_").replacingOccurrences(of:"=",with:"")
    }
}
enum MCPSecretFailure:Error,LocalizedError {
    case unavailable
    var errorDescription:String?{"MCP client credentials are unavailable. Unlock this Mac or create a new client credential."}
}
