import Foundation
import CryptoKit
import Security
import CoreFoundation

/// Per-attempt secrets live in memory and never become worker arguments or preferences.
public struct DesktopAuthorization {
    public enum Failure:Error { case invalidResponse, entropyUnavailable }
    public let verifier:String
    public let state:String
    public init() throws {
        func random(_ count:Int)throws->String {
            var bytes=[UInt8](repeating:0,count:count)
            guard SecRandomCopyBytes(kSecRandomDefault,count,&bytes)==errSecSuccess else {throw Failure.entropyUnavailable}
            return Self.base64URL(Data(bytes))
        }
        verifier=try random(48);state=try random(32)
    }
    private static func base64URL(_ data:Data)->String {
        data.base64EncodedString().replacingOccurrences(of:"+",with:"-").replacingOccurrences(of:"/",with:"_").replacingOccurrences(of:"=",with:"")
    }
    public static func challenge(_ verifier:String)->String {base64URL(Data(SHA256.hash(data:Data(verifier.utf8))))}
    public var startURL:URL {
        var components=URLComponents(string:DesktopDeployment.origin + "/auth/desktop/start")!
        components.queryItems=[.init(name:"challenge",value:Self.challenge(verifier)),.init(name:"state",value:state)]
        return components.url!
    }
    public func code(from url:URL)throws->String {
        guard let value=URLComponents(url:url,resolvingAgainstBaseURL:false),value.scheme==DesktopDeployment.callbackScheme,value.host=="auth",value.path=="/callback",
              value.user==nil,value.password==nil,value.port==nil,value.fragment==nil,
              let items=value.queryItems,items.count==2,Set(items.map(\.name))==Set(["code","state"]),
              items.first(where:{$0.name=="state"})?.value==state,
              let code=items.first(where:{$0.name=="code"})?.value,Self.validToken(code) else {throw Failure.invalidResponse}
        return code
    }
    public static func validToken(_ token:String)->Bool {
        token.utf8.count==43 && token.utf8.allSatisfy { (65...90).contains($0)||(97...122).contains($0)||(48...57).contains($0)||$0==45||$0==95 }
    }
    private static func object(_ data:Data,keys:Set<String>)throws->[String:Any] {
        guard !data.isEmpty,data.count<=65_536,let obj=try JSONSerialization.jsonObject(with:data) as? [String:Any],Set(obj.keys)==keys,
              let version=obj["schemaVersion"] as? NSNumber,CFGetTypeID(version) != CFBooleanGetTypeID(),version==1 else {throw Failure.invalidResponse}
        return obj
    }
    public static func token(from data:Data)throws->String {
        let obj=try object(data,keys:["schemaVersion","token"])
        guard let token=obj["token"] as? String,validToken(token) else {throw Failure.invalidResponse};return token
    }
    public static func customer(from data:Data)throws->UUID {
        let obj=try object(data,keys:["schemaVersion","customerID"])
        guard let text=obj["customerID"] as? String,let id=UUID(uuidString:text) else {throw Failure.invalidResponse};return id
    }
}
