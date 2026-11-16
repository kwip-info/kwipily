import Foundation
import CryptoKit

/// Verification never calls a server or compares the wall clock. Issued v1 rights
/// remain usable independently of account sessions, refunds and service availability.
public struct OfflineLicenseVerifier {
    public enum Failure: Error { case invalidGrant }
    public struct Grant: Equatable, Sendable {
        public let licenseID: String
        public let activationID: String
        public let installationDigest: String
        public let majorVersion: Int
    }
    private struct Envelope: Decodable {
        let keyID:String, payload:String, signature:String
        let schemaVersion:Int
    }
    private struct Claims: Decodable {
        let activationID:String, environment:String, installationDigest:String
        let issuedAt:Int64
        let licenseID:String
        let majorVersion:Int
        let product:String, rights:String
        var canonical:String {
            #"{"activationID":"\#(activationID)","environment":"\#(environment)","installationDigest":"\#(installationDigest)","issuedAt":\#(issuedAt),"licenseID":"\#(licenseID)","majorVersion":\#(majorVersion),"product":"\#(product)","rights":"\#(rights)"}"#
        }
    }
    private let environment:String
    private let keys:[String:Curve25519.Signing.PublicKey]

    public init(environment:String, publicKeys:[String:Data]) throws {
        guard ["staging","production"].contains(environment), (1...8).contains(publicKeys.count) else {throw Failure.invalidGrant}
        var keys=[String:Curve25519.Signing.PublicKey]()
        for (id,data) in publicKeys {
            guard Self.matches(id,"[a-z0-9][a-z0-9-]{0,47}"),data.count==32 else {throw Failure.invalidGrant}
            keys[id]=try Curve25519.Signing.PublicKey(rawRepresentation:data)
        }
        self.environment=environment;self.keys=keys
    }

    public func verify(_ data:Data, installationDigest:String) throws -> Grant {
        do {return try checked(data,installationDigest:installationDigest)}
        catch {throw Failure.invalidGrant}
    }
    private func checked(_ data:Data,installationDigest:String) throws -> Grant {
        guard (1...4096).contains(data.count),Self.matches(installationDigest,"[a-f0-9]{64}") else {throw Failure.invalidGrant}
        let value=try JSONDecoder().decode(Envelope.self,from:data)
        guard value.schemaVersion==1,Self.matches(value.keyID,"[a-z0-9][a-z0-9-]{0,47}"),let key=keys[value.keyID] else {throw Failure.invalidGrant}
        let payload=try Self.decode(value.payload,maximum:2048),signature=try Self.decode(value.signature,maximum:64)
        // Rebuilding the exact restricted format rejects unknown/duplicate fields,
        // alternate escapes, whitespace, booleans and noncanonical numbers.
        let canonical = #"{"keyID":"\#(value.keyID)","payload":"\#(value.payload)","schemaVersion":1,"signature":"\#(value.signature)"}"#
        guard Data(canonical.utf8)==data,signature.count==64,
              key.isValidSignature(signature,for:Data(("KWIPML-OFFLINE-LICENSE-V1\n"+value.keyID+"\n").utf8)+payload) else {throw Failure.invalidGrant}
        let claims=try JSONDecoder().decode(Claims.self,from:payload)
        guard claims.environment==environment,claims.product=="info.kwip.KwipML",claims.rights=="perpetual-v1",claims.majorVersion==1,
              (0...253402300799).contains(claims.issuedAt),Self.canonicalUUID(claims.licenseID),Self.canonicalUUID(claims.activationID),
              claims.installationDigest==installationDigest,Data(claims.canonical.utf8)==payload else {throw Failure.invalidGrant}
        return Grant(licenseID:claims.licenseID,activationID:claims.activationID,installationDigest:claims.installationDigest,majorVersion:1)
    }
    private static func canonicalUUID(_ value:String)->Bool {
        value.count==36 && UUID(uuidString:value)?.uuidString.lowercased()==value
    }
    private static func matches(_ value:String,_ pattern:String)->Bool {
        value.range(of:"\\A(?:"+pattern+")\\z",options:.regularExpression) != nil
    }
    private static func encode(_ value:Data)->String {
        value.base64EncodedString().replacingOccurrences(of:"+",with:"-").replacingOccurrences(of:"/",with:"_").replacingOccurrences(of:"=",with:"")
    }
    private static func decode(_ value:String,maximum:Int)throws->Data {
        guard value.utf8.count<=maximum*2,matches(value,"[A-Za-z0-9_-]+"),
              let bytes=Data(base64Encoded:value.replacingOccurrences(of:"-",with:"+").replacingOccurrences(of:"_",with:"/")+String(repeating:"=",count:(4-value.utf8.count%4)%4)),
              bytes.count<=maximum,encode(bytes)==value else {throw Failure.invalidGrant}
        return bytes
    }
}
