import Foundation

/// Untrusted account/activation responses are separate from signed offline authority.
public enum DesktopLicenseWire {
    public enum Failure:Error {case invalidResponse}
    public struct Device:Decodable,Equatable,Sendable {
        public let activationID:String
        public let name:String
        public let active:Bool
    }
    public struct License:Decodable,Equatable,Sendable {
        public let licenseID:String
        public let majorVersion:Int
        public let perpetual:Bool
        public let updates:String
        public let maximumDevices:Int
        public let state:String
        public let activations:[Device]
    }
    public struct Page:Decodable,Sendable {
        public let schemaVersion:Int
        public let nextCursor:String?
        public let licenses:[License]
    }
    private struct Activation:Decodable {
        let schemaVersion:Int
        let activationID:String,licenseID:String,installationDigest:String,offlineLicense:String
    }
    public static func licenses(_ data:Data)throws->Page {
        guard (1...65_536).contains(data.count) else {throw Failure.invalidResponse}
        let page=try JSONDecoder().decode(Page.self,from:data)
        guard page.schemaVersion==1,page.licenses.count<=20,page.nextCursor.map(validID) ?? true,
              Set(page.licenses.map(\.licenseID)).count==page.licenses.count else {throw Failure.invalidResponse}
        for license in page.licenses {
            guard validID(license.licenseID),license.majorVersion==1,license.perpetual,license.updates=="v1.x",license.maximumDevices==2,
                  ["settled","refunded","disputed"].contains(license.state),license.activations.count<=2,
                  Set(license.activations.map(\.activationID)).count==license.activations.count else {throw Failure.invalidResponse}
            for device in license.activations {
                guard validID(device.activationID),device.active,!device.name.isEmpty,device.name.unicodeScalars.count<=80,
                      device.name.unicodeScalars.allSatisfy({!CharacterSet.controlCharacters.contains($0)}) else {throw Failure.invalidResponse}
            }
        }
        return page
    }
    public static func activation(_ data:Data,licenseID:String,installationDigest:String,verifier:OfflineLicenseVerifier)throws->(Data,OfflineLicenseVerifier.Grant) {
        guard (1...16_384).contains(data.count) else {throw Failure.invalidResponse}
        let response=try JSONDecoder().decode(Activation.self,from:data)
        guard response.schemaVersion==1,response.licenseID==licenseID,response.installationDigest==installationDigest,response.offlineLicense.utf8.count<=4096 else {throw Failure.invalidResponse}
        let bytes=Data(response.offlineLicense.utf8),grant=try verifier.verify(bytes,installationDigest:installationDigest)
        guard response.activationID==grant.activationID,response.licenseID==grant.licenseID else {throw Failure.invalidResponse}
        return (bytes,grant)
    }
    public static func validID(_ value:String)->Bool {value.count==36 && UUID(uuidString:value)?.uuidString.lowercased()==value}
}
