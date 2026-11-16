import Foundation
import CryptoKit
import XCTest
@testable import KwipMLCore

final class OfflineLicenseTests: XCTestCase {
    // Synthetic seed bytes1...32; never a production/staging release key.
    private let publicHex = "79b5562e8fe654f94078b112e8a98ba7901f853ae695bed7e0e3910bad049664"
    private let fixture = #"{"keyID":"fixture-2026","payload":"eyJhY3RpdmF0aW9uSUQiOiIwMDAwMDAwMC0wMDAwLTAwMDAtMDAwMC0wMDAwMDAwMDAwMDEiLCJlbnZpcm9ubWVudCI6InN0YWdpbmciLCJpbnN0YWxsYXRpb25EaWdlc3QiOiJhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhYWFhIiwiaXNzdWVkQXQiOjE4MDAwMDAwMDAsImxpY2Vuc2VJRCI6IjAwMDAwMDAwLTAwMDAtMDAwMC0wMDAwLTAwMDAwMDAwMDAwMiIsIm1ham9yVmVyc2lvbiI6MSwicHJvZHVjdCI6ImluZm8ua3dpcC5Ld2lwTUwiLCJyaWdodHMiOiJwZXJwZXR1YWwtdjEifQ","schemaVersion":1,"signature":"Nw974QkNfRQtjboMChR3-am3ZiYL9qWcxZ_6OWK_BRQXnZCIV_gXL_O3u8xF27g4tojakfqhfoDA6o7R7vGJBg"}"#
    private var publicKey:Data {Data(stride(from:0,to:publicHex.count,by:2).map {n in
        let start=publicHex.index(publicHex.startIndex,offsetBy:n)
        return UInt8(publicHex[start..<publicHex.index(start,offsetBy:2)],radix:16)!
    })}
    private func verifier(_ environment:String = "staging") throws -> OfflineLicenseVerifier {
        try OfflineLicenseVerifier(environment:environment,publicKeys:["fixture-2026":publicKey])
    }
    func testPythonIssuedVectorVerifiesWithoutClockOrNetwork() throws {
        let grant=try verifier().verify(Data(fixture.utf8),installationDigest:String(repeating:"a",count:64))
        XCTAssertEqual(grant.licenseID,"00000000-0000-0000-0000-000000000002")
        XCTAssertEqual(grant.activationID,"00000000-0000-0000-0000-000000000001")
        XCTAssertEqual(grant.majorVersion,1)
    }
    func testWrongEnvironmentDeviceKeyAndTamperRefused() throws {
        let value=Data(fixture.utf8),digest=String(repeating:"a",count:64)
        XCTAssertThrowsError(try verifier("production").verify(value,installationDigest:digest))
        XCTAssertThrowsError(try verifier().verify(value,installationDigest:String(repeating:"b",count:64)))
        XCTAssertThrowsError(try OfflineLicenseVerifier(environment:"staging",publicKeys:["other":publicKey]).verify(value,installationDigest:digest))
        for bad in [fixture.replacingOccurrences(of:"fixture-2026",with:"fixture-2027"), fixture.replacingOccurrences(of:"\"schemaVersion\":1",with:"\"schemaVersion\":2"), " "+fixture, "[]", String(repeating:"a",count:4097)] {
            XCTAssertThrowsError(try verifier().verify(Data(bad.utf8),installationDigest:digest))
        }
    }
    func testRotationKeepsOldKeyAndRejectsUnknownPolicyWithValidSignature() throws {
        let replacement=Curve25519.Signing.PrivateKey()
        let ring=try OfflineLicenseVerifier(environment:"staging",publicKeys:["fixture-2026":publicKey,"fixture-2027":replacement.publicKey.rawRepresentation])
        XCTAssertNoThrow(try ring.verify(Data(fixture.utf8),installationDigest:String(repeating:"a",count:64)))
        var obj=try JSONSerialization.jsonObject(with:Data(fixture.utf8)) as! [String:Any]
        let encoded=obj["payload"] as! String
        let padded=encoded.replacingOccurrences(of:"-",with:"+").replacingOccurrences(of:"_",with:"/")+String(repeating:"=",count:(4-encoded.count%4)%4)
        let payload=String(data:Data(base64Encoded:padded)!,encoding:.utf8)!
        for bad in [payload.replacingOccurrences(of:"\"majorVersion\":1",with:"\"majorVersion\":2"),payload.replacingOccurrences(of:"perpetual-v1",with:"subscription"),payload.replacingOccurrences(of:"\"issuedAt\":1800000000",with:"\"issuedAt\":true"),payload.replacingOccurrences(of:"{",with:"{\"extra\":1,"),payload.replacingOccurrences(of:"{",with:"{\"majorVersion\":1,")] {
            func b64(_ d:Data)->String {d.base64EncodedString().replacingOccurrences(of:"+",with:"-").replacingOccurrences(of:"/",with:"_").replacingOccurrences(of:"=",with:"")}
            obj["keyID"]="fixture-2027";obj["payload"]=b64(Data(bad.utf8))
            obj["signature"]=b64(try replacement.signature(for:Data(("KWIPML-OFFLINE-LICENSE-V1\nfixture-2027\n"+bad).utf8)))
            let wire=try JSONSerialization.data(withJSONObject:obj,options:[.sortedKeys,.withoutEscapingSlashes])
            XCTAssertThrowsError(try ring.verify(wire,installationDigest:String(repeating:"a",count:64)))
        }
    }
}
