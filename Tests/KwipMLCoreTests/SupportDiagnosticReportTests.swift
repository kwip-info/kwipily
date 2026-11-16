import XCTest
@testable import KwipMLCore

final class SupportDiagnosticReportTests: XCTestCase {
    private func report(version: String? = "1.0.0", build: String? = "10", outcome: UpdateDiagnosticOutcome = .ready) -> SupportDiagnosticReport {
        SupportDiagnosticReport(version: version, build: build, os: .init(majorVersion: 26, minorVersion: 6, patchVersion: 0),
            architecture: .arm64, edition: .direct, updaterAvailable: true, automaticChecks: true, updateOutcome: outcome)
    }
    func testAllowlistedSnapshotIsBoundedAndOmitsRetiredAccountState() throws {
        let data = try report().encoded()
        XCTAssertLessThan(data.count, 4096)
        let value = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(value.keys), Set(["schemaVersion", "version", "build", "macOS", "architecture", "edition", "updaterAvailable", "automaticChecks", "updateOutcome"]))
        XCTAssertNil(value["offlineLicenseLoaded"])
        XCTAssertEqual(value["schemaVersion"] as? Int, 2)
        XCTAssertEqual(value["version"] as? String, "1.0.0")
        XCTAssertEqual(value["build"] as? String, "10")
    }
    func testHostileMetadataNeverLeaksThroughReport() throws {
        let secret = "Bearer secret /Users/private@example.com\nhttps://evil.example"
        for value in [secret, String(repeating: "9", count: 100_000), "١.٢.٣", "1.0.0-private", "1\n", ""] {
            let data = try report(version: value, build: value).encoded()
            let text = try XCTUnwrap(String(data: data, encoding: .utf8))
            XCTAssertFalse(text.contains(secret)); XCTAssertFalse(text.contains("evil.example"))
            let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            XCTAssertEqual(json["version"] as? String, "unknown")
            XCTAssertEqual(json["build"] as? String, "unknown")
            XCTAssertLessThan(data.count, 4096)
        }
    }
    func testRawErrorsBecomeOnlyKnownCategoriesWithoutFollowingUnderlyingErrors() throws {
        let privateText = "Bearer token /Users/customer/project.csv"
        let error = NSError(domain: "SUSparkleErrorDomain", code: 3001, userInfo: [NSLocalizedDescriptionKey: privateText, NSUnderlyingErrorKey: NSError(domain: privateText, code: 9)])
        XCTAssertEqual(UpdateDiagnosticOutcome.classify(error), .verificationFailed)
        XCTAssertEqual(UpdateDiagnosticOutcome.classify(NSError(domain: NSURLErrorDomain, code: NSURLErrorCancelled)), .cancelled)
        XCTAssertEqual(UpdateDiagnosticOutcome.classify(NSError(domain: NSURLErrorDomain, code: NSURLErrorTimedOut)), .networkFailed)
        XCTAssertEqual(UpdateDiagnosticOutcome.classify(NSError(domain: privateText, code: 3001)), .otherFailure)
        XCTAssertEqual(UpdateDiagnosticOutcome.classify(NSError(domain: "SUSparkleErrorDomain", code: 1001)), .noUpdate)
        XCTAssertEqual(UpdateDiagnosticOutcome.classify(NSError(domain: "SUSparkleErrorDomain", code: 4007)), .cancelled)
        for outcome in UpdateDiagnosticOutcome.allCases {
            let text = String(decoding: try report(outcome: outcome).encoded(), as: UTF8.self)
            XCTAssertFalse(text.contains(privateText)); XCTAssertLessThan(text.utf8.count, 4096)
        }
    }
    func testCycleCompletionPreservesCancellationAndRefusalRatherThanClaimingSuccess() {
        XCTAssertEqual(UpdateDiagnosticOutcome.checking.completingCycle(error: nil), .finished)
        for outcome in [UpdateDiagnosticOutcome.cancelled, .signInRequired, .activeWork, .refused, .noUpdate, .verificationFailed] {
            XCTAssertEqual(outcome.completingCycle(error: nil), outcome)
        }
        let ownRefusal = NSError(domain: "info.kwip.KwipML.updates", code: 1, userInfo: [NSLocalizedDescriptionKey: "private"])
        XCTAssertEqual(UpdateDiagnosticOutcome.signInRequired.completingCycle(error: ownRefusal), .signInRequired)
        XCTAssertEqual(UpdateDiagnosticOutcome.checking.completingCycle(error: NSError(domain: "unknown", code: 1)), .otherFailure)
    }

}
