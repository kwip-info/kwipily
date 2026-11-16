import Foundation

/// Fixed categories only: never retain an NSError, its domain, userInfo or underlying errors.
public enum UpdateDiagnosticOutcome: String, Codable, CaseIterable {
    case notConfigured, ready, checking, activeWork, signInRequired, refused, offerAvailable
    case downloading, finished, noUpdate, cancelled, configurationFailed, networkFailed
    case verificationFailed, installationFailed, otherFailure

    public func completingCycle(error: NSError?) -> Self {
        if let error {
            return error.domain == "info.kwip.KwipML.updates" ? self : Self.classify(error)
        }
        // Completion can follow cancellation/refusal; preserve the actionable reason.
        switch self {
        case .checking, .offerAvailable, .downloading: return .finished
        default: return self
        }
    }

    public static func classify(_ error: NSError) -> Self {
        if error.domain == NSURLErrorDomain {
            return error.code == NSURLErrorCancelled ? .cancelled : .networkFailed
        }
        // Pinned Sparkle 2.10.0 SUErrors.h. Unknown domains/codes never become success.
        guard error.domain == "SUSparkleErrorDomain" else { return .otherFailure }
        switch error.code {
        case 1...7, 5000: return .configurationFailed
        case 1001: return .noUpdate
        case 1000, 1002, 2001: return .networkFailed
        case 3000...3002: return .verificationFailed
        case 4007, 4008: return .cancelled
        case 2000, 1003, 1005, 4000...4006, 4009, 4010, 4012: return .installationFailed
        default: return .otherFailure
        }
    }
}

/// Voluntary local support snapshot, independent of projects and online authority.
public struct SupportDiagnosticReport: Encodable {
    public enum Architecture: String, Encodable { case arm64, x86_64, unknown }
    public enum Edition: String, Encodable { case direct, development }
    public let schemaVersion = 2
    public let version: String, build: String, macOS: String
    public let architecture: Architecture, edition: Edition
    public let updaterAvailable: Bool, automaticChecks: Bool
    public let updateOutcome: UpdateDiagnosticOutcome

    public init(version: String?, build: String?, os: OperatingSystemVersion,
                architecture: Architecture, edition: Edition,
                updaterAvailable: Bool, automaticChecks: Bool, updateOutcome: UpdateDiagnosticOutcome) {
        self.version = Self.numeric(version, pattern: "^[0-9]{1,3}(\\.[0-9]{1,3}){1,2}$")
        self.build = Self.numeric(build, pattern: "^[1-9][0-9]{0,8}$")
        self.macOS = [os.majorVersion, os.minorVersion, os.patchVersion].allSatisfy { (0...999).contains($0) }
            ? "\(os.majorVersion).\(os.minorVersion).\(os.patchVersion)" : "unknown"
        self.architecture = architecture; self.edition = edition
        self.updaterAvailable = updaterAvailable; self.automaticChecks = automaticChecks
        self.updateOutcome = updateOutcome
    }
    private static func numeric(_ value: String?, pattern: String) -> String {
        guard let value, value.utf8.count <= 16,
              let range = value.range(of: pattern, options: .regularExpression),
              range == value.startIndex..<value.endIndex else { return "unknown" }
        return value
    }
    public func encoded() throws -> Data {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(self)
        guard data.count <= 4096 else { throw EncodingError.invalidValue(schemaVersion, .init(codingPath: [], debugDescription: "Support snapshot exceeds its bound")) }
        return data
    }
}
