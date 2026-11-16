import Foundation

/// Local, application-wide run feedback. Notification content never includes project or source names.
public struct RunFeedbackPreferences: Codable, Equatable {
    public var showStatusIcon: Bool
    public var notifyCompleted: Bool
    public var notifyFailed: Bool
    public static let factory = Self()
    public init(showStatusIcon: Bool = true, notifyCompleted: Bool = false, notifyFailed: Bool = false) {
        self.showStatusIcon = showStatusIcon; self.notifyCompleted = notifyCompleted; self.notifyFailed = notifyFailed
    }
}
