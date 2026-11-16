import XCTest
import UserNotifications
import KwipMLCore
@testable import KwipMLApp

@MainActor final class RunFeedbackTests: XCTestCase {
    func testEnabledPermissionStateIsImmediateBoundedAndStaleCallbacksAreIgnored() async {
        var queries: [(UNAuthorizationStatus) -> Void] = []
        var requests: [(Error?) -> Void] = []
        let feedback = RunFeedbackController(permissionQuery: { queries.append($0) },
            permissionRequest: { requests.append($0) }, permissionTimeoutNanoseconds: 0)
        feedback.configure(.init(notifyCompleted: true))
        XCTAssertEqual(feedback.deliveryMessage, "Checking notification permission…")
        XCTAssertTrue(requests.isEmpty) // configure/load cannot request permission
        feedback.requestAuthorization()
        XCTAssertEqual(feedback.deliveryMessage, "Requesting notification permission…")
        queries[0](.authorized) // superseded by the explicit request
        for _ in 0..<10 { await Task.yield() }
        XCTAssertTrue(feedback.deliveryMessage.contains("Delivery is unconfirmed"))
        feedback.configure(.factory)
        requests[0](NSError(domain: "Late response", code: 1))
        for _ in 0..<10 { await Task.yield() }
        XCTAssertEqual(feedback.deliveryMessage, "Notifications are off.")
        feedback.configure(.init(notifyFailed: true))
        queries.last?(.denied)
        for _ in 0..<10 { await Task.yield() }
        XCTAssertTrue(feedback.deliveryMessage.contains("blocked in macOS Settings"))
    }
    func testUnpackagedPermissionIsHonestImmediatelyAfterEnabling() {
        let feedback = RunFeedbackController()
        feedback.configure(.init(notifyCompleted: true))
        XCTAssertTrue(feedback.deliveryMessage.contains("require the packaged app"))
        feedback.requestAuthorization()
        XCTAssertFalse(feedback.deliveryMessage == "Notifications are off.")
    }

    func testTerminalOutcomesAreOnceOnlyAndCancelledIsNeverSuccess() {
        var deliveries: [RunFeedbackController.State] = []
        let feedback = RunFeedbackController(deliver: { deliveries.append($0) })
        feedback.configure(.init(notifyCompleted: true, notifyFailed: true))
        let first = UUID(), second = UUID()
        feedback.begin(first, state: .queued); feedback.begin(second)
        XCTAssertEqual(feedback.activities.count, 2)
        feedback.transition(first, state: .running)
        feedback.transition(first, state: .completed)
        feedback.transition(first, state: .failed) // late duplicate cannot change outcome
        XCTAssertEqual(deliveries, [.completed])
        feedback.transition(second, state: .cancelling)
        XCTAssertEqual(feedback.activities.first?.state, .cancelling)
        feedback.transition(second, state: .cancelled)
        XCTAssertEqual(feedback.latestOutcome, .cancelled)
        XCTAssertEqual(deliveries, [.completed])
        XCTAssertTrue(feedback.activities.isEmpty)
    }
    func testDefaultsArePrivateAndUnseenHistoricalOutcomesDoNotNotify() {
        var deliveries: [RunFeedbackController.State] = []
        let feedback = RunFeedbackController(deliver: { deliveries.append($0) })
        feedback.transition(UUID(), state: .completed)
        XCTAssertNil(feedback.latestOutcome)
        let id = UUID(); feedback.begin(id); feedback.transition(id, state: .failed)
        XCTAssertEqual(feedback.latestOutcome, .failed)
        XCTAssertTrue(deliveries.isEmpty)
    }
    func testFeedbackPreferencesSurviveOtherPreferenceEditsAndLegacyDefaults() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let store = AppPreferencesStore(fileURL: directory.appendingPathComponent("preferences.json"))
        var preferences = AppPreferences.factory
        XCTAssertEqual(preferences.effectiveRunFeedback, .factory)
        preferences.runFeedback = .init(showStatusIcon: false, notifyFailed: true)
        try store.save(preferences)
        _ = try store.resetAppearance()
        XCTAssertEqual(try store.load().effectiveRunFeedback, preferences.runFeedback)
        preferences.runFeedback = nil; try store.save(preferences)
        XCTAssertEqual(try store.load().effectiveRunFeedback, .factory)
    }
}
