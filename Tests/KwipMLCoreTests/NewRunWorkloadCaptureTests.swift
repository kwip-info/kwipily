import XCTest
@testable import KwipMLCore

final class NewRunWorkloadCaptureTests: XCTestCase {
    func testNewRunCapturesLegacyCeilingsWithoutRewritingOldPreferencesOrBudgets() throws {
        let bytes = Data(#"{"schemaVersion":1,"maximumRunSeconds":3600,"cpuThreads":1}"#.utf8)
        let old = try JSONDecoder().decode(ExecutionPreferences.self, from: bytes)
        XCTAssertNil(old.workloadPolicy)
        let resolved = try old.resolve(project:nil)
        let policy = try XCTUnwrap(resolved.budget.resources?.workloadPolicy)
        XCTAssertEqual(policy.preset,.legacy)
        XCTAssertEqual(policy.limit(.retainedBytes),200_000_000)
        XCTAssertEqual(policy.limit(.parallelTasks),1)
        XCTAssertEqual(policy.limit(.scratchBytes),100_000_000)
        let historic = try JSONDecoder().decode(ExecutionResourcePolicy.self,from:Data(#"{"schemaVersion":1,"cpuThreads":1}"#.utf8))
        XCTAssertNil(historic.workloadPolicy,"Historical evidence stays on its original reader path")
        XCTAssertNil(old.workloadPolicy,"Resolving a new run must not rewrite preferences")
    }
}
