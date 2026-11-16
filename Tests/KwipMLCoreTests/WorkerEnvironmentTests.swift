import Foundation
import XCTest
@testable import KwipMLCore

final class WorkerEnvironmentTests: XCTestCase {
    func testLaunchDoesNotInheritServiceCredentialsOrInjectionVariables() throws {
        let key = "KWIPML_SENTINEL_SECRET_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
        setenv(key, "not-a-real-credential", 1)
        defer { unsetenv(key) }
        let done = expectation(description: "worker exit")
        let process = try ProcessPump.launch(executable: URL(fileURLWithPath: "/usr/bin/env"), arguments: [], onLine: { line in
            XCTAssertFalse(line.hasPrefix(key + "="))
        }, onExit: { code, _ in XCTAssertEqual(code, 0); done.fulfill() })
        XCTAssertNil(process.environment?[key])
        XCTAssertEqual(process.environment?["PATH"], "/usr/bin:/bin")
        wait(for: [done], timeout: 5)
    }
}
