#if KWIPML_RECOVERY_UAT
import XCTest
@testable import KwipMLApp

@MainActor final class RecoveryUATGateTests: XCTestCase {
    private let identity = RecoveryUATGate.Identity(groupID: "group", runID: "run", ordinal: 1, nonce: "launch")

    func testExplicitArmMatchesOnlyRequestedBoundaryAndConsumesOnce() throws {
        let gate = RecoveryUATGate()
        var resumed = 0, cancelled = 0
        XCTAssertNil(gate.hold(.beforeWorkerSpawn, identity: identity, resume: { resumed += 1 }, cancel: { cancelled += 1 }))
        gate.arm(.betweenVariants)
        XCTAssertNil(gate.hold(.beforeWorkerSpawn, identity: identity, resume: {}, cancel: {}))
        let token = try XCTUnwrap(gate.hold(.betweenVariants, identity: identity, resume: { resumed += 1 }, cancel: { cancelled += 1 }))
        XCTAssertEqual(gate.reached?.identity, identity)
        XCTAssertEqual(gate.reached?.stage, .betweenVariants)
        XCTAssertNil(gate.armed)
        gate.arm(.beforeWorkerSpawn)
        XCTAssertNil(gate.armed, "Cannot rearm over a live owned boundary")
        gate.resume()
        gate.resume()
        gate.cancel(token)
        XCTAssertEqual(resumed, 1)
        XCTAssertEqual(cancelled, 0)
        XCTAssertNil(gate.reached)
    }

    func testCancelClearsAuthorityBeforeReentrantCleanupAndNeverResumes() throws {
        let gate = RecoveryUATGate()
        var cancelled = 0, resumed = 0
        var token: UUID?
        gate.arm(.beforeWorkerSpawn)
        token = try XCTUnwrap(gate.hold(.beforeWorkerSpawn, identity: identity, resume: { resumed += 1 }, cancel: {
            cancelled += 1
            XCTAssertNil(gate.reached)
            gate.cancel(token!)
            gate.resume()
        }))
        gate.cancel(token!)
        XCTAssertEqual(cancelled, 1)
        XCTAssertEqual(resumed, 0)
    }

    func testExpiredOldTokenCannotCancelNewBoundaryAndTimeoutCancelsOnce() throws {
        let gate = RecoveryUATGate()
        var cancelled = 0
        gate.arm(.beforeWorkerSpawn)
        let old = try XCTUnwrap(gate.hold(.beforeWorkerSpawn, identity: identity, resume: {}, cancel: { cancelled += 1 }))
        gate.resume()
        gate.arm(.betweenVariants)
        let current = try XCTUnwrap(gate.hold(.betweenVariants, identity: identity, resume: { XCTFail("Timeout must not launch subsequent work") }, cancel: { cancelled += 1 }))
        gate.expire(old)
        XCTAssertEqual(gate.reached?.token, current)
        gate.expire(current)
        gate.expire(current)
        XCTAssertEqual(cancelled, 1)
        XCTAssertNil(gate.reached)
    }

    func testActualPreparationExitDismissesBarrierWithoutCancellingAgain() throws {
        let gate = RecoveryUATGate()
        gate.arm(.duringPreparation)
        XCTAssertNil(gate.reached, "Arming is not evidence that a worker has reached preparation")
        let token = try XCTUnwrap(gate.hold(.duringPreparation, identity: identity,
            resume: { XCTFail("Exited worker cannot resume") }, cancel: { XCTFail("Actual exit must not synthesize cancellation") }))
        gate.dismiss(UUID())
        XCTAssertEqual(gate.reached?.token, token)
        gate.dismiss(token)
        gate.resume()
        gate.expire(token)
        XCTAssertNil(gate.reached)
    }
}
#endif
