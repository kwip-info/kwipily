import Foundation
import XCTest
@testable import KwipMLCore

final class DerivativeCapturePreferencesTests: XCTestCase {
    private func encode<T: Encodable>(_ value: T) throws -> String {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }
    private func decodeGlobal(_ text: String) throws -> ExecutionPreferences {
        try JSONDecoder().decode(ExecutionPreferences.self, from: Data(text.utf8))
    }
    private func decodeProject(_ text: String) throws -> ProjectExecutionOverrides {
        try JSONDecoder().decode(ProjectExecutionOverrides.self, from: Data(text.utf8))
    }
    private let globalLegacy = #"{"cpuThreads":1,"maximumRunSeconds":3600,"preparationCacheLimitBytes":200000000,"sampleCapture":{"activationValues":true,"predictionRows":true,"schemaVersion":1},"schemaVersion":3}"#
    private let projectLegacy = #"{"memory":{"mode":"inherit"},"schemaVersion":2}"#

    func testLegacyDefaultBytesAndOldVersionsPreserveIndependentFalseInheritance() throws {
        XCTAssertEqual(try encode(ExecutionPreferences.factory), globalLegacy)
        XCTAssertEqual(try encode(ProjectExecutionOverrides.factory), projectLegacy)
        XCTAssertEqual(try encode(ExecutionPreferences(derivativeCapture: .factory)), globalLegacy)
        XCTAssertFalse(try decodeGlobal(globalLegacy).derivativeCapture.values)
        XCTAssertNil(try decodeProject(projectLegacy).derivativeCapture)
        for version in [1, 2] {
            let sample = version == 2 ? #", "sampleCapture":{"schemaVersion":1,"activationValues":false,"predictionRows":false}"# : ""
            let global = try decodeGlobal("{\"schemaVersion\":\(version),\"maximumRunSeconds\":3600,\"cpuThreads\":1\(sample)}")
            XCTAssertFalse(global.derivativeCapture.values)
        }
        let project = try decodeProject(#"{"schemaVersion":1,"memory":{"mode":"inherit"}}"#)
        XCTAssertNil(project.derivativeCapture)
        XCTAssertFalse(try ExecutionPreferences.factory.resolve(project: project).derivativeCapture.values)
    }

    func testExplicitPermissionAndProjectFalseHaveDistinctVersionedRoundTrips() throws {
        let global = try ExecutionPreferences(derivativeCapture: .init(values: true))
        XCTAssertEqual(global.schemaVersion, 4)
        XCTAssertTrue(try encode(global).contains(#""derivativeCapture":{"schemaVersion":1,"values":true}"#))
        XCTAssertEqual(try decodeGlobal(encode(global)), global)
        for allowed in [false, true] {
            let project = try ProjectExecutionOverrides(derivativeCapture: .init(values: allowed))
            XCTAssertEqual(project.schemaVersion, 3)
            XCTAssertEqual(try decodeProject(encode(project)), project)
            let resolved = try global.resolve(project: project)
            XCTAssertEqual(resolved.derivativeCapture, .init(values: allowed))
            XCTAssertEqual(resolved.derivativeCaptureSource, .project)
        }
        let inherited = try global.resolve(project: .factory)
        XCTAssertTrue(inherited.derivativeCapture.values)
        XCTAssertEqual(inherited.derivativeCaptureSource, .application)
        XCTAssertEqual(try global.replacingDerivativeCapture(.factory).schemaVersion, 3)
        XCTAssertEqual(try ProjectExecutionOverrides(derivativeCapture: .factory).replacingDerivativeCapture(nil).schemaVersion, 2)
    }

    func testVersionedFieldsAndPolicyTypesAreStrict() throws {
        let base = #""maximumRunSeconds":3600,"cpuThreads":1,"sampleCapture":{"schemaVersion":1,"activationValues":true,"predictionRows":true},"preparationCacheLimitBytes":200000000"#
        for policy in [#"{"schemaVersion":1,"values":1}"#, "null", #"{"schemaVersion":2,"values":true}"#, #"{"schemaVersion":1,"values":true,"activationValues":true}"#] {
            XCTAssertThrowsError(try decodeGlobal("{\"schemaVersion\":4,\(base),\"derivativeCapture\":\(policy)}"))
            XCTAssertThrowsError(try decodeProject("{\"schemaVersion\":3,\"memory\":{\"mode\":\"inherit\"},\"derivativeCapture\":\(policy)}"))
        }
        XCTAssertThrowsError(try decodeGlobal("{\"schemaVersion\":4,\(base)}"))
        XCTAssertThrowsError(try decodeGlobal("{\"schemaVersion\":5,\(base),\"derivativeCapture\":{\"schemaVersion\":1,\"values\":true}}"))
        XCTAssertThrowsError(try decodeGlobal("{\"schemaVersion\":3,\(base),\"derivativeCapture\":{\"schemaVersion\":1,\"values\":false}}"))
        XCTAssertThrowsError(try decodeProject(#"{"schemaVersion":3,"memory":{"mode":"inherit"}}"#))
        XCTAssertThrowsError(try decodeProject(#"{"schemaVersion":2,"memory":{"mode":"inherit"},"derivativeCapture":{"schemaVersion":1,"values":false}}"#))
        XCTAssertThrowsError(try decodeProject(#"{"schemaVersion":4,"memory":{"mode":"inherit"},"derivativeCapture":{"schemaVersion":1,"values":true}}"#))
    }

    func testEveryReplacementAndResourceResetPreservesOtherCaptureGroups() throws {
        let sample = TrainingSampleCapturePolicy(activationValues: false, predictionRows: true)
        let global = try ExecutionPreferences(maximumRunSeconds: 90, cpuThreads: 2, maximumResidentBytes: 268_435_456, managedStorageLimitBytes: 3_000_000, sampleCapture: sample, preparationCacheLimitBytes: 10, derivativeCapture: .init(values: true))
        let replacedSample = try global.replacingSampleCapture(.factory)
        XCTAssertTrue(replacedSample.derivativeCapture.values)
        XCTAssertEqual(replacedSample.maximumRunSeconds, 90)
        XCTAssertEqual(replacedSample.preparationCacheLimitBytes, 10)
        let replacedDerivative = try global.replacingDerivativeCapture(.factory)
        XCTAssertEqual(replacedDerivative.sampleCapture, sample)
        XCTAssertEqual(replacedDerivative.cpuThreads, 2)
        XCTAssertEqual(replacedDerivative.maximumResidentBytes, 268_435_456)
        XCTAssertEqual(replacedDerivative.managedStorageLimitBytes, 3_000_000)
        XCTAssertEqual(replacedDerivative.preparationCacheLimitBytes, 10)
        let reset = try global.resettingResources()
        XCTAssertEqual(reset.sampleCapture, sample); XCTAssertTrue(reset.derivativeCapture.values)
        XCTAssertEqual(reset.maximumRunSeconds, ExecutionPreferences.factory.maximumRunSeconds)
        let project = try ProjectExecutionOverrides(maximumRunSeconds: 90, cpuThreads: 2, memory: .limit(268_435_456), sampleCapture: sample, derivativeCapture: .init(values: false))
        XCTAssertEqual(try project.replacingSampleCapture(.factory).derivativeCapture, .factory)
        XCTAssertEqual(try project.replacingDerivativeCapture(nil).sampleCapture, sample)
        XCTAssertEqual(try project.replacingDerivativeCapture(nil).memory, .limit(268_435_456))
        let projectReset = try project.resettingResources()
        XCTAssertEqual(projectReset.sampleCapture, sample); XCTAssertEqual(projectReset.derivativeCapture, .factory)
        XCTAssertNil(projectReset.cpuThreads); XCTAssertEqual(projectReset.memory, .inherit)
    }

    func testSampleCaptureNeverAuthorizesDerivativeValuesAndSourcesResolveSeparately() throws {
        let global = try ExecutionPreferences(sampleCapture: .factory)
        XCTAssertFalse(try global.resolve().derivativeCapture.values)
        let project = try ProjectExecutionOverrides(sampleCapture: .init(activationValues: false, predictionRows: false), derivativeCapture: .init(values: true))
        let resolved = try global.resolve(project: project)
        XCTAssertTrue(resolved.derivativeCapture.values)
        XCTAssertFalse(resolved.sampleCapture.activationValues)
        XCTAssertFalse(resolved.sampleCapture.predictionRows)
        XCTAssertEqual(resolved.derivativeCaptureSource, .project)
        XCTAssertEqual(resolved.sampleCaptureSource, .project)
        let inheritedSample = try global.resolve(project: ProjectExecutionOverrides(derivativeCapture: .init(values: true)))
        XCTAssertEqual(inheritedSample.sampleCapture, .factory)
        XCTAssertEqual(inheritedSample.sampleCaptureSource, .application)
        XCTAssertEqual(inheritedSample.derivativeCaptureSource, .project)
    }
}
