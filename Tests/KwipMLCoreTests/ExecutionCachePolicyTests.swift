import XCTest
@testable import KwipMLCore

final class ExecutionCachePolicyTests:XCTestCase {
    func testLegacyMigrationExactRoundTripBoundsAndSeparateStorageScope()throws {
        let old=Data(#"{"schemaVersion":1,"maximumRunSeconds":3600,"cpuThreads":1}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(ExecutionPreferences.self,from:old).preparationCacheLimitBytes,200_000_000)
        let v2=Data(#"{"schemaVersion":2,"maximumRunSeconds":3600,"cpuThreads":1,"sampleCapture":{"schemaVersion":1,"activationValues":false,"predictionRows":true}}"#.utf8)
        let migrated=try JSONDecoder().decode(ExecutionPreferences.self,from:v2)
        XCTAssertEqual(migrated.preparationCacheLimitBytes,200_000_000)
        XCTAssertFalse(migrated.sampleCapture.activationValues)
        let value=try ExecutionPreferences(managedStorageLimitBytes:900_000_000,preparationCacheLimitBytes:12345)
        XCTAssertEqual(try JSONDecoder().decode(ExecutionPreferences.self,from:JSONEncoder().encode(value)),value)
        XCTAssertEqual(try value.resolve(project:.init(cpuThreads:2)).preparationCacheLimitBytes,12345)
        XCTAssertEqual(try value.resolve().storageLimitBytes,900_000_000)
        for invalid in [0,-1,200_000_001,Int.max] {XCTAssertThrowsError(try ExecutionPreferences(preparationCacheLimitBytes:invalid))}
        XCTAssertNoThrow(try ExecutionPreferences(preparationCacheLimitBytes:1))
    }
    func testCapturedWorkloadCacheAllowanceRoundTripAndNarrowProject() throws {
        let policy = try ExecutionWorkloadPolicy(preset: .balanced)
        let value = try ExecutionPreferences(preparationCacheLimitBytes: 300_000_000, workloadPolicy: policy)
        XCTAssertEqual(try JSONDecoder().decode(ExecutionPreferences.self, from: JSONEncoder().encode(value)), value)
        XCTAssertThrowsError(try ExecutionPreferences(preparationCacheLimitBytes: Int(policy.limit(.retainedBytes)) + 1, workloadPolicy: policy))
        let narrow = try policy.overriding(["retainedBytes": 50_000_000])
        XCTAssertEqual(try value.resolve(project: .init(workloadPolicy: narrow)).preparationCacheLimitBytes, 50_000_000)
    }
    func testCapturePreferenceMutationPreservesCacheWhileResourceResetRestoresFactory()throws {
        let value=try ExecutionPreferences(preparationCacheLimitBytes:12345)
        XCTAssertEqual(try value.replacingSampleCapture(.factory).preparationCacheLimitBytes,12345)
        XCTAssertEqual(try value.resettingResources().preparationCacheLimitBytes,200_000_000)
    }
}
