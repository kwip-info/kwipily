import XCTest
@testable import KwipMLCore

final class PoolingRuntimeCapabilitiesTests: XCTestCase {
    func testExactDiscoveryOnlyAcceptsPublishedMatchingContracts() throws {
        var descriptors: [String: Data] = [:]
        for (id, entry) in BundledPoolingComponents.entries {
            descriptors[id] = try JSONEncoder().encode(entry.runtimeContract)
        }
        let verified = PoolingRuntimeCapabilities(descriptors: descriptors)
        XCTAssertEqual(verified.componentIDs, ["max_pool1d", "max_pool2d", "max_pool3d", "avg_pool1d", "avg_pool2d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool2d"])
        XCTAssertFalse(verified.contains("unknown"))
        XCTAssertTrue(PoolingRuntimeCapabilities(descriptors: [:]).componentIDs.isEmpty)
        descriptors["max_pool2d"] = Data("{}".utf8)
        XCTAssertEqual(PoolingRuntimeCapabilities(descriptors: descriptors).componentIDs, ["max_pool1d", "max_pool3d", "avg_pool1d", "avg_pool2d", "adaptive_avg_pool1d", "adaptive_avg_pool2d", "adaptive_max_pool2d"])
    }

    func testRejectsWrongComponentOversizedAndUnknownContracts() throws {
        let second = try XCTUnwrap(BundledPoolingComponents.entries["max_pool2d"])
        let data = try JSONEncoder().encode(second.runtimeContract)
        XCTAssertThrowsError(try PoolingRuntimeCapabilities.verify(data, componentID: "max_pool1d"))
        XCTAssertThrowsError(try PoolingRuntimeCapabilities.verify(data, componentID: "unknown"))
        XCTAssertThrowsError(try PoolingRuntimeCapabilities.verify(Data(repeating: 32, count: 4097), componentID: "max_pool2d"))
        XCTAssertNoThrow(try PoolingRuntimeCapabilities.verify(data, componentID: "max_pool2d"))
    }
}
