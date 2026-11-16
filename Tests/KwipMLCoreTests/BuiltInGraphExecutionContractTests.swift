import XCTest
@testable import KwipMLCore

final class BuiltInGraphExecutionContractTests: XCTestCase {
    func testLiveDescriptorRoundTripAndPrimitiveOwnership() throws {
        let registry = try ComponentRegistry.builtIn()
        let expected = try BuiltInCapabilityBridge.executionContract(registry: registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
        let decoded = try JSONDecoder().decode(BuiltInGraphExecutionContract.self, from: JSONEncoder().encode(expected))
        XCTAssertEqual(decoded, expected)
        XCTAssertNoThrow(try decoded.validate(registry: registry, runtimeVersion: "2.14.1", availableDevices: [.cpu]))
        XCTAssertEqual(decoded.tensor.rank, 2)
        XCTAssertEqual(decoded.availableDevices, [.cpu])
        XCTAssertEqual(decoded.operations.exactResume, "cpuEpochBoundary")
        XCTAssertEqual(decoded.operations.training, "experimentOwnedAdam")
        XCTAssertEqual(decoded.rawModelSemantics, "pytorch")
        let primitive = try XCTUnwrap(BuiltInCapabilityBridge.package(registry: registry, runtimeVersion: "2.14.1").capabilities.first { $0.id == "linear" })
        XCTAssertEqual(primitive.lifecycle.resume, .none)
    }

    func testActualWorkerDiscoveryMatchesNativeContract() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let python = root.appendingPathComponent(".venv/bin/python")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw XCTSkip("Development worker runtime is unavailable.") }
        let process = Process(); let pipe = Pipe()
        process.executableURL = python
        process.arguments = [root.appendingPathComponent("runner/main.py").path, "--capabilities"]
        process.currentDirectoryURL = root; process.standardOutput = pipe
        try process.run()
        let data = pipe.fileHandleForReading.readDataToEndOfFile(); process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0)
        let event = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let live = try JSONDecoder().decode(BuiltInGraphExecutionContract.self, from: JSONSerialization.data(withJSONObject: try XCTUnwrap(event["executionContract"])))
        let extensions = try XCTUnwrap(event["componentCapabilities"] as? [String:Any])
        let embedding = try XCTUnwrap(extensions["embedding"])
        XCTAssertNoThrow(try EmbeddingRuntimeCapability.decode(JSONSerialization.data(withJSONObject:embedding)))
        let version = try XCTUnwrap(event["torchVersion"] as? String)
        let devices = try XCTUnwrap(event["devices"] as? [String]).map { try XCTUnwrap(ContractDevice(rawValue: $0)) }
        XCTAssertNoThrow(try live.validate(registry: .builtIn(), runtimeVersion: version, availableDevices: devices))
    }

    func testUnknownFieldsAtEveryLevelAreRejected() throws {
        let expected = try BuiltInCapabilityBridge.executionContract(registry: .builtIn(), runtimeVersion: "2.14.1", availableDevices: [.cpu])
        let original = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(expected)) as? [String: Any])
        for path in ["", "tensor", "task", "limits", "operations"] {
            var changed = original
            if path.isEmpty { changed["unsupported"] = true }
            else {
                var nested = try XCTUnwrap(changed[path] as? [String: Any])
                nested["unsupported"] = true; changed[path] = nested
            }
            XCTAssertThrowsError(try JSONDecoder().decode(BuiltInGraphExecutionContract.self, from: JSONSerialization.data(withJSONObject: changed)))
        }
    }

    func testDriftAndUnsupportedDiscoveryFailClosed() throws {
        let registry = try ComponentRegistry.builtIn()
        let expected = try BuiltInCapabilityBridge.executionContract(registry: registry, runtimeVersion: "2.14.1", availableDevices: [.cpu, .mps])
        let data = try JSONEncoder().encode(expected)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        for patch: [String: Any] in [["dtype": "float64"], ["rank": 3], ["layout": "sparse"], ["noncontiguous": false]] {
            var changed = json
            var tensor = try XCTUnwrap(json["tensor"] as? [String: Any])
            tensor.merge(patch) { _, new in new }; changed["tensor"] = tensor
            let decoded = try JSONDecoder().decode(BuiltInGraphExecutionContract.self, from: JSONSerialization.data(withJSONObject: changed))
            XCTAssertThrowsError(try decoded.validate(registry: registry, runtimeVersion: "2.14.1", availableDevices: [.cpu, .mps]))
        }
        json["availableDevices"] = ["cpu", "cuda"]
        let wrong = try JSONDecoder().decode(BuiltInGraphExecutionContract.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertThrowsError(try wrong.validate(registry: registry, runtimeVersion: "2.14.1", availableDevices: [.cpu]))
        XCTAssertThrowsError(try expected.validate(registry: registry, runtimeVersion: "2.14.2", availableDevices: [.cpu, .mps]))
        XCTAssertThrowsError(try BuiltInCapabilityBridge.executionContract(registry: registry, runtimeVersion: "2.14", availableDevices: [.cpu]))
        XCTAssertThrowsError(try BuiltInCapabilityBridge.executionContract(registry: registry, runtimeVersion: "2.14.1", availableDevices: [.mps]))
        XCTAssertThrowsError(try BuiltInCapabilityBridge.executionContract(registry: registry, runtimeVersion: "2.14.1", availableDevices: [.cpu, .cpu]))
    }
}
