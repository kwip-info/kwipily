import XCTest
@testable import KwipMLCore

final class WorkflowTrainingModeTests: XCTestCase {
    func testModesAreVersionedAndRejectImplicitOrMalformedContinuation() throws {
        for version in [1,2,3] {
            XCTAssertEqual(try WorkflowControlContracts.trainingMode([:], schemaVersion: version), .fresh)
        }
        for mode in WorkflowTrainingMode.allCases {
            XCTAssertEqual(try WorkflowControlContracts.trainingMode(["trainingMode": .string(mode.rawValue)], schemaVersion: 3), mode)
            for version in [1,2] {
                XCTAssertThrowsError(try WorkflowControlContracts.trainingMode(["trainingMode": .string(mode.rawValue)], schemaVersion: version))
            }
        }
        for value in [JSONValue.null, .bool(true), .integer(1), .string("resume")] {
            XCTAssertThrowsError(try WorkflowControlContracts.trainingMode(["trainingMode": value], schemaVersion: 3))
        }
        XCTAssertThrowsError(try WorkflowControlContracts.trainingMode([:], schemaVersion: 99))
    }
    func testMigrationPreservesSourceAndAddsOnlyExplicitVersionBoundary() throws {
        var document = WorkflowDocument(projectID: UUID().uuidString)
        document.nodes = [.init(address: .init(domain: .control, localID: "train"), name: "My experiment", x: 40, y: 70, payload: .train(config: [:]))]
        let before = try WorkflowCodec.encode(document)
        let migrated = try WorkflowCodec.migrateToVersion3(document)
        XCTAssertEqual(migrated.schemaVersion, 3)
        XCTAssertEqual(migrated.nodes, document.nodes)
        XCTAssertEqual(migrated.revision, document.revision)
        XCTAssertEqual(try WorkflowCodec.encode(document), before)
        XCTAssertEqual(try WorkflowCodec.migrateToVersion3(migrated), migrated)
        for mode in WorkflowTrainingMode.allCases {
            var configured = migrated
            configured.nodes[0].payload = .train(config: ["trainingMode": .string(mode.rawValue)])
            XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(configured)), configured)
            configured.schemaVersion = 2
            XCTAssertThrowsError(try WorkflowCodec.encode(configured))
        }
    }
    func testEpochMeaningIsExplicit() {
        XCTAssertTrue(WorkflowTrainingMode.exactResume.epochExplanation.contains("total"))
        XCTAssertTrue(WorkflowTrainingMode.fineTune.epochExplanation.contains("new"))
        XCTAssertTrue(WorkflowTrainingMode.fineTune.stateExplanation.contains("optimizer"))
    }
}
