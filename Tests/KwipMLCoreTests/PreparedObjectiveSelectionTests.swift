import XCTest
@testable import KwipMLCore

final class PreparedObjectiveSelectionTests: XCTestCase {
    private var reference: PreparedDataReference {
        .init(manifestPath: "/private/prepared/manifest.json", manifestSHA256: String(repeating: "a", count: 64), graphDigest: String(repeating: "B", count: 64), trainingNode: "split", trainingPort: "training", validationNode: "split", validationPort: "validation")
    }
    private func selection(_ task: LossLearningTask) throws -> PreparedObjectiveSelection {
        let doc = try WorkflowExamples.preparedLossObjective(task: task, device: "cpu")
        let node = try XCTUnwrap(doc.nodes.first { $0.address.localID == "mapping" })
        guard case .trainingData(let config) = node.payload, let raw = config["lossTargetMapping"] else { throw GraphConfigurationError("Missing real example mapping") }
        let mapping = try JSONDecoder().decode(LossTargetMapping.self, from: JSONEncoder().encode(raw))
        return .init(reference: reference, objectiveRecipe: try LossConfiguration(componentID: task.componentID, values: task.configuration).recipeValue, targetMapping: mapping)
    }
    func testAllEightTypedSelectionsRoundTripWithoutInventedTarget() throws {
        for task in LossLearningTask.allCases {
            do {
                let value = try selection(task); try value.validate()
                XCTAssertEqual(try JSONDecoder().decode(PreparedObjectiveSelection.self, from: JSONEncoder().encode(value)), value)
                let object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
                XCTAssertEqual(Set(object.keys), ["schemaVersion","reference","objectiveRecipe","targetMapping","scalarization"])
                XCTAssertNil(object["targetColumn"])
            } catch { XCTFail("\(task.rawValue): \(error)") }
        }
    }
    func testPinsAndRoleIdentityCannotBeSubstituted() throws {
        var value = try selection(.mse); value.reference.manifestSHA256 = "changed"
        XCTAssertThrowsError(try value.validate())
        value = try selection(.mse); value.targetMapping.componentID = "loss_l1"
        XCTAssertThrowsError(try value.validate())
        value = try selection(.mse); value.schemaVersion = 2
        XCTAssertThrowsError(try value.validate())
        value = try selection(.bce); value.targetMapping.roles.removeAll { $0.role == "weight" }
        XCTAssertThrowsError(try value.validate())
    }
    func testCanonicalRecipeAndItsByteBoundRemainRequired() throws {
        var value = try selection(.mse)
        guard case .object(var recipe) = value.objectiveRecipe else { return XCTFail("Missing recipe") }
        recipe["extra"] = true; value.objectiveRecipe = .object(recipe)
        XCTAssertThrowsError(try value.validate())
        value.objectiveRecipe = .object(["schemaVersion": 1,"configurationVersion": 1,"componentID": .string("loss_mse"),"config": .object(["reduction": .string(String(repeating: "x", count: 32769))])])
        XCTAssertThrowsError(try value.validate())
    }
    func testTransportRefusesUnknownKeysWithoutAffectingLegacyDecoding() throws {
        let value = try selection(.ctc), encoder = JSONEncoder()
        do {
            let bytes = try encoder.encode(value)
            XCTAssertEqual(try PreparedObjectiveSelection.decode(bytes), value)
        } catch { XCTFail("Valid typed transport: \(error)") }
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: encoder.encode(value)) as? [String: Any])
        object["targetColumn"] = "invented"
        XCTAssertThrowsError(try PreparedObjectiveSelection.decode(JSONSerialization.data(withJSONObject: object)))
        object.removeValue(forKey: "targetColumn")
        var ref = try XCTUnwrap(object["reference"] as? [String: Any]); ref["extra"] = true; object["reference"] = ref
        XCTAssertThrowsError(try PreparedObjectiveSelection.decode(JSONSerialization.data(withJSONObject: object)))
        XCTAssertThrowsError(try PreparedObjectiveSelection.decode(Data(repeating: 32, count: 524289)))
    }
    func testLegacySerializationRetainsOnlyItsOriginalKeys() throws {
        let old = PreparedTrainingSelection(manifestPath: reference.manifestPath, manifestSHA256: reference.manifestSHA256, graphDigest: reference.graphDigest, trainingNode: reference.trainingNode, trainingPort: reference.trainingPort, validationNode: reference.validationNode, validationPort: reference.validationPort, featureColumns: ["x","y"], targetColumn: "class")
        try old.validate()
        let bytes = try JSONEncoder().encode(old)
        XCTAssertEqual(try JSONDecoder().decode(PreparedTrainingSelection.self, from: bytes), old)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: bytes) as? [String: Any])
        XCTAssertEqual(Set(object.keys), ["manifestPath","manifestSHA256","graphDigest","trainingNode","trainingPort","validationNode","validationPort","featureColumns","targetColumn"])
    }
}
