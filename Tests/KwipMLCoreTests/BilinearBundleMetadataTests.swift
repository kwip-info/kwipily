import XCTest
@testable import KwipMLCore
final class BilinearBundleMetadataTests: XCTestCase {
    func testAuthoredBilinearPreservesProductionAndDistinctPrivateAssessmentContracts() throws {
        let entry = try XCTUnwrap(BilinearComponent.metadata)
        let definition = BilinearComponent.definition
        XCTAssertEqual(entry.definition, definition)
        let additiveIDs:Set<String>=["bag_input","embedding_bag","dataset_source","data_loader"]
        XCTAssertEqual(fixtureModelComponentsBeforeTextExtensions(try .builtIn()).filter { $0.id != "initialization" && $0.id != "gradient_policy" && $0.id != "trainability" && !additiveIDs.contains($0.id) }.count, 83)
        let currentComponents=fixtureModelComponentsBeforeTextExtensions(try .builtIn())
        XCTAssertEqual(Set(currentComponents.filter {additiveIDs.contains($0.id)}.map(\.id)),additiveIDs)
        XCTAssertEqual(currentComponents.filter { !["initialization","gradient_policy","trainability"].contains($0.id) }.count,87)
        XCTAssertEqual(definition.inputs, ["a", "b"])
        XCTAssertEqual(definition.outputs, ["out"])
        XCTAssertEqual(definition.fields.map(\.key), ["in1_features", "in2_features", "out_features", "bias"])
        XCTAssertEqual(definition.fields.map(\.defaultValue), [2,2,2,true])
        XCTAssertEqual(entry.presentation.icon, "bilinear-v2")
        XCTAssertEqual(entry.learning?.exampleID, "prepared-bilinear-classifier")
        XCTAssertNoThrow(try BilinearRuntimeCapability.decode(JSONEncoder().encode(entry.runtimeContract)))
        let descriptor = try BuiltInCapabilityBridge.bilinearDeclaration(definition)
        XCTAssertEqual(descriptor.version, ContractVersion())
        XCTAssertEqual(descriptor.learning.lessonIDs, ["assessment.bilinear-with-bias", "assessment.bilinear-without-bias"])
        XCTAssertEqual(descriptor.lifecycle.state, [StateDeclaration("parameters", kind: .parameters)])
        for task in BilinearLearningTask.allCases {
            let privateDefinition = try XCTUnwrap(BilinearLearningCanvas.registry(task: task).component(type: "bilinear"))
            XCTAssertEqual(privateDefinition.inputs, task == .withBias ? ["a","b","weight","bias"] : ["a","b","weight"])
            XCTAssertEqual(privateDefinition.fields.map(\.key), ["bias"])
            XCTAssertNotEqual(privateDefinition, definition)
        }
        var wrong = definition; wrong.id = "not-bilinear"
        XCTAssertThrowsError(try BuiltInCapabilityBridge.bilinearDeclaration(wrong))
    }
}
