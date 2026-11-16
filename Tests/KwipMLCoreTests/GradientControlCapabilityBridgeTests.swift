import XCTest
@testable import KwipMLCore

final class GradientControlCapabilityBridgeTests: XCTestCase {
    func testReviewedControlsHaveExactPrivateStateAndNoForwardOrModelParameters() throws {
        for id in ["gradient_policy", "trainability"] {
            let definition = BundledGradientControlComponents.definition(id)
            let value = try GradientControlCapabilityBridge.reviewedDescriptor(definition)
            let key = id == "gradient_policy" ? "gradientPolicyRecipe" : "trainabilityRecipe"
            let kind: StateKind = id == "gradient_policy" ? .gradientPolicy : .trainability
            XCTAssertEqual(value.ports, [PortContract("out", direction: .output, schema: .state(StatePortContract(key: key, kind: kind)), phases: [.initialize, .train, .evaluate])])
            XCTAssertEqual(value.fields, definition.fields)
            XCTAssertEqual(value.lifecycle.state, [StateDeclaration(key, kind: kind)])
            XCTAssertEqual(value.lifecycle.resume, .none)
            XCTAssertEqual(value.lifecycle.fitBoundary, .notApplicable)
            XCTAssertEqual(Set(value.devices), Set([ContractDevice.cpu, .mps]))
            XCTAssertEqual(value.adapterID, "kwip.runner." + id)
            XCTAssertFalse(value.lifecycle.phases.contains(.forward))
            XCTAssertFalse(value.lifecycle.phases.contains(.fit))
            let lessons = id == "gradient_policy"
                ? ["assessment.gradient-selected-influence", "assessment.gradient-combine-small-groups"]
                : ["assessment.trainability-keep-settings", "assessment.trainability-next-run"]
            let examples = id == "gradient_policy"
                ? ["prepared-gradient-selected-influence", "prepared-gradient-combine-small-groups"]
                : ["prepared-trainability-keep-settings", "prepared-trainability-next-run"]
            XCTAssertEqual(value.learning.lessonIDs, lessons)
            XCTAssertEqual(value.learning.exampleIDs, examples)
            XCTAssertEqual(value.availability, .existingPOCAdapter)
            XCTAssertEqual(try GradientControlCapabilityBridge.declaration(definition), value)
            var forged = definition; forged.inputs = ["in"]
            XCTAssertThrowsError(try GradientControlCapabilityBridge.reviewedDescriptor(forged))
            forged = definition; forged.outputs = ["out", "gradient"]
            XCTAssertThrowsError(try GradientControlCapabilityBridge.reviewedDescriptor(forged))
        }
        XCTAssertEqual(Set(BundledGradientControlComponents.declarations.keys), Set(["gradient_policy", "trainability"]))
    }

    func testOnlyExactPolicyControlMayAvoidTrainingPartitionFitBoundary() throws {
        for id in ["gradient_policy", "trainability"] {
            let value = try GradientControlCapabilityBridge.reviewedDescriptor(BundledGradientControlComponents.definition(id))
            let result = CapabilityContractValidator.validate(value)
            XCTAssertTrue(result.isValid, "\(id): \(result.issues)")
            var tensor = value; tensor.ports[0].schema = .tensor(TensorContract([.fixed(2)]))
            XCTAssertFalse(CapabilityContractValidator.validate(tensor).isValid)
            var learned = value; learned.lifecycle.state.append(StateDeclaration("learned", kind: .parameters))
            XCTAssertFalse(CapabilityContractValidator.validate(learned).isValid)
            var fit = value; fit.lifecycle.phases.append(.fit)
            XCTAssertFalse(CapabilityContractValidator.validate(fit).isValid)
            var forward = value; forward.lifecycle.phases.append(.forward)
            XCTAssertFalse(CapabilityContractValidator.validate(forward).isValid)
            var sharing = value; sharing.lifecycle.state[0].serializable = false
            XCTAssertFalse(CapabilityContractValidator.validate(sharing).isValid)
        }
    }
}
