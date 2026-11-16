import XCTest
@testable import KwipMLCore

final class OptimizerDeclarationTests:XCTestCase {
    func testRecipesDeclareTrainingOnlyStateAndDistinctLearning()throws {
        let factories:[(String,(ComponentDefinition)throws->CapabilityDescriptor)]=[
            ("optimizer_sgd",OptimizerCapabilityBridge.sgdDeclaration),
            ("optimizer_adam",OptimizerCapabilityBridge.adamDeclaration),
            ("optimizer_adamw",OptimizerCapabilityBridge.adamwDeclaration),
            ("optimizer_rmsprop",OptimizerCapabilityBridge.rmspropDeclaration),
            ("optimizer_adagrad",OptimizerCapabilityBridge.adagradDeclaration)]
        var lessons=Set<String>()
        for (id,factory) in factories {
            let definition=BundledOptimizerComponents.definition(id)
            let descriptor=try factory(definition)
            XCTAssertTrue(CapabilityContractValidator.validate(descriptor).issues.isEmpty)
            XCTAssertEqual(descriptor.lifecycle.fitBoundary,.trainingPartitionOnly)
            XCTAssertEqual(descriptor.lifecycle.state.map(\.key),["optimizerRecipe"])
            XCTAssertEqual(descriptor.ports.map(\.id),["out"])
            XCTAssertEqual(descriptor.ports[0].schema.kind,.state)
            XCTAssertEqual(descriptor.ports[0].schema.state?.key,"optimizerRecipe")
            XCTAssertEqual(descriptor.ports[0].schema.state?.kind,.optimizer)
            XCTAssertFalse(descriptor.lifecycle.phases.contains(.forward))
            XCTAssertFalse(descriptor.lifecycle.phases.contains(.backward))
            XCTAssertEqual(descriptor.lifecycle.state.map(\.kind),[.optimizer])
            XCTAssertEqual(descriptor.lifecycle.resume,.none)
            XCTAssertEqual(descriptor.devices,[.cpu,.mps])
            XCTAssertEqual(descriptor.learning.lessonIDs.count,1)
            lessons.formUnion(descriptor.learning.lessonIDs)
            var altered=definition;altered.inputs=["in"]
            XCTAssertThrowsError(try factory(altered))
        }
        XCTAssertEqual(lessons.count,5)
    }
    func testExactOptionalDiscoveryDoesNotInferSupportOrAdmitCounterPolicyDrift()throws {
        XCTAssertTrue(OptimizerRuntimeCapabilities().componentIDs.isEmpty)
        var descriptors:[String:Data]=[:]
        for id in OptimizerRuntimeCapabilities.operations {
            descriptors[id]=try JSONEncoder().encode(BundledOptimizerComponents.entries[id]!.runtimeContract)
        }
        XCTAssertEqual(OptimizerRuntimeCapabilities(descriptors:descriptors).componentIDs,OptimizerRuntimeCapabilities.operations)
        let id="optimizer_adam"
        var object=try XCTUnwrap(try JSONSerialization.jsonObject(with:descriptors[id]!) as? [String:Any])
        object["counterDType"]="int64"
        descriptors[id]=try JSONSerialization.data(withJSONObject:object)
        XCTAssertFalse(OptimizerRuntimeCapabilities(descriptors:descriptors).contains(id))
        XCTAssertThrowsError(try OptimizerRuntimeCapabilities.verify(descriptors[id]!,componentID:id))
        XCTAssertThrowsError(try OptimizerRuntimeCapabilities.verify(Data(repeating:32,count:8193),componentID:id))
    }
}
