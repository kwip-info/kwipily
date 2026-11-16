import XCTest
@testable import KwipMLCore

final class SchedulerDeclarationTests:XCTestCase {
    func testRecipesDeclareTrainingOnlyStateAndDistinctLearning()throws {
        let factories:[(String,(ComponentDefinition)throws->CapabilityDescriptor)]=[
            ("scheduler_step_lr",SchedulerCapabilityBridge.stepLRDeclaration),
            ("scheduler_cosine_annealing_lr",SchedulerCapabilityBridge.cosineAnnealingLRDeclaration),
            ("scheduler_reduce_on_plateau",SchedulerCapabilityBridge.reduceOnPlateauDeclaration),
            ("scheduler_one_cycle_lr",SchedulerCapabilityBridge.oneCycleLRDeclaration),
            ("scheduler_linear_lr",SchedulerCapabilityBridge.linearLRDeclaration)]
        var lessons=Set<String>()
        for (id,factory) in factories {
            let definition=BundledSchedulerComponents.definition(id)
            let descriptor=try factory(definition)
            XCTAssertTrue(CapabilityContractValidator.validate(descriptor).issues.isEmpty)
            XCTAssertEqual(descriptor.lifecycle.fitBoundary,.trainingPartitionOnly)
            XCTAssertEqual(descriptor.lifecycle.state.map(\.key),["schedulerRecipe"])
            XCTAssertEqual(descriptor.ports.map(\.id),["out"])
            XCTAssertEqual(descriptor.ports[0].schema.kind,.state)
            XCTAssertEqual(descriptor.ports[0].schema.state?.key,"schedulerRecipe")
            XCTAssertEqual(descriptor.ports[0].schema.state?.kind,.scheduler)
            XCTAssertFalse(descriptor.lifecycle.phases.contains(.forward))
            XCTAssertFalse(descriptor.lifecycle.phases.contains(.backward))
            XCTAssertEqual(descriptor.lifecycle.state.map(\.kind),[.scheduler])
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
        XCTAssertTrue(SchedulerRuntimeCapabilities().componentIDs.isEmpty)
        var descriptors:[String:Data]=[:]
        for id in SchedulerRuntimeCapabilities.operations {
            descriptors[id]=try JSONEncoder().encode(BundledSchedulerComponents.entries[id]!.runtimeContract)
        }
        XCTAssertEqual(SchedulerRuntimeCapabilities(descriptors:descriptors).componentIDs,SchedulerRuntimeCapabilities.operations)
        let id="scheduler_one_cycle_lr"
        var object=try XCTUnwrap(try JSONSerialization.jsonObject(with:descriptors[id]!) as? [String:Any])
        object["requiresExplicitOptimizer"]=false
        descriptors[id]=try JSONSerialization.data(withJSONObject:object)
        XCTAssertFalse(SchedulerRuntimeCapabilities(descriptors:descriptors).contains(id))
        XCTAssertThrowsError(try SchedulerRuntimeCapabilities.verify(descriptors[id]!,componentID:id))
        XCTAssertThrowsError(try SchedulerRuntimeCapabilities.verify(Data(repeating:32,count:8193),componentID:id))
    }
}
