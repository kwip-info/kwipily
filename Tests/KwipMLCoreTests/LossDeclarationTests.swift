import XCTest
@testable import KwipMLCore

final class LossDeclarationTests:XCTestCase {
    private let factories:[(String,(ComponentDefinition)throws->CapabilityDescriptor)] = [
        ("loss_cross_entropy",LossCapabilityBridge.crossEntropyDeclaration),
        ("loss_bce_with_logits",LossCapabilityBridge.bceWithLogitsDeclaration),
        ("loss_mse",LossCapabilityBridge.mseDeclaration),("loss_l1",LossCapabilityBridge.l1Declaration),
        ("loss_nll",LossCapabilityBridge.nllDeclaration),("loss_kl_div",LossCapabilityBridge.klDivDeclaration),
        ("loss_ctc",LossCapabilityBridge.ctcDeclaration),("loss_triplet_margin",LossCapabilityBridge.tripletMarginDeclaration)]
    func testObjectiveMetadataDeclaresScoringRecipeNotFittingOrForwardState()throws {
        XCTAssertEqual(Set(BundledLossComponents.entries.keys),LossConfiguration.componentIDs)
        for (id,_) in factories {
            let definition=BundledLossComponents.definition(id)
            let descriptor=try LossCapabilityBridge.reviewedDescriptor(definition)
            XCTAssertTrue(CapabilityContractValidator.validate(descriptor).issues.isEmpty)
            XCTAssertEqual(descriptor.ports.map(\.id),["out"])
            XCTAssertEqual(descriptor.ports[0].schema.kind,.state)
            XCTAssertEqual(descriptor.ports[0].schema.state?.kind,.objective)
            XCTAssertEqual(descriptor.ports[0].schema.state?.key,"objectiveRecipe")
            XCTAssertEqual(descriptor.ports[0].phases,[.train,.evaluate])
            XCTAssertEqual(descriptor.lifecycle.phases,[.initialize,.train,.evaluate,.checkpoint,.resume,.export,.reset])
            XCTAssertEqual(descriptor.lifecycle.fitBoundary,.notApplicable)
            XCTAssertEqual(descriptor.lifecycle.state.map(\.kind),[.objective])
            XCTAssertEqual(descriptor.lifecycle.state.map(\.key),["objectiveRecipe"])
            XCTAssertEqual(descriptor.lifecycle.resume,.none)
            XCTAssertFalse(descriptor.lifecycle.phases.contains(.forward))
            XCTAssertFalse(descriptor.lifecycle.phases.contains(.backward))
            XCTAssertEqual(descriptor.devices,[.cpu,.mps])
            XCTAssertEqual(descriptor.adapterID,"kwip.runner.objective.\(id)")
            XCTAssertEqual(descriptor.fields,definition.fields)
            var altered=definition;altered.inputs=["in"]
            XCTAssertThrowsError(try LossCapabilityBridge.reviewedDescriptor(altered))
            altered=definition;altered.fields.removeLast()
            XCTAssertThrowsError(try LossCapabilityBridge.reviewedDescriptor(altered))
        }
    }
    func testObjectiveBoundaryExemptionCannotHideFittingOrLearnedTensorState()throws {
        let baseline=try LossCapabilityBridge.reviewedDescriptor(BundledLossComponents.definition("loss_mse"))
        func refuses(_ forged:CapabilityDescriptor,file:StaticString = #filePath,line:UInt = #line) {
            XCTAssertTrue(CapabilityContractValidator.validate(forged).issues.contains {$0.code == "fit.boundary"},file:file,line:line)
        }
        var forged=baseline;forged.lifecycle.phases.append(.fit);refuses(forged)
        forged=baseline;forged.ports[0].schema = .tensor(TensorContract([.symbol("batch"),.fixed(2)],backend:.pytorch));refuses(forged)
        forged=baseline;forged.ports[0].schema = .state(StatePortContract(key:"objectiveRecipe",kind:.optimizer));refuses(forged)
        forged=baseline;forged.lifecycle.state.append(StateDeclaration("weights",kind:.parameters));refuses(forged)
        forged=baseline;forged.lifecycle.state.append(StateDeclaration("fixedWeights",kind:.buffers));refuses(forged)
        forged=baseline;forged.ports[0].direction = .input;refuses(forged)
        forged=baseline;forged.ports.append(forged.ports[0]);refuses(forged)
        forged=baseline;forged.lifecycle.phases.append(.forward);refuses(forged)
        for id in ["optimizer_sgd","scheduler_step_lr"] {
            var other:CapabilityDescriptor
            if id == "optimizer_sgd" {other = try OptimizerCapabilityBridge.sgdDeclaration(BundledOptimizerComponents.definition(id))}
            else {other = try SchedulerCapabilityBridge.stepLRDeclaration(BundledSchedulerComponents.definition(id))}
            other.lifecycle.fitBoundary = .notApplicable;refuses(other)
        }
    }
    func testDeclarationPublicationGateCannotBeBypassedByMetadataReview()throws {
        for (id,factory) in factories {
            let entry=try XCTUnwrap(BundledLossComponents.metadata(id))
            _ = try LossCapabilityBridge.reviewedDescriptor(entry.definition)
            if entry.published {
                XCTAssertEqual(try factory(entry.definition).id,id)
            } else {
                XCTAssertThrowsError(try factory(entry.definition))
                XCTAssertNil(BundledLossComponents.declarations[id])
                XCTAssertNil(BundledLossComponents.examples[id])
            }
        }
        XCTAssertEqual(try JSONDecoder().decode(StateKind.self,from:Data("\"objective\"".utf8)),.objective)
    }
}
