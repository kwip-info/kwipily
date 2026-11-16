import XCTest
@testable import KwipMLCore

final class LossGenericExampleRoutingTests:XCTestCase {
    private let resources:[String:String] = [
        "loss_cross_entropy":"loss-cross-entropy-observations",
        "loss_bce_with_logits":"loss-bce-observations",
        "loss_mse":"loss-mse-observations",
        "loss_l1":"loss-l1-observations",
        "loss_nll":"loss-nll-observations",
        "loss_kl_div":"loss-kl-observations",
        "loss_ctc":"loss-ctc-observations",
        "loss_triplet_margin":"loss-triplet-observations"
    ]
    func testGenericComponentRouteCompilesEveryRealObjective() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let registry=try ComponentRegistry.builtIn()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        for id in resources.keys.sorted() {
            do {
                let document=try PreparedWorkflowExample.component(id).document(device:"cpu")
                let plan=try WorkflowCompiler.compile(document,train:WorkflowExamples.trainAddress,modelRegistry:registry,dataRegistry:data,contractVersion:12)
                XCTAssertEqual(document.schemaVersion,14,id)
                XCTAssertEqual(plan.lossRecipe?.configuration.componentID,id)
                XCTAssertNil(plan.targetColumn,id)
                XCTAssertEqual(try WorkflowCodec.decode(WorkflowCodec.encode(document)),document,id)
            } catch {XCTFail("\(id): generic example route failed: \(error)")}
        }
    }
    func testGenericRouteBindsEachObjectivesActualObservationResource() throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        for (id,name) in resources {
            let example=PreparedWorkflowExample.component(id)
            XCTAssertEqual(example.observationsResource,name,id)
            XCTAssertEqual(example.sourceResources.count,1,id)
            XCTAssertEqual(example.sourceResources.first?.address,WorkflowExamples.sourceAddress,id)
            XCTAssertEqual(example.sourceResources.first?.resource,name,id)
            let actual=root.appendingPathComponent("Sources/KwipMLApp/Resources/\(name).csv")
            XCTAssertTrue(FileManager.default.fileExists(atPath:actual.path),id)
        }
        XCTAssertEqual(PreparedWorkflowExample.component("max_pool2d").observationsResource,"workflow-observations")
        XCTAssertThrowsError(try PreparedWorkflowExample.component("unknown_objective").document(device:"cpu"))
    }
}
