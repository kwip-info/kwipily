import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModuleCompositionLearnCatalogTests:XCTestCase {
    func testPairedTasksAndPreparedExamplesUseExistingExactRoutes()throws {
        XCTAssertEqual(LearnCatalogContent.compositionPracticeTasks.map(\.rawValue),["composition-build-reuse","composition-order-name"])
        XCTAssertEqual(LearnCatalogContent.compositionExampleIDs,["prepared-composition-build-reuse","prepared-composition-order-name"])
        let lab=try XCTUnwrap(LearnCatalogContent.items.first {$0.id=="module-composition"})
        let examples=try XCTUnwrap(LearnCatalogContent.items.first {$0.id=="module-composition-examples"})
        XCTAssertEqual(lab.kind,"Construction");XCTAssertEqual(examples.kind,"Example")
        XCTAssertEqual(lab.category,"Layers & activations");XCTAssertEqual(examples.category,lab.category)
        for raw in ["composition-build-reuse","composition-order-name"] {
            XCTAssertTrue(lab.matches(section:"Library",search:raw))
            XCTAssertTrue(examples.matches(section:"Examples",search:"prepared-"+raw))
        }
        XCTAssertFalse(lab.matches(section:"Examples",search:"block"))
        XCTAssertFalse(examples.matches(section:"Practice",search:"block"))
    }
    @MainActor func testApproachableSearchTermsAndPriorCatalogIdentitiesRemainDistinct()throws {
        let lab=try XCTUnwrap(LearnCatalogContent.items.first {$0.id=="module-composition"})
        for term in ["Module","Sequential","ModuleList","ModuleDict","private copies","order","names","influence"] {
            XCTAssertTrue(lab.matches(section:"Library",search:term),term)
        }
        XCTAssertTrue(LearnCatalogContent.compositionPracticeSummary.contains("No code"))
        XCTAssertTrue(LearnCatalogContent.compositionPracticeSummary.contains("connections"))
        let items=LearnCatalogContent.items
        XCTAssertEqual(items.count,Set(items.map(\.id)).count)
        for (id,category) in [("recurrent-memory","Layers & activations"),("tensor-routing","Tensors & linear algebra")] {
            XCTAssertEqual(items.first {$0.id==id}?.category,category)
        }
        for id in ["gradient-controls","loss-results"] {
            XCTAssertNil(items.first {$0.id==id}, "Inline Train configuration must not restore duplicate public recipe lessons")
            let historical = try XCTUnwrap(LearnCatalogContent.historicalItems.first {$0.id==id})
            XCTAssertEqual(historical.category,"Training & evaluation")
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:id), "Saved prior lesson identities remain readable")
        }
        let train = try XCTUnwrap(items.first {$0.id=="train-data-access"})
        XCTAssertEqual(train.category,"Training & evaluation")
        for term in ["optimizer","objective","gradients","batches"] {
            XCTAssertTrue(train.matches(section:"Training & evaluation",search:term),term)
        }
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"train"),"train-data-access")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"gradient_policy"),"train-data-access")
        XCTAssertTrue(Set(items.map(\.id)).isDisjoint(with:Set(LearnCatalogContent.historicalItems.map(\.id))))
    }
}
