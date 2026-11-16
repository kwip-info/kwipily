import XCTest
import KwipMLCore
@testable import KwipMLApp

final class LossLearnCatalogTests:XCTestCase {
    private let expected:Set<String>=["loss_cross_entropy","loss_bce_with_logits","loss_mse","loss_l1","loss_nll","loss_kl_div","loss_ctc","loss_triplet_margin"]
    func testAllEightPublishedLossExamplesAreSearchableAndKeepAuthoredIdentities() throws {
        let entries=LearnCatalogContent.componentExampleEntries.filter {expected.contains($0.componentID)}
        XCTAssertEqual(Set(entries.map(\.componentID)),expected)
        for task in LossLearningTask.allCases {
            let entry=try XCTUnwrap(entries.first {$0.componentID==task.componentID})
            let authored=try XCTUnwrap(BundledLossComponents.entries[task.componentID]?.learning)
            XCTAssertEqual(entry.item.id,task.componentID)
            XCTAssertEqual(entry.item.category,"Training & evaluation")
            XCTAssertEqual(entry.item.kind,"Example")
            XCTAssertEqual(entry.learning.exampleID,authored.exampleID)
            XCTAssertEqual(authored.contentVersion,1,"Original bundle metadata remains unchanged")
            XCTAssertEqual(authored.updatedAt,"2026-10-08")
            XCTAssertEqual(entry.learning.contentVersion,2,"Inline Train presentation is a new public content revision")
            XCTAssertEqual(entry.learning.updatedAt,"2026-10-14")
            XCTAssertEqual(authored.lessonIDs,["assessment."+task.rawValue])
            for term in [task.componentID,task.rawValue,authored.exampleID,"desired outcomes","loss"] {
                XCTAssertTrue(entry.item.matches(section:"Examples",search:term),"\(task.componentID): \(term)")
            }
            XCTAssertFalse(entry.item.matches(section:"Practice",search:task.componentID))
        }
    }
    func testHistoricalScoringMetadataRetainsEightExistingTaskIdentitiesWithApproachableCopy() throws {
        let lab=try XCTUnwrap(LearnCatalogContent.historicalItems.first {$0.id=="loss-results"})
        XCTAssertEqual(lab.title,"Scoring model results lab")
        XCTAssertEqual(lab.category,"Training & evaluation");XCTAssertEqual(lab.kind,"Practice")
        XCTAssertEqual(Set(LearnCatalogContent.lossPracticeTasks.map(\.componentID)),expected)
        XCTAssertEqual(LearnCatalogContent.lossPracticeTasks.map(\.rawValue),LossLearningTask.allCases.map(\.rawValue))
        for task in LossLearningTask.allCases {
            XCTAssertTrue(lab.matches(section:"Practice",search:task.rawValue))
            XCTAssertTrue(lab.matches(section:"Learning paths",search:task.componentID))
        }
        for term in ["desired outcomes","target","sequence","distance","lower"] {XCTAssertTrue(lab.matches(section:"Practice",search:term),term)}
        XCTAssertFalse(lab.matches(section:"Examples",search:"loss"))
        XCTAssertFalse(lab.searchText.localizedCaseInsensitiveContains("mastery"))
    }
    func testGenericExamplesAndPracticeStayDistinctAndPriorCategoriesRemainUnchanged() throws {
        let entries=LearnCatalogContent.componentExampleEntries
        XCTAssertEqual(entries.count,Set(entries.map(\.componentID)).count)
        let displayed=LearnCatalogContent.displayedComponentExampleEntries
        XCTAssertEqual(Set(displayed.filter {expected.contains($0.componentID)}.map(\.componentID)),expected)
        let rendered=LearnCatalogContent.items.map(\.id)+displayed.map { $0.item.id }
        XCTAssertEqual(rendered.count,Set(rendered).count)
        for (id,category) in [("optimizer-updates","Training & evaluation"),("scheduler-rates","Training & evaluation"),("attention-connections","Layers & activations"),("recurrent-memory","Layers & activations"),("tensor-routing","Tensors & linear algebra")] {
            XCTAssertEqual((LearnCatalogContent.items + LearnCatalogContent.historicalItems).first {$0.id==id}?.category,category)
        }
    }
}
