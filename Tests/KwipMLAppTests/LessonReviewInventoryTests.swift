import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Opt-in local authoring projection. No projects, learner progress or private fixtures are read.
@MainActor final class LessonReviewInventoryTests:XCTestCase {
    func testExportCurrentDiscoveryAndAssessmentQueue()throws {
        guard let destination=ProcessInfo.processInfo.environment["KWIPML_LESSON_REVIEW_INVENTORY"] else {throw XCTSkip("Local review inventory export is opt-in")}
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let output=URL(fileURLWithPath:destination).standardizedFileURL
        XCTAssertTrue(output.path.hasPrefix(root.appendingPathComponent(".local/lesson-review/").path))
        guard output.path.hasPrefix(root.appendingPathComponent(".local/lesson-review/").path) else {return}
        let entries=LearningLibraryPresentation.items
        XCTAssertEqual(Set(entries.map(\.id)).count,entries.count)
        let rows:[[String:Any]]=entries.map {item in
            ["entryID":item.id,"title":item.title,"topic":item.category,"type":item.kind,"backend":item.backend,
             "assessments":(ConstructionLessonCatalog.groups[item.id] ?? []).map {["operation":$0.id,"title":$0.title]},
             "affineFamily":ConstructionLessonCatalog.affineEntries.contains(item.id)]
        }
        try FileManager.default.createDirectory(at:output.deletingLastPathComponent(),withIntermediateDirectories:true)
        let packet:[String:Any]=["schema":1,"scope":"Current discovery only; no approval or native acceptance implied. Affine family tasks are expanded separately in the owner review ledger.","entries":rows]
        try JSONSerialization.data(withJSONObject:packet,options:[.prettyPrinted,.sortedKeys]).write(to:output,options:.atomic)
    }
}
