import XCTest
import KwipMLCore
@testable import KwipMLApp

final class Data02PracticeRoutesTests: XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    func testSixAuthoredLessonsRemainUnscoredAndBundleBytesMatchSources() throws {
        let catalog=try DataEngineeringPractice.catalog()
        XCTAssertEqual(catalog.lessons.count,6); XCTAssertEqual(catalog.sharedGuidance.count,3)
        XCTAssertEqual(Set(catalog.lessons.map(\.operatorID)),Set(["kwip.data.filterRows", "kwip.data.formula", "kwip.data.outputDelimited"]))
        XCTAssertEqual(try Data(contentsOf:DataEngineeringPractice.resourceURL("data02-lessons.json")),try Data(contentsOf:repository.appendingPathComponent("runner/dataflow_examples/data02-lessons.json")))
        for lesson in catalog.lessons {
            XCTAssertFalse(lesson.isAssessed); XCTAssertEqual(lesson.mode,"guidedPractice")
            XCTAssertTrue(lesson.disclosure.contains("does not award completion"))
            for file in [lesson.graph,lesson.source,lesson.reading] {
                XCTAssertEqual(try Data(contentsOf:DataEngineeringPractice.resourceURL(file)),try Data(contentsOf:repository.appendingPathComponent("runner/dataflow_examples/"+file)))
            }
        }
        XCTAssertThrowsError(try DataEngineeringPractice.resourceURL("../private.csv"))
        XCTAssertThrowsError(try DataEngineeringPractice.lesson("not-a-lesson"))
    }
    @MainActor func testEveryPracticeDocumentCompilesAsPureDataAndPreservesAuthoredConfiguration() throws {
        let registry=try DataFlowOperatorRegistry.load(from:repository.appendingPathComponent("runner/dataflow-operators.json"))
        for lesson in try DataEngineeringPractice.catalog().lessons {
            let document=try DataEngineeringPractice.document(lesson.id,projectID:UUID().uuidString)
            let graph=try WorkflowDataPreparationPlan.compile(document,registry:registry)
            let authored=try DataFlowCodec.decodeGraph(Data(contentsOf:DataEngineeringPractice.resourceURL(lesson.graph)))
            XCTAssertEqual(graph,authored)
            XCTAssertEqual(document.name,lesson.title)
            XCTAssertTrue(document.nodes.allSatisfy {$0.address.domain == .data})
            XCTAssertFalse(try String(decoding:WorkflowCodec.encode(document),as:UTF8.self).contains(repository.path))
        }
    }
    @MainActor func testCurrentCatalogDestinationsAndHistoricalWindowRoutesRemainAvailable() throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:root) }
        let store=WorkspaceStore(rootOverride:repository,storageOverride:root,detectCapabilities:false)
        var routed:[String]=[]
        store.exampleInWindow={ _,kind in
            if case .component(let id)=kind {routed.append(id); return true}; return false
        }
        let consolidated: [String:String] = ["data02-calculate":"formula", "data02-regex-extract":"formula", "data02-regex-replace":"formula", "data02-output":"table-output"]
        for lesson in try DataEngineeringPractice.catalog().lessons {
            let destination = consolidated[lesson.id] ?? lesson.id
            let toolDestination = lesson.operatorID == "kwip.data.filterRows" ? "filter" : lesson.operatorID == "kwip.data.formula" ? "formula" : "table-output"
            XCTAssertEqual(ComponentLessonDestination.entryID(type:lesson.operatorID), toolDestination, lesson.id)
            XCTAssertTrue(ComponentLessonDestination.isAvailable(entryID:lesson.id), "Historical lesson links must remain readable: \(lesson.id)")
            if consolidated[lesson.id] != nil {
                XCTAssertTrue(LearnCatalogContent.historicalEntryIDs.contains(lesson.id))
                XCTAssertFalse(LearnCatalogContent.items.contains {$0.id == lesson.id}, "Consolidation must not restore duplicate public lessons")
                let item = try XCTUnwrap(LearnCatalogContent.items.first {$0.id == destination})
                XCTAssertEqual(item.kind,"Practice")
                XCTAssertTrue(item.matches(section:"Data preparation",search:destination == "formula" ? "Formula":"Table Output"))
            } else {
                XCTAssertTrue(LearnCatalogContent.items.contains {$0.id == lesson.id && $0.kind == "Guided" && $0.matches(section:"Data preparation",search:lesson.title)})
            }
            // Stable lesson links open the rewritten current-tool graph with the same exercise ID.
            XCTAssertTrue(store.loadDataEngineeringPracticeExample(lesson.id))
        }
        XCTAssertEqual(routed,try DataEngineeringPractice.catalog().lessons.map(\.id))
        XCTAssertFalse(store.loadDataEngineeringPracticeExample("not-a-lesson"))
    }
}
