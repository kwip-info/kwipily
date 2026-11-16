import XCTest
@testable import KwipMLApp
import KwipMLCore
@MainActor final class GameRuleRegistrationTests:XCTestCase {
    func testLessonAndInformationResolveToActualPractice()throws {
        let entry=try XCTUnwrap(LearnCatalogContent.items.first{$0.id==GameRuleConfiguration.lessonID})
        XCTAssertEqual(entry.kind,"Practice")
        XCTAssertEqual(PreparedWorkflowExample.component(GameRuleConfiguration.lessonID).sourceResources.count,0)
        XCTAssertEqual(try PreparedWorkflowExample.component(GameRuleConfiguration.lessonID).document().nodes.count,3)
        XCTAssertEqual(ComponentLessonDestination.entryID(type:GameRuleConfiguration.operatorID),GameRuleConfiguration.lessonID)
    }
    func testTableProjectionUsesDeclaredPortsWithoutRequiringFileInput()throws {
        let doc=try GameRuleConfiguration.document()
        let graph=DataFlowDocument(nodes:doc.nodes.compactMap {n in if case .dataOperator(let id,let c)=n.payload{return .init(id:n.address.localID,operatorID:id,config:c)};return nil},edges:doc.edges.map{.init(from:.init(node:$0.from.address.localID,port:$0.from.port),to:.init(node:$0.to.address.localID,port:$0.to.port))})
        let columns=DataNodeSchemaProjection.upstreamSchema(for:"tests",in:graph,sourceSchema:{_ in []})
        XCTAssertEqual(columns.map(\.name),["test","status","detail","moves","definition"])
        XCTAssertTrue(DataFlowWorkspaceStore.editableTypes.contains(GameRuleConfiguration.operatorID))
    }
}
extension GameRuleRegistrationTests {
    func testSourceFreeRuleProvenanceAcceptsExactDefinitionAndRejectsForgery()throws {
        let file=URL(fileURLWithPath:#filePath).deletingLastPathComponent().appendingPathComponent("Fixtures/game-rule-preview.json")
        let fixture=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:file))
        let graph=try JSONDecoder().decode(DataFlowDocument.self,from:JSONEncoder().encode(fixture["graph"]!))
        guard case .object(let event)?=fixture["event"] else{return XCTFail("Missing fixture event")}
        let identity=PreparationObservationIdentity(requestID:"test",graphSHA256:String(repeating:"a",count:64),sourceBindingsSHA256:String(repeating:"b",count:64),revision:"revision1")
        let kinds=["rules":Dictionary(uniqueKeysWithValues:["definition","summary","tests","trace"].map{($0,"table")})]
        var ledger=try PreparationObservationLedger(identity:identity,graph:graph,outputKinds:kinds)
        XCTAssertTrue(ledger.nodeStarted("rules",identity:identity));XCTAssertTrue(ledger.consume(event,identity:identity))
        var altered=graph;altered.nodes[0].config["definition"] = .object(try GameRuleConfiguration.example("placement"))
        var forged=try PreparationObservationLedger(identity:identity,graph:altered,outputKinds:kinds)
        XCTAssertTrue(forged.nodeStarted("rules",identity:identity));XCTAssertFalse(forged.consume(event,identity:identity))
    }
}
extension GameRuleRegistrationTests {
    func testCanvasHistoryReloadsRemovedDefinitionFieldsWithoutRewritingHistory()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:root)}
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:true)
        var first=try GameRuleConfiguration.document()
        let journal=try CanvasVersionJournal(root:root,projectID:first.projectID)
        let a=try journal.save(first)
        first.nodes[0].payload = .dataOperator(operatorID:GameRuleConfiguration.operatorID,config:["definition":.object(try GameRuleConfiguration.example("placement"))])
        let b=try journal.save(first)
        let originalBytes=try Data(contentsOf:journal.fileURL)
        let reopened=try CanvasVersionJournal(root:root,projectID:first.projectID)
        XCTAssertEqual(try reopened.document(version:b.id),first)
        XCTAssertEqual(try reopened.document(version:a.id).nodes[0].payload,.dataOperator(operatorID:GameRuleConfiguration.operatorID,config:["definition":.object(try GameRuleConfiguration.example("take-away"))]))
        XCTAssertEqual(try Data(contentsOf:journal.fileURL),originalBytes)
    }
}
