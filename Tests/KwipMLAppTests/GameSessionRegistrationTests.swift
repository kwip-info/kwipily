import XCTest
@testable import KwipMLApp
import KwipMLCore
@MainActor final class GameSessionRegistrationTests:XCTestCase {
    func testCreationLessonAndRequiredArtworkStayConnected()throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let data=try DataFlowOperatorRegistry.load(from:root.appendingPathComponent("runner/dataflow-operators.json"))
        let entries=try WorkflowCanvasAdapter.catalog(modelRegistry:ComponentRegistry.builtIn(),dataRegistry:data)
        let games=entries.filter{if case .gameSession=$0.payload{return true};return false}
        XCTAssertEqual(games.count,1);XCTAssertEqual(games.first?.iconType,"workflow.gameSession")
        XCTAssertEqual(ComponentLessonDestination.entryID(type:"workflow.gameSession"),GameEpisodePractice.lessonID)
        XCTAssertNotNil(LearnCatalogContent.items.first{$0.id==GameEpisodePractice.lessonID})
        XCTAssertEqual(ComponentArtworkCatalog.assets["workflow.gameSession"],"game-session-v1")
        XCTAssertEqual(ComponentVisualFamily.resolve(type:"workflow.gameSession"),.evaluation)
    }
}
extension GameSessionRegistrationTests {
    func testActualLessonLoadUsesSessionPlanAndNoSourceGrant()throws {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent(),storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(GameEpisodePractice.lessonID,device:"cpu"),workspace.statusMessage)
        XCTAssertEqual(workspace.workflowSession?.owner.document.schemaVersion,22)
    }
}
extension GameSessionRegistrationTests {
    func testActualManagedSessionPublishesVerifiedTablesAndHistory()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(GameEpisodePractice.lessonID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),runtime=try XCTUnwrap(workspace.runtime)
        let env=GameEpisodeSession.Environment(root:workspace.workspaceDirectory,accounting:workspace.applicationDirectory,runtime:runtime,policy:try workspace.executionPolicy(for:host.owner.projectID))
        var acquired=false,released=false
        let session=GameEpisodeSession(document:host.owner.document,address:GameEpisodePractice.session,validateOnly:false,environment:env,reservation:.init(acquire:{acquired=true;return UUID()},release:{_ in released=true}))
        let done=expectation(description:"real managed game session")
        session.start(modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry,references:[:],isCurrent:{true},onFinish:{done.fulfill()})
        await fulfillment(of:[done],timeout:30)
        XCTAssertTrue(acquired && released);XCTAssertFalse(session.isBusy)
        XCTAssertEqual(Set(session.artifacts.keys),["summary","episodes","moves"],session.message)
        XCTAssertEqual(session.summary["terminal"],8,session.message)
        let records=try GameEpisodeRecord.list(root:workspace.workspaceDirectory,projectID:host.owner.projectID)
        let record=try XCTUnwrap(records.first,session.message)
        XCTAssertEqual(try record.verify(root:workspace.workspaceDirectory).count,3)
        let path=try record.directory(workspace.workspaceDirectory).appendingPathComponent("artifacts/moves.table.json")
        try Data("{}".utf8).write(to:path)
        XCTAssertThrowsError(try record.verify(root:workspace.workspaceDirectory))
    }
}

extension GameSessionRegistrationTests {
    func testHostCompletionKeepsDownstreamViewsCurrent()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(GameEpisodePractice.lessonID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let session=try XCTUnwrap(host.startGameSession(GameEpisodePractice.session),host.statusMessage)
        for _ in 0..<300 {
            if !session.isBusy && !host.hasActiveOperation {break}
            try await Task.sleep(for:.milliseconds(100))
        }
        XCTAssertFalse(host.hasActiveOperation,host.statusMessage)
        XCTAssertEqual(session.summary["terminal"],8,session.message)
        XCTAssertEqual(host.terminalStatus(for:GameEpisodePractice.session),"Completed")
        let address=WorkflowNodeAddress(domain:.data,localID:"episodes")
        XCTAssertNotNil(host.snapshots(for:[address])[address],host.statusMessage)
        XCTAssertNotNil(host.automationTableViewSession(for:.init(address:address,port:"table")),host.statusMessage)
    }
}
