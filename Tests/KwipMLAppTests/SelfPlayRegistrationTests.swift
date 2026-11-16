import XCTest
@testable import KwipMLApp
import KwipMLCore
@MainActor final class SelfPlayRegistrationTests:XCTestCase {
    func testActualManagedLearnerHistoryTablesAndPairedReuse()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        XCTAssertTrue(workspace.loadComponentWorkflowExample(SelfPlayPractice.lessonID,device:"cpu"),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession),runtime=try XCTUnwrap(workspace.runtime)
        let env=SelfPlaySession.Environment(root:workspace.workspaceDirectory,accounting:workspace.applicationDirectory,runtime:runtime,policy:try workspace.executionPolicy(for:host.owner.projectID))
        let session=try XCTUnwrap(host.startSelfPlay(SelfPlayPractice.train),host.statusMessage)
        for _ in 0..<900 {
            if !session.isBusy && !host.hasActiveOperation{break}
            try await Task.sleep(for:.milliseconds(100))
        }
        XCTAssertFalse(host.hasActiveOperation,host.statusMessage)
        let view=WorkflowNodeAddress(domain:.data,localID:"iterations")
        XCTAssertNotNil(host.snapshots(for:[view])[view],host.statusMessage)
        XCTAssertNotNil(host.automationTableViewSession(for:.init(address:view,port:"table")),host.statusMessage)
        XCTAssertEqual(Set(session.artifacts.keys),Set(SelfPlayToolContract.tablePorts),session.message)
        let record=try XCTUnwrap(SelfPlayRecord.list(root:workspace.workspaceDirectory,projectID:host.owner.projectID).first,session.message)
        XCTAssertEqual(try record.verify(root:workspace.workspaceDirectory).count,4)
        XCTAssertFalse(try record.resumeInputs(root:workspace.workspaceDirectory).isEmpty)
        let output=root.appendingPathComponent("paired-export"),pending=try WorkflowExportPublication.stageSelfPlay(record:record,root:workspace.workspaceDirectory,to:output)
        try pending.commit();pending.close();XCTAssertTrue(FileManager.default.fileExists(atPath:output.appendingPathComponent("model.pt").path))
        let game=GameEpisodeSession(document:host.owner.document,address:.init(domain:.control,localID:"session"),validateOnly:false,environment:env,reservation:.init(acquire:{UUID()},release:{_ in}))
        let gameDone=expectation(description:"paired model frozen episodes")
        game.start(modelRegistry:host.owner.modelRegistry,dataRegistry:host.owner.dataRegistry,references:[:],paired:[SelfPlayPractice.train:record],isCurrent:{true},onFinish:{gameDone.fulfill()})
        await fulfillment(of:[gameDone],timeout:30);XCTAssertEqual(game.summary["terminal"],8,game.message)
        for _ in 0..<100 where workspace.isBusy {try await Task.sleep(for:.milliseconds(50))}
        XCTAssertTrue(workspace.beginWorkflowProject(id:host.owner.projectID,forceReload:true),workspace.statusMessage)
        let reopened=try XCTUnwrap(workspace.workflowSession)
        XCTAssertNil(reopened.selfPlay)
        let retainedGame=try XCTUnwrap(reopened.startGameSession(.init(domain:.control,localID:"session")),reopened.statusMessage)
        for _ in 0..<300 {if !retainedGame.isBusy && !reopened.hasActiveOperation {break};try await Task.sleep(for:.milliseconds(100))}
        try await Task.sleep(for:.milliseconds(200))
        XCTAssertEqual(retainedGame.summary["terminal"],8,retainedGame.message)
        XCTAssertFalse(reopened.statusMessage.contains("downstream data:"),reopened.statusMessage)
        XCTAssertFalse(reopened.statusMessage.contains("Run the connected"),reopened.statusMessage)
        let directory=try record.directory(workspace.workspaceDirectory)
        try Data("corrupt".utf8).write(to:directory.appendingPathComponent("artifacts/model.pt"))
        XCTAssertThrowsError(try record.verify(root:workspace.workspaceDirectory))
    }
    func testLessonUsesSharedBlocksAndNoEmptyCanvas()throws {
        XCTAssertNotNil(LearnCatalogContent.items.first{$0.id==SelfPlayPractice.lessonID})
        XCTAssertEqual(LessonContentCatalog.content(entryID:SelfPlayPractice.lessonID)?.steps.count,7)
        XCTAssertTrue(SelfPlayLessonContent.steps.allSatisfy{!$0.blocks.isEmpty})
    }
}

extension SelfPlayRegistrationTests {
    func testActualCampaignChainsSlicesAndKeepsOneReplayStore()async throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer{try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        var document=try SelfPlayPractice.document();document.name="Self-play campaign checkpoint"
        var c=SelfPlayConfiguration.defaults;c["iterations"]=3;c["games"]=2;c["updates"]=2;c["batch"]=4;c["maxMoves"]=24;c["evaluationPairs"]=1
        c["campaign"] = .object(["enabled":true,"sliceIterations":1,"maximumSeconds":600,"maximumBytes":268435456])
        c["compute"]=SelfPlayConfiguration.advancedDefaults["compute"]
        c["evaluation"] = .object(["opponents":.array(["initial","random","uniform-search"].map(JSONValue.string)),"openings":.array([.array([])]),"seed":8301,"checkpoints":.array([])])
        c["search"] = .object(["simulations":4,"reuseTree":true])
        document.nodes[document.nodes.firstIndex(where:{$0.address==SelfPlayPractice.train})!].payload = .train(config:["selfPlay":.object(c)])
        if let export=ProcessInfo.processInfo.environment["KWIPML_SELFPLAY_FIXTURE"] {try WorkflowCodec.encode(document).write(to:URL(fileURLWithPath:export))}
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        let validation=try XCTUnwrap(host.startSelfPlay(SelfPlayPractice.train,validateOnly:true))
        for _ in 0..<300 {if !validation.isBusy{break};try await Task.sleep(for:.milliseconds(100))}
        XCTAssertEqual(validation.message,"Bindings validated · no training run.")
        XCTAssertEqual(validation.artifacts.count,7)
        XCTAssertNotNil(host.startSelfPlay(SelfPlayPractice.train),host.statusMessage)
        let identity=host.operationIdentity
        for _ in 0..<600 {if !host.hasActiveOperation{break};try await Task.sleep(for:.milliseconds(100))}
        XCTAssertFalse(host.hasActiveOperation,host.statusMessage)
        XCTAssertEqual(host.operationIdentity,identity)
        XCTAssertEqual(host.selfPlay?.campaign?["iteration"],3,host.statusMessage)
        XCTAssertEqual(host.selfPlay?.campaign?["complete"],true,host.statusMessage)
        let records=try SelfPlayRecord.list(root:workspace.workspaceDirectory,projectID:document.projectID)
        XCTAssertEqual(records.filter(\.completed).count,3,host.statusMessage)
        XCTAssertEqual(Set(try records.map{try SelfPlayCampaignGrant.captured($0,root:workspace.workspaceDirectory)?["campaignID"]?.stringValue}).count,1)
        for record in records {XCTAssertEqual(try record.verify(root:workspace.workspaceDirectory).count,7)}
        XCTAssertNotNil(host.snapshots(for:[.init(domain:.data,localID:"iterations")])[.init(domain:.data,localID:"iterations")])
        let last=try XCTUnwrap(records.first)
        let retained=try SelfPlayCampaignGrant.boundaries(last,root:workspace.workspaceDirectory)
        XCTAssertEqual(retained.count,3)
        try SelfPlayCampaignGrant.pin(last,root:workspace.workspaceDirectory,boundary:retained[0].id,value:true)
        XCTAssertTrue(try SelfPlayCampaignGrant.boundaries(last,root:workspace.workspaceDirectory)[0].pinned)
        try SelfPlayCampaignGrant.pin(last,root:workspace.workspaceDirectory,boundary:retained[0].id,value:false)
    }
}
