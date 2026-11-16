import XCTest
import Combine
@testable import KwipMLCore
@testable import KwipMLApp

private actor EvaluationExportBarrier:WorkflowEvaluationExportIO {
    private var stagedContinuation:CheckedContinuation<WorkflowEvaluationExportPackage,Error>?
    private var save:CheckedContinuation<Void,Error>?
    private var reached:XCTestExpectation
    init(_ reached:XCTestExpectation) {self.reached=reached}
    func stage(reference:WorkflowEvaluationHistoryReference,root:URL,runtime:RuntimeLocation,expected:WorkflowEvaluationExportIdentity,includesTestRows:Bool,executionBudget:ResolvedExecutionBudget?)async throws->WorkflowEvaluationExportPackage {
        try await withCheckedThrowingContinuation {stagedContinuation=$0;reached.fulfill()}
    }
    func publish(_ package:WorkflowEvaluationExportPackage,to destination:URL)async throws {
        try await withCheckedThrowingContinuation {save=$0;reached.fulfill()}
    }
    func next(_ value:XCTestExpectation) {reached=value}
    func finishStage(_ value:WorkflowEvaluationExportPackage) {stagedContinuation?.resume(returning:value);stagedContinuation=nil}
    func finishSave() {save?.resume();save=nil}
}

@MainActor final class WorkflowEvaluationExportStoreTests:XCTestCase {
    private func package()throws->WorkflowEvaluationExportPackage {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at:root,withIntermediateDirectories:false)
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let hash=String(repeating:"a",count:64)
        return .init(directory:root,cleanupDirectory:root,identity:.init(evaluationID:UUID().uuidString,evaluationManifestSHA256:hash,checkpointSHA256:hash,stateDigest:hash,snapshotReceiptSHA256:hash),includesTestRows:false,files:[],manifestBytes:Data(),columns:[],featureColumns:["x","y"],targetColumn:"label",dtype:"float32",sampleShape:[2],testCount:14,fittedStateCount:0)
    }
    private func store(_ package:WorkflowEvaluationExportPackage,_ io:EvaluationExportBarrier,current:@escaping()->Bool={true},release:@escaping(UUID)->Void={_ in})->WorkflowEvaluationExportStore {
        let runtime=RuntimeLocation(kind:.development,packageVersions:[:],executableURL:URL(fileURLWithPath:"/unused"),entryURL:URL(fileURLWithPath:"/unused"),registryURL:URL(fileURLWithPath:"/unused"),workingDirectoryURL:package.directory)
        return WorkflowEvaluationExportStore(reference:.init(projectID:UUID().uuidString,evaluationID:UUID(uuidString:package.identity.evaluationID)!),root:package.directory,runtime:runtime,expected:package.identity,reservation:.init(acquire:{UUID()},release:release),io:io,isCurrent:current)
    }
    private func idle(_ store:WorkflowEvaluationExportStore)async {
        if !store.isBusy {return}
        let done=expectation(description:"Export operation finished")
        let sub=store.$isBusy.filter{!$0}.prefix(1).sink{_ in done.fulfill()}
        await fulfillment(of:[done],timeout:3);sub.cancel()
    }
    func testChangingPrivacyWhileStagingCancelsAndDiscardsLateReviewAfterExit()async throws {
        let package=try package(),reached=expectation(description:"Staging started"),io=EvaluationExportBarrier(reached)
        var releases=0
        let store=store(package,io,release:{_ in releases+=1})
        XCTAssertTrue(store.prepare());await fulfillment(of:[reached],timeout:3)
        store.setIncludesTestRows(true)
        XCTAssertTrue(store.isBusy);XCTAssertEqual(releases,0)
        await io.finishStage(package);await idle(store)
        XCTAssertNil(store.package);XCTAssertTrue(store.includesTestRows);XCTAssertEqual(releases,1)
        XCTAssertFalse(FileManager.default.fileExists(atPath:package.directory.path))
    }
    func testChangedSelectionRejectsLatePackageAndCannotSave()async throws {
        let package=try package(),reached=expectation(description:"Staging started"),io=EvaluationExportBarrier(reached)
        var current=true
        let store=store(package,io,current:{current})
        XCTAssertTrue(store.prepare());await fulfillment(of:[reached],timeout:3)
        current=false;await io.finishStage(package);await idle(store)
        XCTAssertNil(store.package);XCTAssertFalse(store.save(to:package.directory))
        XCTAssertFalse(FileManager.default.fileExists(atPath:package.directory.path))
    }
    func testOptionChangeInvalidatesReadyReviewAndCommittedSaveSurvivesLateCancel()async throws {
        let package=try package(),reached=expectation(description:"Staging started")
        let io=EvaluationExportBarrier(reached),store=store(package,io)
        XCTAssertTrue(store.prepare())
        await fulfillment(of:[reached],timeout:3)
        await io.finishStage(package);await idle(store)
        XCTAssertNotNil(store.package)
        let destination=package.directory.appendingPathComponent("saved")
        let saving=expectation(description:"Saving started");await io.next(saving)
        XCTAssertTrue(store.save(to:destination));await fulfillment(of:[saving],timeout:3)
        store.cancel();XCTAssertTrue(store.isBusy)
        await io.finishSave();await idle(store)
        XCTAssertEqual(store.savedURL,destination);XCTAssertNil(store.package)
    }
    func testPrivacyChangeDropsIdleReview()async throws {
        let package=try package(),reached=expectation(description:"Staging started"),io=EvaluationExportBarrier(reached),store=store(package,io)
        XCTAssertTrue(store.prepare());await fulfillment(of:[reached],timeout:3)
        await io.finishStage(package);await idle(store)
        store.setIncludesTestRows(true)
        XCTAssertNil(store.package);XCTAssertFalse(store.save(to:package.directory))
        XCTAssertFalse(FileManager.default.fileExists(atPath:package.directory.path))
    }
}
