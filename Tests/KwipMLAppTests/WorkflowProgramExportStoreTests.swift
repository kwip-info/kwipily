import XCTest
import Combine
@testable import KwipMLCore
@testable import KwipMLApp

private actor ProgramExportBarrier:WorkflowProgramExportIO {
    var waiter:CheckedContinuation<WorkflowProgramExportPackage,Error>?
    let reached:XCTestExpectation?
    init(_ reached:XCTestExpectation?) {self.reached=reached}
    func stage(reference:WorkflowHistoryReference,root:URL,runtime:RuntimeLocation,expected:WorkflowProgramExportIdentity,configuration:ProgramExportConfiguration,executionBudget:ResolvedExecutionBudget?)async throws->WorkflowProgramExportPackage {
        try await withCheckedThrowingContinuation {waiter=$0;reached?.fulfill()}
    }
    func publish(_ package:WorkflowProgramExportPackage,to destination:URL)async throws {try package.publish(to:destination)}
    func finish(_ value:WorkflowProgramExportPackage) {waiter?.resume(returning:value);waiter=nil}
}

@MainActor final class WorkflowProgramExportStoreTests:XCTestCase {
    private func fixture()throws->WorkflowProgramExportPackage {
        let(root,identity,config,_,sourcePin)=try WorkflowProgramExportTests.fixture()
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        return .init(directory:root,cleanupDirectory:root,identity:identity,sourceManifestSHA256:sourcePin,configuration:config,torchVersion:"2.14.1",inputDtype:"float32",inputSampleShape:[2],outputSampleShape:[3],files:[],manifestBytes:try Data(contentsOf:root.appendingPathComponent("manifest.json")))
    }
    private func store(_ package:WorkflowProgramExportPackage,_ io:ProgramExportBarrier,current:@escaping()->Bool={true},release:@escaping(UUID)->Void={_ in})->WorkflowProgramExportStore {
        let runtime=RuntimeLocation(kind:.development,packageVersions:[:],executableURL:URL(fileURLWithPath:"/unused"),entryURL:URL(fileURLWithPath:"/unused"),registryURL:URL(fileURLWithPath:"/unused"),workingDirectoryURL:package.directory)
        return .init(reference:.init(projectID:UUID().uuidString,directoryID:UUID()),root:package.directory,runtime:runtime,expected:package.identity,reservation:.init(acquire:{UUID()},release:release),io:io,isCurrent:current)
    }
    private func idle(_ store:WorkflowProgramExportStore)async {
        if !store.isBusy {return}
        let done=expectation(description:"Operation terminal")
        let sub=store.$isBusy.filter{!$0}.prefix(1).sink{_ in done.fulfill()}
        await fulfillment(of:[done],timeout:3);sub.cancel()
    }
    func testInvalidDraftNeverPartiallyChangesCommittedConfigurationAndDiscardRestores()throws {
        let package=try fixture(),io=ProgramExportBarrier(nil),store=store(package,io)
        store.draft["maximum_batch_size"]=9
        XCTAssertFalse(store.applyDraft());XCTAssertEqual(store.configuration.values,ProgramExportConfiguration.defaults)
        store.discardDraft();XCTAssertFalse(store.hasDraftChanges)
        store.draft["batch_mode"] = .string("bounded");store.draft["example_batch_size"]=3;store.draft["minimum_batch_size"]=1;store.draft["maximum_batch_size"]=9
        XCTAssertTrue(store.applyDraft());XCTAssertEqual(store.configuration.maximumBatchSize,9);XCTAssertFalse(store.hasDraftChanges)
    }
    func testCancellationRetainsReservationUntilStagingActuallyReturnsAndDropsLatePackage()async throws {
        let package=try fixture(),reached=expectation(description:"Staging entered"),io=ProgramExportBarrier(reached)
        var releases=0
        let store=store(package,io,release:{_ in releases+=1})
        guard store.prepare() else {XCTFail("Valid current record must begin staging");return}
        await fulfillment(of:[reached],timeout:3)
        store.cancel();XCTAssertTrue(store.isBusy);XCTAssertEqual(releases,0)
        await io.finish(package);await idle(store)
        XCTAssertNil(store.package);XCTAssertEqual(releases,1);XCTAssertFalse(FileManager.default.fileExists(atPath:package.directory.path))
    }
    func testChangedSelectionRejectsLatePackageAndUnappliedDraftBlocksPreparation()async throws {
        let package=try fixture(),reached=expectation(description:"Staging entered"),io=ProgramExportBarrier(reached)
        var current=true
        let store=store(package,io,current:{current})
        store.draft["strict"]=false;XCTAssertFalse(store.prepare());store.discardDraft()
        guard store.prepare() else {XCTFail("Valid committed configuration must begin staging");return}
        await fulfillment(of:[reached],timeout:3);current=false;await io.finish(package);await idle(store)
        XCTAssertNil(store.package);XCTAssertFalse(store.save(to:package.directory));XCTAssertFalse(store.prepare())
    }
}
