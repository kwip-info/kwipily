import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowExperimentOwnedStoreTests:XCTestCase {
    @MainActor private func fixture(metadataGroupID:String?=nil)throws->([WorkflowExecutionRequest],URL,WorkflowGroupOwner) {
        let helper=WorkflowExecutionRequestTests()
        let(root,authoring,runtime)=try helper.setup()
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(authoring.setExperimentSpecification(spec,for:.init(domain:.control,localID:"setup"),lease:authoring.lease()))
        let base=try helper.capture(authoring,runtime),file=root.appendingPathComponent("input.csv")
        try Data("a,b,label\n1,2,0\n".utf8).write(to:file)
        let inputs=try WorkflowExperimentInputs.capture(addresses:base.plan.sourceAddresses,parent:root) {_ in try WorkflowCapturedSource.capture(file)}
        let requests=try base.experimentVariants(inputs:inputs,owner:authoring)
        let owner=try WorkflowGroupOwnership.initialize(directory:inputs.directory,groupID:metadataGroupID ?? inputs.directory.lastPathComponent)
        return (requests,inputs.directory,owner)
    }
    @MainActor func testMatchingDirectoryWithWrongMetadataGroupRejectsBeforePublication()throws {
        let(requests,directory,owner)=try fixture(metadataGroupID:UUID().uuidString)
        defer {owner.close()}
        XCTAssertNotEqual(owner.metadata.groupID,directory.lastPathComponent)
        XCTAssertThrowsError(try WorkflowExperimentGroupStore.create(requests:requests,directory:directory,groupOwner:owner))
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("authoring.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("group.json").path))
    }
    @MainActor func testReopenedStoreIsReadOnlyAndPreservesLedger()throws {
        let(requests,directory,owner)=try fixture();defer {owner.close()}
        _=try WorkflowExperimentGroupStore.create(requests:requests,directory:directory,groupOwner:owner)
        let reader=try WorkflowExperimentGroupStore.open(directory:directory)
        let path=directory.appendingPathComponent("group.json"),bytes=try Data(contentsOf:path)
        XCTAssertThrowsError(try reader.begin(ordinal:1,runID:UUID(),directoryID:UUID()))
        XCTAssertThrowsError(try reader.cancelPending())
        XCTAssertThrowsError(try reader.recoverInterrupted())
        XCTAssertEqual(try Data(contentsOf:path),bytes)
        XCTAssertEqual(reader.record.revision,0)
    }
    @MainActor func testWrongAndClosedOwnerCannotPublish()throws {
        let(requests,directory,owner)=try fixture();defer {owner.close()}
        let(_,_,foreign)=try fixture();defer {foreign.close()}
        XCTAssertThrowsError(try WorkflowExperimentGroupStore.create(requests:requests,directory:directory,groupOwner:foreign))
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("authoring.json").path))
        owner.close()
        XCTAssertThrowsError(try WorkflowExperimentGroupStore.create(requests:requests,directory:directory,groupOwner:owner))
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("group.json").path))
    }
    @MainActor func testClosedOwnerAndChangedLedgerRejectMutationWithoutPublishingState()throws {
        let(requests,directory,owner)=try fixture();defer {owner.close()}
        let store=try WorkflowExperimentGroupStore.create(requests:requests,directory:directory,groupOwner:owner)
        let path=directory.appendingPathComponent("group.json"),bytes=try Data(contentsOf:path),record=store.record
        let changed=bytes+Data(" ".utf8);try changed.write(to:path)
        XCTAssertThrowsError(try store.cancelPending())
        XCTAssertEqual(try Data(contentsOf:path),changed);XCTAssertEqual(store.record,record)
        try bytes.write(to:path);owner.close()
        XCTAssertThrowsError(try store.begin(ordinal:1,runID:UUID(),directoryID:UUID()))
        XCTAssertThrowsError(try store.cancelPending())
        XCTAssertEqual(try Data(contentsOf:path),bytes);XCTAssertEqual(store.record,record)
    }
    @MainActor func testBackgroundCreationRejectsClosedOwnerBeforePublishing()async throws {
        let(requests,directory,owner)=try fixture();owner.close()
        do {
            _=try await WorkflowExperimentGroupStore.createInBackground(requests:requests,directory:directory,groupOwner:owner)
            XCTFail("Closed owner cannot publish background validation results")
        } catch {}
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("group.json").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("authoring.json").path))
    }
}
