import XCTest
import KwipMLCore
@testable import KwipMLApp

final class WorkflowEvaluationRequestTests:XCTestCase {
    private let frozen=WorkflowNodeAddress(domain:.control,localID:"frozen")
    private let terminal=WorkflowNodeAddress(domain:.control,localID:"evaluate")
    @MainActor func fixture()throws->(URL,WorkflowAuthoringStore,WorkflowPrivateFrozenModelBindings,RuntimeLocation) {
        let helper=WorkflowExecutionRequestTests(),(root,original,runtime)=try helper.setup()
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        var document=try WorkflowExamples.preparedGroupSplitClassifier(projectID:original.projectID)
        document.schemaVersion=5
        document.nodes += [.init(address:frozen,payload:.frozenModel(config:[:])),.init(address:terminal,payload:.evaluateTest(config:[:]))]
        document.edges += [.init(from:.init(address:frozen,port:"model"),to:.init(address:terminal,port:"model"),kind:.frozenModel),.init(from:.init(address:.init(domain:.data,localID:"split"),port:"test"),to:.init(address:terminal,port:"test"),kind:.table)]
        let library=ProjectLibrary(directory:root.appendingPathComponent("evaluation-projects"))
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:original.modelRegistry,dataRegistry:original.dataRegistry)
        let bindings=try WorkflowPrivateFrozenModelBindings(owner:owner,directory:root)
        let hash=String(repeating:"a",count:64)
        let reference=WorkflowFrozenModelReference(projectID:owner.projectID,directoryID:UUID().uuidString,runID:UUID().uuidString,requestIdentity:hash,outcomeSHA256:hash,modelManifestSHA256:hash,preparationManifestSHA256:hash,checkpointSHA256:hash)
        XCTAssertTrue(bindings.setReference(reference,for:frozen,lease:owner.lease(),expectedRevision:0))
        return(root,owner,bindings,runtime)
    }
    /// Deliberately synthetic stage seam: these tests prove capture lifecycle,
    /// not archive authenticity (covered by real frozen snapshot integration).
    func staged(_ reference:WorkflowFrozenModelReference,_ document:WorkflowDocument,_ plan:WorkflowEvaluationPlan,_ root:URL)throws->WorkflowFrozenModelSnapshot {
        let directory=root.appendingPathComponent("staged-"+UUID().uuidString)
        try FileManager.default.createDirectory(at:directory,withIntermediateDirectories:false)
        struct Receipt:Encodable {let schemaVersion:Int;let reference:WorkflowFrozenModelReference;let testEndpoint:DataFlowEndpoint;let files:[WorkflowArchiveFile]}
        let bytes=try WorkflowArchiveCodec.encode(Receipt(schemaVersion:1,reference:reference,testEndpoint:plan.testEndpoint,files:[]))
        try bytes.write(to:directory.appendingPathComponent("snapshot.json"))
        return .init(reference:reference,directory:directory,files:[],receiptSHA256:WorkflowArchiveCodec.hash(bytes))
    }
    @MainActor func testStorageLimitRejectsUnaccountedCustomStageBeforeInvocation() async throws {
        let (root, owner, bindings, runtime) = try fixture()
        var invoked = false
        do {
            _ = try await WorkflowEvaluationRequest.capture(from: owner, bindings: bindings, evaluate: terminal,
                runtime: runtime, root: root, stage: { reference, document, plan, root in
                    invoked = true
                    return try self.staged(reference, document, plan, root)
                }, storageLimitBytes: Int64.max)
            XCTFail("A custom stage cannot bypass the managed snapshot scope")
        } catch { XCTAssertFalse(invoked) }
        XCTAssertEqual(try WorkflowStorageAdmission(workspace: root).snapshot().committedBytes, 0)
    }
    @MainActor func testBudgetExhaustedDuringInputCaptureRefusesAndRemovesItsStage() async throws {
        let (root, owner, bindings, runtime) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        let policy = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 1), projectMaximumRunSeconds: nil)
        do {
            _ = try await WorkflowEvaluationRequest.capture(from: owner, bindings: bindings,
                evaluate: .init(domain: .control, localID: "evaluate"), runtime: runtime, root: root,
                stage: { reference, document, plan, root in
                    let snapshot = try self.staged(reference, document, plan, root)
                    Thread.sleep(forTimeInterval: 1.05)
                    return snapshot
                }, executionBudget: policy)
            XCTFail("An exhausted capture must not become a runnable request")
        } catch {
            guard case ExecutionBudgetError.wallTimeLimitExceeded(let seconds) = error else { return XCTFail(error.localizedDescription) }
            XCTAssertEqual(seconds, 1)
        }
        XCTAssertFalse(owner.isBusy)
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: root.path).contains { $0.hasPrefix("staged-") })
    }
    @MainActor func testCapturePinsSelectionAndRechecksStagedBytesAndRuntime()async throws {
        let(root,owner,bindings,runtime)=try fixture()
        let request=try await WorkflowEvaluationRequest.capture(from:owner,bindings:bindings,evaluate:terminal,runtime:runtime,root:root,stage:staged)
        defer {try? FileManager.default.removeItem(at:request.snapshot.directory)}
        XCTAssertTrue(request.isCurrent(in:owner,bindings:bindings))
        XCTAssertEqual(request.reference,bindings.reference(for:frozen))
        XCTAssertEqual(request.bindingRevision,bindings.revision)
        XCTAssertEqual(request.plan.testEndpoint,.init(node:"split",port:"test"))
        XCTAssertTrue(WorkflowArchiveCodec.isHash(request.requestIdentity))
        try request.verifyCapturedInputs()
        let snapshot=request.snapshot.directory.appendingPathComponent("snapshot.json"),bytes=try Data(contentsOf:snapshot)
        try Data("{}".utf8).write(to:snapshot)
        XCTAssertThrowsError(try request.verifyCapturedInputs())
        try bytes.write(to:snapshot)
        let entry=runtime.entryURL
        var code=try Data(contentsOf:entry);code.append(Data("\n# runtime changed\n".utf8));try code.write(to:entry)
        XCTAssertThrowsError(try request.verifyCapturedInputs())
    }
    @MainActor func testSuccessfulStagingIsRemovedWhenEditorBindingOrCancellationChanges()async throws {
        for change in ["editor","binding","cancel","busy"] {
            let(root,owner,bindings,runtime)=try fixture()
            let reached=expectation(description:"Stage reached barrier"),release=DispatchSemaphore(value:0)
            defer {release.signal()}
            let task=Task {try await WorkflowEvaluationRequest.capture(from:owner,bindings:bindings,evaluate:terminal,runtime:runtime,root:root,stage:{reference,document,plan,root in
                let result=try self.staged(reference,document,plan,root)
                reached.fulfill()
                guard release.wait(timeout:.now()+10) == .success else {throw DataFlowContractError("Test stage barrier expired")}
                return result // Intentionally returns even after cancellation.
            })}
            await fulfillment(of:[reached],timeout:5)
            switch change {
            case "editor":XCTAssertTrue(owner.rename("Changed during staging",lease:owner.lease()))
            case "binding":XCTAssertTrue(bindings.setReference(nil,for:frozen,lease:owner.lease(),expectedRevision:bindings.revision))
            case "busy":owner.setExecutionBusy(true)
            default:task.cancel()
            }
            release.signal()
            do {_=try await task.value;XCTFail("Stale/cancelled stage was accepted")}
            catch {}
            owner.setExecutionBusy(false)
            XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath:root.path).contains {$0.hasPrefix("staged-")})
        }
    }
    @MainActor func testForeignBinderAndMissingRealArchiveNeverBecomeCapturedRequests()async throws {
        let(root,owner,bindings,runtime)=try fixture()
        let(_,other,foreign,_)=try fixture()
        XCTAssertFalse(owner === other)
        do {_=try await WorkflowEvaluationRequest.capture(from:owner,bindings:foreign,evaluate:terminal,runtime:runtime,root:root,stage:{_,_,_,_ in XCTFail("Foreign binder reached staging");throw DataFlowContractError("Unexpected stage")});XCTFail("Foreign binder accepted")}
        catch {}
        do {_=try await WorkflowEvaluationRequest.capture(from:owner,bindings:bindings,evaluate:terminal,runtime:runtime,root:root);XCTFail("Default production staging accepted missing archive")}
        catch {}
    }
}
