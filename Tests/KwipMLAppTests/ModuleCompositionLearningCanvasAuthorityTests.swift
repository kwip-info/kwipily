import XCTest
import Combine
import KwipMLCore
@testable import KwipMLApp

@MainActor final class ModuleCompositionLearningCanvasAuthorityTests: XCTestCase {
    private func fixture(_ task: ModuleCompositionLearningTask, prepared: Bool = true) throws -> (URL, TensorLearningDraftSession, ComponentRegistry) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let drafts = TensorLearningDraftSession(repository: .init(directory: root))
        let project = UUID().uuidString
        XCTAssertTrue(drafts.create(projectID: project, operation: task.rawValue, mode: .guided), drafts.errorMessage ?? "")
        let model = try ModuleCompositionLearningDraft.model(projectID: project, task: task, prepared: prepared)
        XCTAssertTrue(drafts.update { $0.compositionModel = model }, drafts.errorMessage ?? "")
        XCTAssertTrue(drafts.save())
        return (root,drafts,try .builtIn())
    }
    private func source(_ owner: ModuleCompositionLearningCanvasAuthority) throws -> CompositeProjectDocument {
        guard case .blockModel(let model) = owner.canvasDocument else { throw LearningContractError("Expected authored lesson model") }
        return model.source
    }
    func testRootMovementPersistsOnlyExistingDraftAndUndoRedoRestoresAppliedRoot() throws {
        let (root,drafts,registry) = try fixture(.buildReuse); defer {try? FileManager.default.removeItem(at:root)}
        let owner = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:{true},isBusy:{false})
        let before = try source(owner),node = try XCTUnwrap(before.root.nodes.first)
        XCTAssertNotNil(owner.applyRootCommand(.move(nodeIDs:[node.id],offset:.init(x:25,y:10)),selection:.init(),lease:owner.lease()))
        XCTAssertNotEqual(try source(owner).root.nodes.first?.x,node.x)
        XCTAssertTrue(owner.scopeCanUndo(path:[]))
        XCTAssertTrue(owner.scopeUndo(lease:owner.lease()))
        XCTAssertEqual(try source(owner).root,before.root)
        XCTAssertTrue(owner.scopeRedo(lease:owner.lease()))
        let reopened = TensorLearningDraftSession(repository:.init(directory:root))
        XCTAssertTrue(reopened.open(projectID:owner.projectID))
        XCTAssertEqual(reopened.draft?.compositionModel.map(ModelProjectDocument.blockModel),owner.canvasDocument)
    }
    func testNestedPolicyOrderKeysApplyAndDiscardKeepPrivateCopiesAndUndo() throws {
        let (root,drafts,registry) = try fixture(.orderName); defer {try? FileManager.default.removeItem(at:root)}
        let owner = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:{true},isBusy:{false})
        let initial = try source(owner),instance = try XCTUnwrap(initial.root.instances.first)
        XCTAssertTrue(owner.beginDraft(path:[instance.id],lease:owner.lease()))
        guard case .blockModel(let opened) = owner.canvasDocument, var proposed = opened.drafts.first?.proposed else { return XCTFail("Expected borrowed Block draft") }
        proposed.title = "My sequence"
        XCTAssertTrue(owner.updateDraft(proposed,lease:owner.lease(path:[instance.id])))
        XCTAssertTrue(owner.applyDraft(lease:owner.lease(path:[instance.id])))
        XCTAssertNotEqual(try source(owner).root.instances.first?.definition,instance.definition)
        XCTAssertTrue(owner.scopeUndo(lease:owner.lease()))
        XCTAssertEqual(try source(owner),initial)
        XCTAssertTrue(owner.beginDraft(path:[instance.id],lease:owner.lease()))
        XCTAssertTrue(owner.discardDraft(lease:owner.lease(path:[instance.id])))
        guard case .blockModel(let discarded) = owner.canvasDocument else {return XCTFail()}
        XCTAssertTrue(discarded.drafts.isEmpty)
    }
    func testNamedKeysOrderAndBoundariesPersistThroughExistingPrivateDraft() throws {
        let (root,drafts,registry) = try fixture(.orderName); defer {try? FileManager.default.removeItem(at:root)}
        let owner = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:{true},isBusy:{false})
        let instance = try XCTUnwrap(try source(owner).root.instances.last)
        XCTAssertTrue(owner.beginDraft(path:[instance.id],lease:owner.lease()))
        guard case .blockModel(let opened) = owner.canvasDocument, var proposed = opened.drafts.first?.proposed,
              let originalPolicy = proposed.composition else {return XCTFail("Expected named container draft")}
        XCTAssertEqual(originalPolicy.kind,.namedMap)
        let policy = CompositionPolicy(kind:.namedMap,children:originalPolicy.children.reversed().enumerated().map { .init(localID:$0.element.localID,key:$0.offset == 0 ? "firstStep":"lastStep") })
        proposed.composition = policy
        XCTAssertTrue(owner.updateDraft(proposed,lease:owner.lease(path:[instance.id])))
        XCTAssertTrue(owner.applyDraft(lease:owner.lease(path:[instance.id])))
        let applied = try source(owner),pin=try XCTUnwrap(applied.root.instances.last?.definition)
        let definition=try XCTUnwrap(applied.definitions.first{$0.reference == pin})
        XCTAssertEqual(definition.composition,policy)
        XCTAssertEqual(definition.inputs,proposed.inputs);XCTAssertEqual(definition.outputs,proposed.outputs)
    }
    func testPartialStarterCanApplyProtectedCopyBeforeWholeGraphIsComplete() throws {
        let (root,drafts,registry) = try fixture(.buildReuse,prepared:false); defer {try? FileManager.default.removeItem(at:root)}
        let owner = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:{true},isBusy:{false})
        let instance = try XCTUnwrap(try source(owner).root.instances.first)
        XCTAssertTrue(owner.beginDraft(path:[instance.id],lease:owner.lease()))
        XCTAssertTrue(owner.applyDraft(lease:owner.lease(path:[instance.id])),owner.error ?? "")
        XCTAssertTrue(try source(owner).root.edges.isEmpty)
    }
    func testFixedMembershipAndConfigurationRefuseThroughRootAndChildCommands() throws {
        let (root,drafts,registry) = try fixture(.orderName); defer {try? FileManager.default.removeItem(at:root)}
        let owner = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:{true},isBusy:{false})
        let before = owner.canvasDocument
        XCTAssertFalse(owner.editRootScope({$0.nodes.removeAll()},lease:owner.lease()))
        XCTAssertEqual(owner.canvasDocument,before)
        let instance = try XCTUnwrap(try source(owner).root.instances.first)
        XCTAssertTrue(owner.beginDraft(path:[instance.id],lease:owner.lease()))
        let opened=owner.canvasDocument
        XCTAssertFalse(owner.editDraftScope({$0.nodes[0].config["out_features"] = 9},lease:owner.lease(path:[instance.id])))
        XCTAssertEqual(owner.canvasDocument,opened)
        XCTAssertNil(owner.applyDraftCommand(.delete(.init(nodeIDs:["a"])),selection:.init(),lease:owner.lease(path:[instance.id])))
        XCTAssertEqual(owner.canvasDocument,opened)
    }
    func testBusyStaleLeaseRetiredAndChangedSessionRefuseWithoutLosingDraft() throws {
        let (root,drafts,registry) = try fixture(.buildReuse); defer {try? FileManager.default.removeItem(at:root)}
        var busy=false,current=true
        let owner = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:{current},isBusy:{busy})
        let stale = owner.lease(),before=owner.canvasDocument
        busy=true
        XCTAssertFalse(owner.editRootScope({$0.nodes[0].name="Busy edit"},lease:stale));XCTAssertEqual(owner.canvasDocument,before)
        busy=false
        XCTAssertTrue(owner.editRootScope({$0.nodes[0].name="My input"},lease:owner.lease()))
        XCTAssertFalse(owner.editRootScope({$0.nodes[0].name="Stale edit"},lease:stale))
        current=false;XCTAssertFalse(owner.editRootScope({$0.nodes[0].name="Closed"},lease:owner.lease()))
        current=true;owner.invalidate();XCTAssertFalse(owner.editRootScope({$0.nodes[0].name="Retired"},lease:owner.lease()))
        let second = try ModuleCompositionLearningCanvasAuthority(session:drafts,registry:registry,isCurrent:{true},isBusy:{false})
        XCTAssertTrue(drafts.open(projectID:owner.projectID))
        XCTAssertFalse(second.editRootScope({$0.nodes[0].name="Reopened"},lease:second.lease()))
    }
}
