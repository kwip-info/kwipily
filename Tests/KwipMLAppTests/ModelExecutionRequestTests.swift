import XCTest
import KwipMLCore
@testable import KwipMLApp

final class ModelExecutionRequestTests:XCTestCase {
    @MainActor private func setup()throws->(URL,ProjectLibrary,ModelAuthoringStore) {
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects"))
        let record=try library.save(ExampleFlows.parallelClassifier)
        return (root,library,try ModelAuthoringStore(library:library,projectID:record.id))
    }
    @MainActor func testFlatSignatureExactlyMatchesLegacyWorkspaceAlgorithmAndDefaultSettings()throws {
        let (root,_,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        let request=try ModelExecutionRequest.capture(from:owner,registry:registry)
        let workspace=WorkspaceStore(rootOverride:root.appendingPathComponent("workspace"),detectCapabilities:false)
        XCTAssertEqual(request.executionSignature,workspace.semanticSignature(request.graph))
        // Independent frozen legacy encoding shape, not merely two callers of the new helper.
        struct LegacySignature:Encodable {let architecture:Data;let execution:ExecutionSettings?}
        let encoder=JSONEncoder();encoder.outputFormatting=[.sortedKeys]
        let expected=try encoder.encode(LegacySignature(architecture:GraphSemantics.canonicalData(request.graph,registry:registry),execution:request.graph.execution))
        XCTAssertEqual(request.executionSignature,expected)
        XCTAssertNil(request.graph.execution);XCTAssertEqual(request.settings,ExecutionSettings())
        XCTAssertEqual(request.projectID,owner.projectID);XCTAssertTrue(request.isCurrent(in:owner))
        XCTAssertEqual(request.executionSignature,try ModelExecutionIdentity.fullSignature(request.graph,registry:registry))
        owner.setExecutionBusy(true);XCTAssertTrue(request.isCurrent(in:owner));owner.setExecutionBusy(false)
    }
    @MainActor func testCosmeticNameChangesExportIdentityButNotArchitecture()throws {
        let (root,_,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        let before=try ModelExecutionRequest.capture(from:owner,registry:registry)
        XCTAssertTrue(owner.editRootScope({$0.nodes[1].name="Renamed feature map";$0.nodes[1].x+=17},lease:owner.lease()))
        let after=try ModelExecutionRequest.capture(from:owner,registry:registry)
        XCTAssertEqual(before.snapshot.semanticSHA256,after.snapshot.semanticSHA256)
        XCTAssertEqual(before.executionSignature,after.executionSignature)
        XCTAssertNotEqual(before.exportContentIdentity,after.exportContentIdentity)
        XCTAssertFalse(before.isCurrent(in:owner));XCTAssertEqual(before.graph.nodes[1].name,"Branch A")
    }
    @MainActor func testSettingsChangeFullSignatureAndAreFrozenAtCapture()throws {
        let (root,_,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        let before=try ModelExecutionRequest.capture(from:owner,registry:registry)
        XCTAssertTrue(owner.editMetadata({_,settings in settings=ExecutionSettings(epochs:7,seed:123)},lease:owner.lease()))
        let after=try ModelExecutionRequest.capture(from:owner,registry:registry)
        XCTAssertEqual(before.snapshot.semanticSHA256,after.snapshot.semanticSHA256)
        XCTAssertNotEqual(before.executionSignature,after.executionSignature)
        XCTAssertNotEqual(before.exportContentIdentity,after.exportContentIdentity)
        XCTAssertEqual(before.settings.seed,42);XCTAssertEqual(after.settings.seed,123)
        XCTAssertEqual(before.settings.epochs,100);XCTAssertEqual(after.settings.epochs,7)
    }
    @MainActor func testProjectAndSessionIdentityRejectStaleCallbacks()throws {
        let (root,library,owner)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        let request=try ModelExecutionRequest.capture(from:owner,registry:.builtIn())
        let sameProjectNewSession=try ModelAuthoringStore(library:library,projectID:owner.projectID)
        XCTAssertFalse(request.isCurrent(in:sameProjectNewSession))
        let record=try library.save(ExampleFlows.parallelClassifier)
        let otherProject=try ModelAuthoringStore(library:library,projectID:record.id)
        XCTAssertFalse(request.isCurrent(in:otherProject));XCTAssertFalse(request.isCurrent(in:nil))
        XCTAssertEqual(request.executionSignature,try ModelExecutionRequest.capture(from:otherProject,registry:.builtIn()).executionSignature)
    }
    @MainActor func testNestedGELUCaptureAndUnresolvedDraftGate()throws {
        let (root,library,_)=try setup();defer{try? FileManager.default.removeItem(at:root)}
        var source=CompositeExamples.parallelClassifier
        for i in source.definitions[0].body.nodes.indices where ["relu","tanh"].contains(source.definitions[0].body.nodes[i].type) {
            source.definitions[0].body.nodes[i].type="gelu";source.definitions[0].body.nodes[i].config=["approximate_tanh":false]
        }
        let id=UUID().uuidString
        _ = try library.saveDocument(.blockModel(ModelAuthoringDocument(projectID:id,revision:11,source:source)),id:id)
        let owner=try ModelAuthoringStore(library:library,projectID:id)
        let request=try ModelExecutionRequest.capture(from:owner,registry:.builtIn())
        XCTAssertEqual(request.snapshot.authoringRevision,11);XCTAssertNotNil(request.snapshot.provenance)
        let activations=request.graph.nodes.filter{$0.type=="gelu"};XCTAssertEqual(activations.count,2)
        for node in activations {XCTAssertEqual(node.config["approximate_tanh"],false);XCTAssertEqual(request.snapshot.origins[node.id]?.instancePath,["features"])}
        XCTAssertTrue(owner.beginDraft(path:["features"],lease:owner.lease()))
        XCTAssertThrowsError(try ModelExecutionRequest.capture(from:owner,registry:.builtIn()))
        XCTAssertFalse(request.isCurrent(in:owner))
    }
}
