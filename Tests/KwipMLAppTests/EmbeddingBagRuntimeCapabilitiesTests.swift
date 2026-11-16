import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class EmbeddingBagRuntimeCapabilitiesTests: XCTestCase {
    // Literal reviewed contracts independent of generated authored metadata and Python producers.
    let descriptors: [String: Data] = [
        "bag_input": Data(#"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.bag_input","graphSchemaVersions":[2],"packetSchemaVersion":1,"layouts":["fixed","packed"],"coordinateDTypes":{"indices":"int64","offsets":"int64","per_sample_weights":"float32"},"maximumItemsPerBag":128,"maximumBags":4096,"maximumItems":1000000,"maximumOutputElements":2000000,"integerDerivatives":"unavailable","sourceWeights":"realPairedInput","batching":"wholeBags","emptyPackedBags":"preserve"}"#.utf8),
        "embedding_bag": Data(#"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.embedding_bag","graphSchemaVersions":[2],"packetSchemaVersion":1,"layouts":["fixed","packed"],"coordinateDTypes":{"indices":"int64","offsets":"int64","per_sample_weights":"float32"},"maximumItemsPerBag":128,"maximumBags":4096,"maximumItems":1000000,"maximumOutputElements":2000000,"integerDerivatives":"unavailable","sourceWeights":"realPairedInput","batching":"wholeBags","emptyPackedBags":"preserve","modes":["mean","sum","max"],"weightedModes":["sum"],"outputDType":"float32","maximumParameters":2000000,"maximumStaticTableElements":100000000,"paddingGradient":"suppressed","frozenWeightsMayRenormalize":true,"trainableSparseAdam":false,"mpsMaxNormExecution":"reject","mpsTrainableFrequencyScaling":"reject","sourceWeightsDerivatives":"sumOnly"}"#.utf8)
    ]
    func testExactIndependentPairDescriptorsAreAdmitted() throws {
        for (id, data) in descriptors { XCTAssertNoThrow(try EmbeddingBagRuntimeCapabilities.verify(data, componentID: id)) }
        let actual = EmbeddingBagRuntimeCapabilities(descriptors: descriptors)
        XCTAssertEqual(actual.componentIDs, Set(["bag_input", "embedding_bag"]))
        XCTAssertTrue(actual.isComplete)
    }
    func testPairRequiresBothAndNeverBorrowsOrdinaryEmbedding() throws {
        XCTAssertFalse(EmbeddingBagRuntimeCapabilities().isComplete)
        for id in ["bag_input", "embedding_bag"] {
            let partial = EmbeddingBagRuntimeCapabilities(descriptors: [id: try XCTUnwrap(descriptors[id])])
            XCTAssertTrue(partial.contains(id)); XCTAssertFalse(partial.isComplete)
        }
        XCTAssertFalse(EmbeddingBagRuntimeCapabilities(descriptors: ["embedding": try XCTUnwrap(descriptors["embedding_bag"])]).isComplete)
        XCTAssertThrowsError(try EmbeddingBagRuntimeCapabilities.verify(try XCTUnwrap(descriptors["embedding_bag"]), componentID: "embedding"))
    }
    func testEveryChangedFieldUnknownFieldWrongTypeVersionAndOversizeRefuses() throws {
        for (id, data) in descriptors {
            let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            for key in object.keys {
                var missing = object; missing.removeValue(forKey: key)
                XCTAssertThrowsError(try EmbeddingBagRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject: missing), componentID: id), key)
                var wrong = object; wrong[key] = NSNull()
                XCTAssertThrowsError(try EmbeddingBagRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject: wrong), componentID: id), key)
            }
            var unknown = object; unknown["cpuFallback"] = true
            XCTAssertThrowsError(try EmbeddingBagRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject: unknown), componentID: id))
            for version: Any in [["major": true, "minor": 0, "patch": 0], ["major": 2, "minor": 0, "patch": 0], "1.0.0"] {
                var changed = object; changed["version"] = version
                XCTAssertThrowsError(try EmbeddingBagRuntimeCapabilities.verify(JSONSerialization.data(withJSONObject: changed), componentID: id))
            }
            var tooLarge = data; tooLarge.append(Data(repeating: 32, count: 8193))
            XCTAssertThrowsError(try EmbeddingBagRuntimeCapabilities.verify(tooLarge, componentID: id))
        }
    }
    func testWorkspaceDefaultsNeverGrantBagPair() throws {
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let workspace = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
        XCTAssertFalse(workspace.embeddingBagRuntimeCapabilities.isComplete)
        XCTAssertFalse(workspace.embeddingBagRuntimeCapabilities.contains("bag_input"))
        XCTAssertFalse(workspace.embeddingBagRuntimeCapabilities.contains("embedding_bag"))
    }
    func testDiscoveryInstallsBothRevokesMissingAndFailedContract() throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let contract=try BuiltInCapabilityBridge.executionContract(registry:workspace.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu])
        let bytes=try JSONEncoder().encode(contract)
        func install(_ bags:[String:Data],execution:Data?=nil)throws {
            try workspace.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:execution ?? bytes,bagDescriptors:bags)
        }
        try install([:]);let before=workspace.verifiedRuntimeComponentCount
        try install(descriptors)
        XCTAssertTrue(workspace.embeddingBagRuntimeCapabilities.isComplete)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount,before+2)
        try install(["bag_input":try XCTUnwrap(descriptors["bag_input"])])
        XCTAssertTrue(workspace.embeddingBagRuntimeCapabilities.contains("bag_input"))
        XCTAssertFalse(workspace.embeddingBagRuntimeCapabilities.isComplete)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount,before,"A half pair never grants a usable bag component")
        try install(descriptors)
        XCTAssertThrowsError(try install(descriptors,execution:Data("{}".utf8)))
        XCTAssertFalse(workspace.embeddingBagRuntimeCapabilities.isComplete)
        try install(descriptors);try install([:])
        XCTAssertFalse(workspace.embeddingBagRuntimeCapabilities.isComplete)
        XCTAssertEqual(workspace.verifiedRuntimeComponentCount,before)
    }
    func testWorkflowPalettePredicateProtectsChooserAndDirectPlacement() throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let registry=try ComponentRegistry.builtIn()
        for id in ["bag_input","embedding_bag"] {XCTAssertNotNil(registry.component(type:id),"Run this gate against the genuine published pair")}
        let library=ProjectLibrary(directory:root.appendingPathComponent("projects")),document=try WorkflowExamples.preparedParallelClassifier(device:"cpu")
        _=try library.saveSource(.workflow(document),id:document.projectID)
        let owner=try WorkflowAuthoringStore(library:library,projectID:document.projectID,modelRegistry:registry,
            dataRegistry:.load(from:repo.appendingPathComponent("runner/dataflow-operators.json")))
        let canvas=try WorkflowCanvasAdapter(owner:owner,modelRegistry:registry,dataRegistry:owner.dataRegistry)
        let entries=canvas.creationEntries.filter {EmbeddingBagRuntimeCapabilities.operations.contains($0.iconType)}
        XCTAssertEqual(entries.count,2)
        canvas.componentAvailability={ !EmbeddingBagRuntimeCapabilities.operations.contains($0) }
        let original=owner.document
        for entry in entries {
            XCTAssertFalse(canvas.canAddType(entry.id))
            XCTAssertFalse(canvas.creationEntries.contains {$0.id == entry.id})
            XCTAssertNil(canvas.canvasCatalog.component(type:entry.id))
            canvas.addNode(type:entry.id,x:20,y:20)
            XCTAssertEqual(owner.document,original,"A drop cannot bypass unavailable bag palette state")
        }
        XCTAssertTrue(canvas.creationEntries.contains {$0.iconType == LinearPalette.type})
        canvas.componentAvailability={_ in true}
        for entry in entries {XCTAssertTrue(canvas.canAddType(entry.id));XCTAssertTrue(canvas.creationEntries.contains {$0.id == entry.id})}
    }
    func testMissingPairRefusesActualSelectedWorkflowBeforeSourceCapture() throws {
        let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:root)}
        let workspace=WorkspaceStore(rootOverride:repo,storageOverride:root,detectCapabilities:false)
        let document=try BagWorkflowArchiveTests().fixture(packed:true,weighted:true)
        _=try workspace.projectLibrary.saveSource(.workflow(document),id:document.projectID)
        XCTAssertTrue(workspace.beginWorkflowProject(id:document.projectID),workspace.statusMessage)
        let host=try XCTUnwrap(workspace.workflowSession)
        defer {host.cancel();workspace.retireWorkflowSession()}
        let execution=try BuiltInCapabilityBridge.executionContract(registry:workspace.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu])
        try workspace.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:JSONEncoder().encode(execution))
        host.captureRunSources={_,_,_,_ in XCTFail("Unavailable bag support must refuse before source capture");throw CancellationError()}
        host.executionLaunch={_,_,_,_,_ in XCTFail("Unavailable bag support must never launch");throw CancellationError()}
        XCTAssertFalse(host.run(train:WorkflowExamples.trainAddress))
        XCTAssertTrue(host.statusMessage.localizedCaseInsensitiveContains("bag"),host.statusMessage)
        XCTAssertTrue(host.statusMessage.localizedCaseInsensitiveContains("runtime"),host.statusMessage)
        XCTAssertNil(host.queuedRequestID)
        XCTAssertFalse(workspace.isBusy)
    }

}
