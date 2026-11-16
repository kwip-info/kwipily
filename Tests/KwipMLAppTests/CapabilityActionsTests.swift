import Foundation
import KwipMLCore
import XCTest

@testable import KwipMLApp

final class CapabilityActionsTests: XCTestCase {
  @MainActor func testEmbeddingDiscoveryDoesNotSurviveMissingInvalidOrVersionOnlyRefresh()throws {
    let root=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer {try? FileManager.default.removeItem(at:root)}
    let store=WorkspaceStore(rootOverride:root,detectCapabilities:false)
    let contract=try BuiltInCapabilityBridge.executionContract(registry:store.registry,runtimeVersion:"2.14.1",availableDevices:[.cpu])
    let descriptor=Data((#"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.embedding","graphSchemaVersions":[2],"inputDType":"int64","weightDType":"float32","maximumInputSampleRank":6,"maximumOutputRank":8,"maximumPreparedColumns":128,"maximumParameters":2000000,"preservesBatch":true,"outOfRangePolicy":"reject","paddingGradient":"suppressed","frequencyScaling":true,"frozenWeightsMayRenormalize":true,"trainingOptimizer":"Adam","trainableSparseGradients":false,"previewCacheSchemaForMaxNorm":2,"mpsMaxNormExecution":"cpuFallback"}"#).utf8)
    func install(_ value:Data?)throws {
      try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:JSONEncoder().encode(contract),embeddingDescriptor:value)
    }
    try install(descriptor);XCTAssertNotNil(store.embeddingRuntimeCapability)
    XCTAssertTrue(store.embeddingRuntimeCapability?.mpsMaximumNormExplanation.contains("CPU") == true)
    let count=store.verifiedRuntimeComponentCount
    try install(nil);XCTAssertNil(store.embeddingRuntimeCapability)
    XCTAssertEqual(store.verifiedRuntimeComponentCount,count - 2,"Missing runtime support must remove both integer input and embedding from the verified count")
    try install(descriptor)
    try install(Data("{}".utf8));XCTAssertNil(store.embeddingRuntimeCapability)
    try install(descriptor)
    try store.installRuntimeContract(version:"2.14.1",expectedVersion:nil)
    XCTAssertNil(store.embeddingRuntimeCapability)
    try install(descriptor)
    XCTAssertThrowsError(try store.installDiscoveredRuntimeContract(version:"2.14.1",expectedVersion:nil,devices:["cpu"],descriptor:Data("{}".utf8),embeddingDescriptor:descriptor))
    XCTAssertNil(store.embeddingRuntimeCapability)
  }
  @MainActor func testActualProbeInstallsEmbeddingExtensionAndMissingSupportBlocksIndices()async throws {
    let repo=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    guard FileManager.default.isExecutableFile(atPath:repo.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development worker unavailable")}
    let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer {try? FileManager.default.removeItem(at:storage)}
    let store=WorkspaceStore(rootOverride:repo,storageOverride:storage,detectCapabilities:false)
    let ready=await withCheckedContinuation {continuation in
      store.refreshCapabilities {continuation.resume(returning:$0)}
    }
    XCTAssertTrue(ready,store.capabilityStatus)
    XCTAssertNotNil(store.embeddingRuntimeCapability)
    var unsupported=store.graph
    unsupported.nodes[0].type="index_input"
    // Missing extension must give a clear capability boundary before execution.
    store.embeddingRuntimeCapability=nil
    XCTAssertTrue(store.capabilityBoundaryError(for:unsupported)?.contains("verified runtime support") == true)
  }
  @MainActor func testRuntimeVersionPinAloneDoesNotVerifyExecutionAdapter() throws {
    let store = WorkspaceStore(
      rootOverride: FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString), detectCapabilities: false)
    try store.installRuntimeContract(version: "2.14.1", expectedVersion: "2.14.1")
    XCTAssertNotNil(
      store.capabilityBoundaryError,
      "An exact framework version does not establish the implemented execution contract.")
  }

  @MainActor func testMismatchedDescriptorCannotLeavePartialReadiness() throws {
    let store = WorkspaceStore(
      rootOverride: FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString), detectCapabilities: false)
    try installFixtureExecutionContract(store, version: "2.14.1")
    XCTAssertNil(store.capabilityBoundaryError)
    let contract = try BuiltInCapabilityBridge.executionContract(
      registry: store.registry, runtimeVersion: "2.14.1", availableDevices: [.cpu])
    var object = try XCTUnwrap(
      JSONSerialization.jsonObject(with: JSONEncoder().encode(contract)) as? [String: Any])
    object["adapterID"] = "unknown.adapter"
    XCTAssertThrowsError(
      try store.installDiscoveredRuntimeContract(
        version: "2.14.1", expectedVersion: nil, devices: ["cpu"],
        descriptor: JSONSerialization.data(withJSONObject: object)))
    XCTAssertNil(store.runtimeContractVersion)
    XCTAssertNil(store.runtimeExecutionContract)
    XCTAssertNotNil(store.capabilityBoundaryError)
    try installFixtureExecutionContract(store, version: "2.14.1")
    XCTAssertEqual(store.availableDevices, ["cpu"])
    XCTAssertEqual(store.runtimeExecutionContract?.tensor.dtype, "float32")
    XCTAssertThrowsError(
      try store.installDiscoveredRuntimeContract(
        version: "2.14.1", expectedVersion: nil, devices: ["cpu", "mps"],
        descriptor: JSONEncoder().encode(contract), tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(), moduleCompositionDescriptor: fixtureReviewedModuleCompositionDescriptor()))
    XCTAssertNil(store.runtimeExecutionContract)
  }
  @MainActor func testUnknownRuntimeCannotCrossExecutionBoundary() {
    let store = WorkspaceStore(
      rootOverride: FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString), detectCapabilities: false)
    XCTAssertNotNil(store.capabilityBoundaryError)
  }
  @MainActor func testExactVersionBuildsRealSevenComponentContract() throws {
    let store = WorkspaceStore(
      rootOverride: FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString), detectCapabilities: false)
    try installFixtureExecutionContract(store, version: "2.14.1+fixture")
    XCTAssertEqual(store.runtimeContractVersion, "2.14.1+fixture")
    XCTAssertNil(store.capabilityBoundaryError)
    XCTAssertFalse(store.runtimeContractDeferredIssues.isEmpty)
    store.graph.nodes.firstIndex(where: { $0.type == "linear" }).map {
      store.graph.nodes[$0].config["in_features"] = 99
    }
    XCTAssertNotNil(store.capabilityBoundaryError)
  }
  @MainActor func testVersionOnlyWorkerCannotBecomeReadyEvenWithSuccessfulExit() async throws {
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    for exitCode: Int32 in [0, 1] {
      let store = WorkspaceStore(rootOverride: repository, storageOverride: root, detectCapabilities: false)
      _ = try XCTUnwrap(store.runtime)
      store.capabilityLaunch = { _, _, _, _, line, exit in
        line(#"{"event":"capabilities","devices":["cpu"],"defaultDevice":"cpu","torchVersion":"2.14.1"}"#)
        exit(exitCode, "")
        return .init(cancel: {})
      }
      let accepted = await withCheckedContinuation { continuation in
        store.refreshCapabilities { continuation.resume(returning: $0) }
      }
      XCTAssertFalse(accepted)
      if exitCode == 0 { XCTAssertTrue(store.capabilityStatus.contains("execution contract"), store.capabilityStatus) }
      XCTAssertNil(store.runtimeContractVersion)
      XCTAssertNotNil(store.capabilityBoundaryError)
    }
  }
  @MainActor func testMismatchAndBroadVersionCannotBecomeReady() {
    let store = WorkspaceStore(
      rootOverride: FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString), detectCapabilities: false)
    XCTAssertThrowsError(
      try store.installRuntimeContract(version: "2.14.1", expectedVersion: "2.14.0"))
    XCTAssertNil(store.runtimeContractVersion)
    for version in ["", "latest", "2.14", ">=2.0", "2.14.1\n"] {
      XCTAssertThrowsError(try store.installRuntimeContract(version: version, expectedVersion: nil))
    }
  }
}

@MainActor func verifyRuntimeForIntegration(_ store: WorkspaceStore) async {
  let success = await withCheckedContinuation { continuation in
    store.refreshCapabilities { continuation.resume(returning: $0) }
  }
  XCTAssertTrue(success, store.capabilityStatus)
  XCTAssertNotNil(store.runtimeContractVersion)
}

@MainActor func installFixtureExecutionContract(_ store: WorkspaceStore, version: String) throws {
  let contract = try BuiltInCapabilityBridge.executionContract(
    registry: store.registry, runtimeVersion: version, availableDevices: [.cpu])
  try fixtureInstallDiscoveredRuntimeContract(store,
    version: version, expectedVersion: nil, devices: ["cpu"],
    descriptor: JSONEncoder().encode(contract), tensorCompositionDescriptors: fixtureReviewedCompositionDescriptors(), moduleCompositionDescriptor: fixtureReviewedModuleCompositionDescriptor())
}

/// Explicit independent fixtures for the existing parallel flow. Production
/// discovery remains absent by default; these descriptors are never auto-granted.
func fixtureReviewedCompositionDescriptors() -> [String: Data] {
    [
        "concat": Data(#"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.concat","graphSchemaVersions":[1,2],"maximumRank":8,"preservesBatch":true,"graphDTypes":["float32","int64"],"maximumTensorElements":100000000,"inputCount":2,"axisMinimum":-8,"axisMaximum":7,"excludesBatch":true,"inputOrder":["a","b"]}"#.utf8),
        "add": Data(#"{"version":{"major":1,"minor":0,"patch":0},"adapterID":"kwip.runner.graph-v2.add","graphSchemaVersions":[1,2],"maximumRank":8,"preservesBatch":true,"graphDTypes":["float32","int64"],"maximumTensorElements":100000000,"inputCount":2,"alphaMinimum":-3.4028234663852886e+38,"alphaMaximum":3.4028234663852886e+38,"sampleBroadcastModes":[false,true],"broadcastBatch":false,"integerAlpha":1,"integerBroadcast":false}"#.utf8)
    ]
}

func fixtureReviewedModuleCompositionDescriptor() -> Data { Data(#"{"version":"1.0.0","adapterID":"kwip.runner.module-composition","graphSchemaVersions":[2],"planSchemaVersion":1,"modelAuthoringSchemaVersion":2,"compositeSourceSchemaVersion":3,"nativeExpansionContractVersion":2,"workflowSchemaVersion":11,"compilerContractVersion":10,"captureFormatVersion":6,"devices":["cpu","mps"],"modes":["branched-structure","nested-structure","parameter-ownership","train-eval","architecture-export","sequential-structure","ordered-structure","named-structure"],"sourceMembers":["torch.nn.Module","torch.nn.Sequential","torch.nn.ModuleList","torch.nn.ModuleDict"],"containerKinds":["graph","sequential","orderedList","namedMap"],"maximumDefinitions":64,"maximumExpandedNodes":128,"maximumEdges":256,"maximumContainers":129,"maximumChildReferences":256,"maximumBoundaryEndpoints":4096,"maximumDepth":8,"maximumPlanBytes":131072,"maximumSourceBytes":2000000,"maximumModelSourceBytes":2000000,"maximumParameters":200000,"parameterOwnership":"expandedNodeRelativeNativeParameter","stateOwnership":"bijectiveNativePersistentState","sharedParameters":false,"sequentialExecution":"nativeForwardValidatedChain","collectionExecution":"explicitDAG","sourceVerification":"independentCapturedRegistryReExpansion","exactResumeDevices":["cpu"],"fineTuneDevices":["cpu","mps"],"mpsResume":"weightsOnly","portableExport":"selfContainedNativeHierarchy","legacyAbsence":"unchanged"}"#.utf8) }
