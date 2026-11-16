import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class TypedDecisionIntegrationTests:XCTestCase {
 private func wait(_ predicate:()->Bool)async throws {
  let deadline=Date().addingTimeInterval(180)
  while Date()<deadline {if predicate(){return};try await Task.sleep(for:.milliseconds(30))}
  throw WorkflowArchiveError("Timed out waiting for typed-decision workflow")
 }
 func testRealOwnedWorkerTrainDecideSaveReloadWithCaptureOff()async throws {
  guard let development=ProcessInfo.processInfo.environment["KWIPML_TEST_DEVELOPMENT_ROOT"],let fixture=ProcessInfo.processInfo.environment["KWIPML_TEST_LAYA_FOLDER"] else {throw XCTSkip("Needs explicit local Laya fixture and isolated runtime")}
  let root=try canonicalBackendTemporaryDirectory()
  print("Typed-decision disposable evidence: "+root.path)
  let workspace=WorkspaceStore(rootOverride:URL(fileURLWithPath:development),storageOverride:root,detectCapabilities:false)
  await verifyRuntimeForIntegration(workspace)
  XCTAssertTrue(workspace.loadComponentWorkflowExample(TypedDecisionPracticeExample.exampleID,device:"cpu"),workspace.statusMessage)
  let host=try XCTUnwrap(workspace.workflowSession)
  defer {host.cancel()}
  try host.owner.setRunHistoryPolicy(.none)
  let runtime=try XCTUnwrap(workspace.runtime),policy=try workspace.executionPolicy(for:host.owner.projectID)
  try await host.modelLoader.importFolder(URL(fileURLWithPath:fixture),address:TypedDecisionPracticeExample.model,runtime:runtime,policy:policy)
  XCTAssertEqual(host.modelLoader.reference(TypedDecisionPracticeExample.model)?.metadata["task"],.string("typedDecisions"))
  XCTAssertTrue(host.owner.select(.init(nodeAddresses:[TypedDecisionPracticeExample.train]),lease:host.owner.lease()))
  let trainContext=try WorkflowControlNodeContext(owner:host.owner,address:TypedDecisionPracticeExample.train)
  let settingsSnapshot=try workspace.projectSettingsSnapshot(projectID:host.owner.projectID)
  let settingsProposal=try workspace.previewProjectSettings(.derivativeCapture(.init(values:true)),baseline:settingsSnapshot)
  XCTAssertTrue(host.inspectFineTuning(TypedDecisionPracticeExample.train,draft:trainContext.configuration),host.statusMessage)
  // Observe the real preparation-to-model handoff before its deferred task starts.
  let preparation=try XCTUnwrap(host.currentDataPreparation)
  let handoff=preparation.onTerminal
  var observedHandoff=false
  preparation.onTerminal={ success in
   observedHandoff=true
   XCTAssertTrue(success)
   XCTAssertTrue(host.isInspectingFineTuning(TypedDecisionPracticeExample.train))
   XCTAssertTrue(host.hasActiveOperation,"Inspection must remain active between table preparation and model inspection")
   do {_ = try workspace.applyProjectSettings(settingsProposal);XCTFail("Settings changed during an active operation")} catch {XCTAssertEqual((error as? ProjectSettingsFailure)?.code,"settings_busy")}
   XCTAssertEqual(try? workspace.projectExecutionPreferenceStore.load(projectID:host.owner.projectID),settingsSnapshot.overrides)
   handoff?(success)
  }
  try await wait {!host.hasActiveOperation && !workspace.isBusy && !host.modelLoader.isBusy}
  XCTAssertTrue(observedHandoff)
  XCTAssertEqual(host.modelLoader.fineTuningPreview?["kind"],.string("typedDecisionPreparation"),host.statusMessage)
  XCTAssertTrue(host.modelLoader.trainingRecords.isEmpty,"Mapping inspection must not train or require a Save destination")
  let save=WorkflowNodeAddress(domain:.control,localID:"save-model"),destination=root.appendingPathComponent("saved-decisions")
  XCTAssertTrue(host.modelDestinations.set(destination:destination,for:save,lease:host.owner.lease()))
  XCTAssertTrue(host.owner.select(.init(nodeAddresses:[TypedDecisionPracticeExample.train]),lease:host.owner.lease()))
  XCTAssertTrue(host.run(),host.statusMessage)
  try await wait {!host.hasActiveOperation && !workspace.isBusy && !host.modelLoader.isBusy}
  XCTAssertEqual(host.modelLoader.trainingRecords.count,1,host.statusMessage)
  XCTAssertNil(host.modelLoader.failure(TypedDecisionPracticeExample.train),host.statusMessage)
  XCTAssertEqual(host.currentDataPreparation?.phase,.completed,host.statusMessage)
  XCTAssertNotNil(host.currentDataPreparation?.preparationOutputs["decide"])
  XCTAssertTrue(FileManager.default.fileExists(atPath:destination.appendingPathComponent("kwip-decisions.json").path),host.statusMessage)
  let trained=try XCTUnwrap(host.modelLoader.reference(TypedDecisionPracticeExample.train))
  XCTAssertEqual(trained.metadata["task"],.string("typedDecisions"))
  XCTAssertNotNil(trained.metadata["questionSet"])
  let evaluate=WorkflowNodeAddress(domain:.control,localID:"evaluate-decisions"),partition=WorkflowNodeAddress(domain:.data,localID:"partition")
  XCTAssertTrue(host.canvas.apply(.insert(.init(address:evaluate,payload:.evaluateTest(config:["device":.string("cpu"),"batchSize":1]))),lease:host.owner.lease()))
  XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:TypedDecisionPracticeExample.train,port:"model"),to:.init(address:evaluate,port:"model")),lease:host.owner.lease()))
  XCTAssertTrue(host.canvas.apply(.connect(from:.init(address:partition,port:"test"),to:.init(address:evaluate,port:"test")),lease:host.owner.lease()))
  XCTAssertTrue(host.evaluate(address:evaluate),host.statusMessage)
  try await wait {!host.hasActiveOperation && !workspace.isBusy && !host.modelLoader.isBusy}
  XCTAssertEqual(host.modelLoader.evaluationSummary?["kind"],.string("typedDecisionEvaluation"),host.statusMessage)
  XCTAssertEqual(host.modelLoader.evaluationRecords.count,1,host.statusMessage)
  let count=host.modelLoader.trainingRecords.count
  XCTAssertTrue(host.owner.select(.init(nodeAddresses:[TypedDecisionPracticeExample.decision]),lease:host.owner.lease()))
  XCTAssertTrue(host.run(),host.statusMessage)
  try await wait {!host.hasActiveOperation && !workspace.isBusy}
  XCTAssertEqual(host.modelLoader.trainingRecords.count,count,"Viewing decisions must not retrain")
  XCTAssertEqual(host.currentDataPreparation?.phase,.completed,host.statusMessage)
  let record=try XCTUnwrap(host.modelLoader.trainingRecords.first)
  XCTAssertEqual(try record.verify(root:root)["task"],.string("typedDecisions"),"Cold evidence must verify without the expired worker source snapshot")
  let exported=try await ModelAssetOperation().run(action:"inspect",settings:[:],source:destination,files:ModelAssetFiles.inventory(destination),projectID:host.owner.projectID,root:root,accountingRoot:root,runtime:runtime,policy:policy,isCurrent:{true},onProgress:{_ in})
  let metadata=try JSONDecoder().decode([String:JSONValue].self,from:Data(contentsOf:exported.directory.appendingPathComponent("output/model-metadata.json")))
  XCTAssertEqual(metadata["identity"],trained.metadata["identity"])
  XCTAssertEqual(metadata["questionSet"],trained.metadata["questionSet"])
 }
}
