import XCTest
import KwipMLCore
@testable import KwipMLApp

/// Historical fixture receipts are explicit synthetic compatibility evidence, not new worker runs.
/// Numerical optimizer/objective/gradient tests and current inline Train practices remain independent.
@MainActor final class RetiredTrainingLessonCompatibilityTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func graph(_ operation:String)throws->GraphDocument {
        if let task=OptimizerLearningTask(rawValue:operation) {
            return .init(schemaVersion:6,nodes:[.init(id:"recipe",type:task.componentID,config:task.configuration),.init(id:"train",type:"train")],edges:[.init(id:"optimizer",source:"recipe",sourcePort:"out",target:"train",targetPort:"optimizer")])
        }
        if let task=SchedulerLearningTask(rawValue:operation) {
            return .init(schemaVersion:7,nodes:[.init(id:"schedule",type:task.componentID,config:task.configuration),.init(id:"optimizer",type:"optimizer_sgd",config:SchedulerLearningTask.suppliedOptimizerValues),.init(id:"train",type:"train")],edges:[.init(id:"schedule",source:"schedule",sourcePort:"out",target:"train",targetPort:"scheduler"),.init(id:"optimizer",source:"optimizer",sourcePort:"out",target:"train",targetPort:"optimizer")])
        }
        if let task=LossLearningTask(rawValue:operation) {return LossLearningCanvas.preparedGraph(task:task)}
        if let task=GradientLearningTask(rawValue:operation) {return try GradientLearningExamples.preparedGraph(task:task)}
        if let task=InitializationLearningTask(rawValue:operation) {return try InitializationLearningExamples.preparedGraph(task:task)}
        if let task=DataLoadingLearningTask(rawValue:operation) {return try DataLoadingExamples.lessonGraph(task:task)}
        throw LearningContractError("Unexpected historical operation")
    }
    private func installSyntheticCapabilities(_ workspace:WorkspaceStore)throws {
        // Admission must remain blocked even when every retired lesson's capability is present.
        // This fixture is deliberately not labeled as real runtime discovery.
        try installFixtureExecutionContract(workspace,version:"2.14.1")
        workspace.optimizerRuntimeCapabilities = .init(descriptors:try Dictionary(uniqueKeysWithValues:OptimizerConfiguration.componentIDs.map {($0,try WorkflowArchiveCodec.encode(XCTUnwrap(BundledOptimizerComponents.metadata($0)).runtimeContract))}))
        workspace.schedulerRuntimeCapabilities = .init(descriptors:try Dictionary(uniqueKeysWithValues:SchedulerConfiguration.componentIDs.map {($0,try WorkflowArchiveCodec.encode(XCTUnwrap(BundledSchedulerComponents.metadata($0)).runtimeContract))}))
        workspace.lossRuntimeCapabilities = .init(descriptors:try Dictionary(uniqueKeysWithValues:LossConfiguration.componentIDs.map {($0,try WorkflowArchiveCodec.encode(XCTUnwrap(BundledLossComponents.entries[$0]).runtimeContract))}))
        workspace.initializationRuntimeCapabilities = .init(descriptor:try WorkflowArchiveCodec.encode(XCTUnwrap(BundledInitializationComponents.metadata("initialization")).runtimeContract))
        workspace.gradientControlRuntimeCapabilities = .init(gradientPolicy:try WorkflowArchiveCodec.encode(XCTUnwrap(BundledGradientControlComponents.metadata("gradient_policy")).runtimeContract),trainability:try WorkflowArchiveCodec.encode(XCTUnwrap(BundledGradientControlComponents.metadata("trainability")).runtimeContract))
        workspace.dataLoadingRuntimeCapabilities = .init(dataset:try WorkflowArchiveCodec.encode(XCTUnwrap(BundledDataLoadingComponents.entries["dataset_source"]).runtimeContract),loader:try WorkflowArchiveCodec.encode(XCTUnwrap(BundledDataLoadingComponents.entries["data_loader"]).runtimeContract))
    }
    private func files(_ root:URL)throws->[String:Data] {
        guard let enumerator=FileManager.default.enumerator(at:root,includingPropertiesForKeys:[.isRegularFileKey]) else {return [:]}
        var result:[String:Data]=[:]
        for case let file as URL in enumerator where try file.resourceValues(forKeys:[.isRegularFileKey]).isRegularFile == true {
            result[String(file.path.dropFirst(root.path.count))]=try Data(contentsOf:file)
        }
        return result
    }
    private func assertPrivateTablesHidden(_ lesson:TensorLearningStore,file:StaticString=#filePath,line:UInt=#line) {
        XCTAssertNil(lesson.formativeProjection,file:file,line:line)
        XCTAssertNil(lesson.optimizerProjection,file:file,line:line)
        XCTAssertNil(lesson.schedulerProjection,file:file,line:line)
        XCTAssertNil(lesson.lossProjection,file:file,line:line)
        XCTAssertNil(lesson.gradientProjection,file:file,line:line)
        XCTAssertNil(lesson.initializationProjection,file:file,line:line)
        XCTAssertNil(lesson.recipeProjection,file:file,line:line)
    }
    func testEveryHistoricalOperationAndModeRestoresProgressButCannotLaunchOrRewriteEvidence()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:storage)}
        let directory=storage.appendingPathComponent("learning/projects")
        let drafts=TensorLearningDraftRepository(directory:directory),receipts=TensorLearningReceiptRepository(directory:directory)
        var records:[(TensorLearningDraft,TensorLearningReceipt)]=[]
        let operations=RetiredTrainingLessonDestination.operations.sorted()
        XCTAssertEqual(operations.count,OptimizerLearningTask.allCases.count+SchedulerLearningTask.allCases.count+LossLearningTask.allCases.count+GradientLearningTask.allCases.count+InitializationLearningTask.allCases.count+DataLoadingLearningTask.allCases.count)
        for operation in operations {
            for mode in [LearningMode.guided,.practice,.assessment] {
                let project=UUID().uuidString
                var draft=TensorLearningDraft(projectID:project,operation:operation,mode:mode)
                draft.graph=try graph(operation);draft.notes="Owner's saved historical notes"
                let submission=try draft.submissionData(),plan=try TensorLearningTask.plan(operation:operation,runtimeVersion:"2.14.1",seed:41)
                let evidence=LearningEvidence(assessmentID:plan.id,assessmentVersion:plan.assessmentVersion,fixtureDigest:plan.fixtureDigest,runtime:plan.runtime,submissionDigest:WorkflowPhaseArtifacts.hash(submission),fixtures:plan.fixtures.map {.init(fixtureID:$0.id,tensors:$0.expected)},elapsedMilliseconds:1,maximumTensorElements:plan.budget.maximumTensorElements)
                let receipt=try TensorLearningReceipt(projectID:project,operation:operation,mode:mode,submissionData:submission,plan:plan,evidence:evidence)
                XCTAssertEqual(receipt.attempt.grade.status,.passed,operation)
                _=try drafts.save(draft,expectedRevision:nil);try receipts.save(receipt)
                // JSON numbers have one persisted canonical representation (e.g. 0.0 decodes as 0).
                // Compare cold state to the exact saved draft, then retain byte-for-byte refusal checks.
                records.append((try drafts.load(projectID:project).draft,receipt))
            }
        }
        // A pre-existing History file must survive every refused submission and mode change.
        let history=storage.appendingPathComponent("historical-run-record.json")
        let historyBytes=Data("{\"status\":\"completed\",\"owner\":\"retained historical evidence\"}".utf8)
        try historyBytes.write(to:history,options:.atomic)
        // A new controller opens only persisted historical state; no assessment is replayed.
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
        let lesson=TensorLearningStore(workspace:workspace)
        try installSyntheticCapabilities(workspace)
        XCTAssertNotNil(workspace.runtime)
        for record in records {
            let (draft,receipt)=record
            let project=draft.projectID,operation=draft.operation,mode=draft.mode
            let ownerGraph=GraphDocument(name:"Archived lesson · "+operation)
            XCTAssertTrue(workspace.beginProject(ownerGraph,id:project),workspace.statusMessage)
            XCTAssertTrue(lesson.isActive,operation);XCTAssertEqual(lesson.draft,draft,operation)
            XCTAssertEqual(lesson.grade?.status,.passed,operation);XCTAssertTrue(lesson.resultSaved,operation)
            XCTAssertEqual(try receipts.latest(projectID:project)?.attempt.id,receipt.attempt.id)
            let before=try files(storage)
            XCTAssertFalse(lesson.canSubmit,operation);XCTAssertFalse(lesson.canRefreshLessonRuntime,operation)
            XCTAssertFalse(workspace.canRunCurrentFlow,"Historical lessons cannot fall back to ordinary training: \(operation)")
            XCTAssertTrue(lesson.runtimeReadinessError?.contains("read-only") == true,operation)
            XCTAssertFalse(lesson.refreshLessonRuntime(),operation)
            lesson.submit()
            XCTAssertTrue(lesson.statusMessage.contains("read-only"),operation)
            XCTAssertFalse(workspace.isAssessingLearning,operation);XCTAssertFalse(workspace.isBusy,operation)
            XCTAssertEqual(lesson.draft,draft,operation);XCTAssertEqual(lesson.grade?.status,.passed,operation)
            XCTAssertEqual(try files(storage),before,"Refusal must preserve all drafts, attempts, latest pointers and existing History: \(operation)")
            XCTAssertEqual(try receipts.latest(projectID:project)?.attempt.id,receipt.attempt.id)
            if mode == .assessment {
                assertPrivateTablesHidden(lesson)
                XCTAssertTrue(lesson.setMode(.guided));assertPrivateTablesHidden(lesson)
                XCTAssertNil(lesson.grade,"Private-origin evidence cannot become a public grade by switching mode")
                XCTAssertEqual(try receipts.latest(projectID:project)?.attempt.id,receipt.attempt.id)
                XCTAssertTrue(lesson.save());assertPrivateTablesHidden(lesson)
                lesson.restoreEditor();assertPrivateTablesHidden(lesson)
                XCTAssertNil(lesson.grade,"Cold restore cannot relabel a private-origin attempt as public")
            }
            XCTAssertEqual(try Data(contentsOf:history),historyBytes)
        }
    }
}
