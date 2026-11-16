import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor private final class ControlledAssessmentDriver:MeasuredAssessmentDriving {
    var runtimeVersions:[String:String]=[:]
    var error:String?
    var grade:LearningGrade?
    var evidence:LearningEvidence?
    var isRunning=false
    var completion:(()->Void)?
    var callbacks:[()->Void]=[]
    var submissions:[Data]=[]
    var plans:[LearningAssessmentPlan]=[]
    var cancellations=0
    var configured: [(ResolvedExecutionBudget, Int64?)] = []
    var accountingRoot:URL?
    func configureExecution(workspace:URL,projectID:String,budget:ResolvedExecutionBudget,storageLimitBytes:Int64?)throws {
        configured.append((budget, storageLimitBytes))
    }
    func discoverRuntime(_ runtime:RuntimeLocation) {isRunning=true;runtimeVersions=[:];grade=nil;evidence=nil;callbacks.append(completion!)}
    func assess(plan:LearningAssessmentPlan,submission:Data,runtime:RuntimeLocation) {isRunning=true;grade=nil;evidence=nil;plans.append(plan);submissions.append(submission);callbacks.append(completion!)}
    func cancel() {cancellations+=1}
    func finish(_ index:Int,versions:[String:String]=[:],status:LearningGradeStatus?=nil) {
        isRunning=false;runtimeVersions=versions;grade=status.map {LearningGrade(status:$0,summary:"Controlled outcome")};callbacks[index]()
    }
}

final class MeasuredLearningSessionTests:XCTestCase {
    @MainActor private func workspace(_ folder:URL)throws->WorkspaceStore {
        let repository=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace=WorkspaceStore(rootOverride:repository,storageOverride:folder,detectCapabilities:false)
        guard workspace.runtime != nil else {throw XCTSkip("Local runtime unavailable")}
        XCTAssertTrue(workspace.beginProject(GraphDocument(name:"Controlled lesson")))
        workspace.learningProjects.activateAffine(projectID:try XCTUnwrap(workspace.activeProjectID))
        return workspace
    }
    @MainActor func testDeniedExecutionHasActionableStartupFailureAndRetryClearsIt() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver(),session=MeasuredLearningSession(driver:driver)
        let id=try XCTUnwrap(workspace.activeProjectID)
        workspace.executionAdmission = {throw NSError(domain:"test.execution",code:1,userInfo:[NSLocalizedDescriptionKey:"Execution unavailable"])}
        var completions=0
        func start()->Bool {session.start(workspace:workspace,projectID:id,submission:Data(),runtimeVersionKey:"pytorch",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{_ in completions+=1})}
        XCTAssertFalse(start())
        XCTAssertEqual(completions,0)
        XCTAssertTrue(try XCTUnwrap(session.startFailureMessage).contains("Execution unavailable"))
        XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertTrue(driver.callbacks.isEmpty)
        workspace.executionAdmission = {}
        XCTAssertTrue(start())
        XCTAssertNil(session.startFailureMessage)
        session.cancel()
        driver.finish(0,status:.cancelled)
        XCTAssertEqual(completions,1)
        XCTAssertFalse(workspace.isAssessingLearning)
    }
    @MainActor func testSavedPolicyIsCapturedOnceAndExplicitBudgetStillUsesSavedStorage() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver(),session=MeasuredLearningSession(driver:driver)
        let id=try XCTUnwrap(workspace.activeProjectID)
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds:120,cpuThreads:3,managedStorageLimitBytes:8_000_000)))
        XCTAssertTrue(workspace.setProjectExecutionOverrides(try .init(maximumRunSeconds:90,cpuThreads:2,memory:.limit(536_870_912)),projectID:id))
        func start(_ budget:ResolvedExecutionBudget?=nil)->Bool {
            session.start(workspace:workspace,projectID:id,submission:Data(),runtimeVersionKey:"pytorch",executionBudget:budget,
                makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{_ in})
        }
        XCTAssertTrue(start())
        let captured=try XCTUnwrap(driver.configured.first)
        XCTAssertEqual(captured.0.requestedSeconds,90);XCTAssertEqual(captured.0.maximumRunSeconds,65)
        XCTAssertEqual(captured.0.resources?.cpuThreads,2);XCTAssertEqual(captured.0.resources?.maximumResidentBytes,536_870_912)
        XCTAssertEqual(captured.1,8_000_000)
        XCTAssertTrue(workspace.setExecutionPreferences(try .init(maximumRunSeconds:5,cpuThreads:8,managedStorageLimitBytes:1)))
        driver.finish(0,versions:["pytorch":"2.14.1"])
        XCTAssertEqual(driver.configured.count,1)
        session.cancel();driver.finish(1,status:.cancelled)
        let explicit=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:20),projectMaximumRunSeconds:nil)
        XCTAssertTrue(start(explicit))
        XCTAssertEqual(driver.configured.last?.0,explicit);XCTAssertEqual(driver.configured.last?.1,1)
        session.cancel();driver.finish(2,status:.cancelled)
    }

    @MainActor func testDiscoveryCancellationReleasesExactlyOnceAndOldCallbacksCannotFinishNewRun() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver(),session=MeasuredLearningSession(driver:driver)
        let id=try XCTUnwrap(workspace.activeProjectID)
        var results:[MeasuredLearningResult]=[]
        func start()->Bool {session.start(workspace:workspace,projectID:id,submission:Data("captured".utf8),runtimeVersionKey:"pytorch",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{results.append($0)})}
        XCTAssertTrue(start());XCTAssertFalse(start())
        workspace.cancelRun();XCTAssertEqual(driver.cancellations,1)
        driver.finish(0,versions:["pytorch":"2.14.1"])
        XCTAssertEqual(results.count,1);XCTAssertEqual(results.first?.grade.status,.cancelled)
        XCTAssertNil(results.first?.evidence);XCTAssertTrue(driver.plans.isEmpty)
        XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertTrue(start());driver.callbacks[0]()
        XCTAssertTrue(workspace.isAssessingLearning);XCTAssertEqual(results.count,1)
        session.cancel();driver.finish(1,status:.cancelled)
        XCTAssertFalse(workspace.isAssessingLearning);XCTAssertEqual(results.count,2)
    }
    @MainActor func testCapturedSubmissionFreshSeedAndPhaseGuard() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver()
        var seed:UInt64=40
        let session=MeasuredLearningSession(driver:driver,fixtureSeed:{seed+=1;return seed})
        let id=try XCTUnwrap(workspace.activeProjectID)
        var results:[MeasuredLearningResult]=[]
        var bytes=Data("original".utf8)
        XCTAssertTrue(session.start(workspace:workspace,projectID:id,submission:bytes,runtimeVersionKey:"pytorch",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{result in XCTAssertFalse(workspace.isAssessingLearning);results.append(result)}))
        bytes=Data("changed".utf8)
        driver.finish(0,versions:["pytorch":"2.14.1"])
        XCTAssertEqual(driver.submissions,[Data("original".utf8)]);XCTAssertEqual(driver.plans.first?.fixtureSeed,41)
        driver.callbacks[0]();XCTAssertEqual(driver.plans.count,1)
        driver.finish(1,status:.executionFailed)
        XCTAssertEqual(results.count,1);XCTAssertNil(results[0].evidence)
        driver.callbacks[1]();XCTAssertEqual(results.count,1)
    }
    @MainActor func testChangedContextAndInvalidPlanNeverDispatchOrAttach() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver(),session=MeasuredLearningSession(driver:driver)
        let id=try XCTUnwrap(workspace.activeProjectID)
        var completions=0
        XCTAssertTrue(session.start(workspace:workspace,projectID:id,submission:Data(),runtimeVersionKey:"pytorch",makePlan:{_,_ in throw LearningContractError("Unsupported task")},onComplete:{result in completions+=1;XCTAssertEqual(result.grade.status,.executionFailed)}))
        driver.finish(0,versions:["pytorch":"2.14.1"])
        XCTAssertEqual(completions,1);XCTAssertTrue(driver.plans.isEmpty);XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertTrue(session.start(workspace:workspace,projectID:id,submission:Data(),runtimeVersionKey:"pytorch",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{_ in completions+=1}))
        workspace.activeProjectID=UUID().uuidString
        driver.finish(1,versions:["pytorch":"2.14.1"])
        XCTAssertEqual(completions,1);XCTAssertTrue(driver.plans.isEmpty);XCTAssertFalse(workspace.isAssessingLearning)
    }
    @MainActor func testExecutionCancellationBudgetAndUnbackedSuccessNeverBecomeMeasuredEvidence() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver()
        var seed:UInt64=10
        let session=MeasuredLearningSession(driver:driver,fixtureSeed:{seed+=1;return seed})
        let id=try XCTUnwrap(workspace.activeProjectID)
        var results:[MeasuredLearningResult]=[]
        func start()->Bool {session.start(workspace:workspace,projectID:id,submission:Data(),runtimeVersionKey:"pytorch",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{results.append($0)})}
        XCTAssertTrue(start());driver.finish(0,versions:["pytorch":"2.14.1"])
        session.cancel();session.cancel();XCTAssertEqual(driver.cancellations,1)
        driver.finish(1,status:.passed)
        XCTAssertEqual(results.last?.grade.status,.cancelled);XCTAssertNil(results.last?.evidence)
        XCTAssertTrue(start());driver.finish(2,versions:["pytorch":"2.14.1"]);driver.finish(3,status:.budgetExceeded)
        XCTAssertEqual(results.last?.grade.status,.budgetExceeded);XCTAssertNil(results.last?.evidence)
        XCTAssertEqual(driver.plans.map(\.fixtureSeed),[11,12])
        XCTAssertTrue(start());driver.finish(4,versions:["pytorch":"2.14.1"]);driver.finish(5,status:.passed)
        XCTAssertEqual(results.last?.grade.status,.executionFailed);XCTAssertNil(results.last?.evidence)
        XCTAssertFalse(workspace.isAssessingLearning)
    }
    @MainActor func testMissingOrContradictoryRuntimeCannotDispatchAssessment() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver(),session=MeasuredLearningSession(driver:driver)
        let id=try XCTUnwrap(workspace.activeProjectID)
        var statuses:[LearningGradeStatus]=[]
        XCTAssertTrue(session.start(workspace:workspace,projectID:id,submission:Data(),runtimeVersionKey:"numpy",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{statuses.append($0.grade.status)}))
        driver.finish(0,versions:["pytorch":"2.14.1"])
        XCTAssertEqual(statuses,[.executionFailed]);XCTAssertTrue(driver.plans.isEmpty)
        XCTAssertTrue(session.start(workspace:workspace,projectID:id,submission:Data(),runtimeVersionKey:"numpy",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{statuses.append($0.grade.status)}))
        driver.finish(1,versions:["numpy":"2.2.6"])
        XCTAssertEqual(statuses,[.executionFailed,.executionFailed]);XCTAssertTrue(driver.plans.isEmpty)
        XCTAssertFalse(workspace.isAssessingLearning)
    }

    @MainActor func testCoordinatorKeepsAcceptedOperationAliveUntilTerminalCleanup() throws {
        let folder=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:folder)}
        let workspace=try workspace(folder),driver=ControlledAssessmentDriver()
        var session:MeasuredLearningSession?=MeasuredLearningSession(driver:driver)
        weak var weakSession=session
        var finished=false
        XCTAssertTrue(session!.start(workspace:workspace,projectID:try XCTUnwrap(workspace.activeProjectID),submission:Data(),runtimeVersionKey:"pytorch",makePlan:{LearningExamples.affineComposition(runtimeVersion:$0,seed:$1)},onComplete:{_ in finished=true}))
        session=nil
        XCTAssertNotNil(weakSession)
        workspace.cancelRun();driver.finish(0,status:.cancelled)
        XCTAssertTrue(finished);XCTAssertFalse(workspace.isAssessingLearning)
        XCTAssertNil(weakSession)
    }

}
