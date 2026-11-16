import Foundation
import KwipMLCore

@MainActor protocol MeasuredAssessmentDriving:AnyObject {
    var runtimeVersions:[String:String] {get}
    var error:String? {get}
    var grade:LearningGrade? {get}
    var evidence:LearningEvidence? {get}
    var isRunning:Bool {get}
    var completion:(()->Void)? {get set}
    var accountingRoot:URL? {get set}
    func configureExecution(workspace:URL,projectID:String,budget:ResolvedExecutionBudget,storageLimitBytes:Int64?)throws
    func configureCompositionRegistry(_ data:Data?)throws
    func discoverRuntime(_ runtime:RuntimeLocation)
    func assess(plan:LearningAssessmentPlan,submission:Data,runtime:RuntimeLocation)
    func cancel()
}
extension MeasuredAssessmentDriving {
    func configureCompositionRegistry(_ data:Data?)throws {
        guard data == nil else {throw LearningContractError("This assessment driver cannot capture a component registry.")}
    }
}
extension LearningAssessmentAdapter:MeasuredAssessmentDriving {}

struct MeasuredLearningResult {
    let projectID:String
    let plan:LearningAssessmentPlan?
    let grade:LearningGrade
    let evidence:LearningEvidence?
}

/// Shared process lifecycle only. Task controllers retain submission/receipt codecs and durable progress.
@MainActor final class MeasuredLearningSession {
    private enum Phase {case discovery,execution}
    private let driver:any MeasuredAssessmentDriving
    private let fixtureSeed:()->UInt64
    private var generation:UUID?
    private var phase=Phase.discovery
    private var cancelled=false
    private weak var workspace:WorkspaceStore?
    private var projectID:String?
    private var coordinatorToken:UUID?
    private var completion:((MeasuredLearningResult)->Void)?
    private(set) var isRunning=false
    private(set) var startFailureMessage:String?
    init(driver:(any MeasuredAssessmentDriving)?=nil,fixtureSeed:@escaping()->UInt64={UInt64.random(in:1...UInt64.max)}) {
        self.driver=driver ?? LearningAssessmentAdapter();self.fixtureSeed=fixtureSeed
    }
    @discardableResult func start(workspace:WorkspaceStore,projectID:String,submission:Data,runtimeVersionKey:String,
        executionBudget:ResolvedExecutionBudget?=nil,compositionRegistry:Data?=nil,makePlan:@escaping(String,UInt64)throws->LearningAssessmentPlan,
        onComplete:@escaping(MeasuredLearningResult)->Void)->Bool {
        startFailureMessage=nil
        guard !isRunning,!driver.isRunning,submission.count<=2_000_000,!runtimeVersionKey.isEmpty,
              let runtime=workspace.runtime,
              // Coordinator retains the accepted session until finish releases its token.
              let token=workspace.learningProjects.beginAssessment(projectID:projectID,cancel:{self.cancel()},onAdmissionFailure:{self.startFailureMessage=$0}) else {return false}
        let operation=UUID()
        self.workspace=workspace;self.projectID=projectID;coordinatorToken=token;completion=onComplete
        generation=operation;phase = .discovery;cancelled=false;isRunning=true
        do {
            let saved = try workspace.executionPolicy(for: projectID, capabilityMaximumSeconds: 65)
            let policy = executionBudget ?? saved.budget
            try driver.configureCompositionRegistry(compositionRegistry)
            driver.accountingRoot = workspace.applicationDirectory
            try driver.configureExecution(workspace: workspace.workspaceDirectory, projectID: projectID, budget: policy, storageLimitBytes: saved.storageLimitBytes)
        } catch { finish(plan:nil,grade:failure(error.localizedDescription),evidence:nil);return false }
        driver.completion = {[weak self] in
            guard let self,self.generation == operation,self.phase == .discovery else {return}
            guard self.contextIsCurrent else {self.finish(plan:nil,grade:self.failure("Lesson context changed during runtime discovery."),evidence:nil);return}
            if self.cancelled || self.driver.grade?.status == .cancelled {self.finish(plan:nil,grade:self.cancelledGrade,evidence:nil);return}
            guard self.driver.grade == nil,let version=self.driver.runtimeVersions[runtimeVersionKey],!version.isEmpty else {
                self.finish(plan:nil,grade:self.driver.grade ?? self.failure(self.driver.error ?? "Runtime discovery returned no \(runtimeVersionKey) version."),evidence:nil);return
            }
            do {
                let plan=try makePlan(version,self.fixtureSeed())
                try plan.validate()
                guard plan.runtime.backend == runtimeVersionKey,plan.runtime.version == version else {throw LearningContractError("Lesson plan runtime differs from the discovered backend.")}
                self.phase = .execution
                self.driver.completion = {[weak self] in
                    guard let self,self.generation == operation,self.phase == .execution else {return}
                    if self.cancelled || self.driver.grade?.status == .cancelled {self.finish(plan:plan,grade:self.cancelledGrade,evidence:nil);return}
                    var grade=self.driver.grade ?? self.failure("Worker completed without a grade.")
                    var evidence=self.driver.evidence
                    if [.passed,.failed].contains(grade.status),evidence == nil {grade=self.failure("Worker completed without measured evidence.")}
                    if ![.passed,.failed].contains(grade.status) {evidence=nil}
                    self.finish(plan:plan,grade:grade,evidence:evidence)
                }
                self.driver.assess(plan:plan,submission:submission,runtime:runtime)
            } catch {self.finish(plan:nil,grade:self.failure(error.localizedDescription),evidence:nil)}
        }
        driver.discoverRuntime(runtime)
        return true
    }
    func cancel() {guard isRunning,!cancelled else {return};cancelled=true;driver.cancel()}
    private var contextIsCurrent:Bool {workspace?.activeProjectID == projectID && workspace?.lessonContext?.projectID == projectID && workspace?.lessonContext?.restoreState == .ready}
    private func failure(_ message:String)->LearningGrade {LearningGrade(status:.executionFailed,summary:message)}
    private var cancelledGrade:LearningGrade {LearningGrade(status:.cancelled,summary:"Assessment cancelled; no correctness judgment recorded.")}
    private func finish(plan:LearningAssessmentPlan?,grade:LearningGrade,evidence:LearningEvidence?) {
        guard isRunning,let token=coordinatorToken,let projectID else {return}
        let attach=contextIsCurrent,callback=completion
        generation=nil;isRunning=false;coordinatorToken=nil;completion=nil;driver.completion=nil
        workspace?.learningProjects.finishAssessment(token:token)
        if attach {callback?(MeasuredLearningResult(projectID:projectID,plan:plan,grade:grade,evidence:evidence))}
    }
}
