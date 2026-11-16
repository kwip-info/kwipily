import XCTest
import KwipMLCore
@testable import KwipMLApp

@MainActor final class WorkflowSupervisedSessionTests:XCTestCase {
    private final class Harness {
        var arguments:[String]=[]
        var line:((String)->Void)?
        var exit:((Int32,String)->Void)?
        var cancellations=0,releases=0
        var outcomes:[WorkflowExecutionOutcome]=[]
        var synchronous:String?
        var launch:WorkflowWorkerLauncher {{[self] _,args,_,line,exit in
            arguments=args;self.line=line;self.exit=exit
            if let synchronous {line(synchronous)}
            return .init(cancel:{[self] in cancellations+=1})
        }}
        var nonce:String {arguments[arguments.firstIndex(of:"--kwip-supervise")!+2]}
        func ready(nonce:String?=nil,phase:String="preparation")->String {"{\"event\":\"launchReady\",\"nonce\":\"\(nonce ?? self.nonce)\",\"phase\":\"\(phase)\"}"}
    }
    private func fixture(_ h:Harness,includeGroupOwner:Bool=true)throws->(WorkflowExecutionSession,WorkflowGroupOwner) {
        let helper=WorkflowExecutionRequestTests(),(root,authoring,runtime)=try helper.setup()
        addTeardownBlock {try? FileManager.default.removeItem(at:root)}
        let spec=WorkflowExperimentSpecification(axes:[.init(field:"seed",values:[7,8])],budget:.init(maximumRuns:2,maximumTotalEpochs:6))
        XCTAssertTrue(authoring.setExperimentSpecification(spec,for:.init(domain:.control,localID:"setup"),lease:authoring.lease()))
        try WorkflowGuardEvidenceFixture.stage(runtime)
        let base=try helper.capture(authoring,runtime),owned=try WorkflowGroupOwnership.createGroup(workspace:root,projectID:authoring.projectID)
        let inputs=try WorkflowExperimentInputs.captureBytes(addresses:base.plan.sourceAddresses,directory:owned.directory) {_ in Data("a,b,label\n1,2,0\n".utf8)}
        let requests=try base.experimentVariants(inputs:inputs,owner:authoring),group=try WorkflowExperimentGroupStore.create(requests:requests,directory:owned.directory,groupOwner:owned.owner)
        let directoryID=UUID(),directory=root.appendingPathComponent("workflow-runs/\(authoring.projectID)/\(directoryID.uuidString)")
        let session=WorkflowExecutionSession(request:requests[0],owner:authoring,sources:inputs.sources,directory:directory,reservation:.init(acquire:{UUID()},release:{_ in h.releases+=1}),groupOwner:includeGroupOwner ? owned.owner:nil,launch:h.launch,onTerminal:{h.outcomes.append($0)})
        try group.begin(ordinal:1,runID:session.runID,directoryID:directoryID)
        return(session,owned.owner)
    }
    private func launched(_ session:WorkflowExecutionSession,_ h:Harness)async throws {
        XCTAssertTrue(session.start(),session.statusMessage)
        let deadline=ProcessInfo.processInfo.systemUptime+5
        while h.exit == nil && ProcessInfo.processInfo.systemUptime<deadline {try await Task.sleep(nanoseconds:1_000_000)}
        XCTAssertNotNil(h.exit,session.statusMessage)
        XCTAssertTrue(h.arguments.contains("--kwip-supervise"))
        XCTAssertFalse(h.arguments.contains("--execute-dataflow"),"Worker arguments live in the immutable intent, not supervisor invocation.")
    }
    func testGroupedRequestCannotLaunchWithoutOwner()throws {
        let h=Harness(),(session,owner)=try fixture(h,includeGroupOwner:false);defer {owner.close()}
        XCTAssertFalse(session.start())
        XCTAssertTrue(h.arguments.isEmpty)
        XCTAssertTrue(session.statusMessage.contains("owner"))
    }
    func testPrematureProgressFailsCancelsAndRetainsUntilExit()async throws {
        let h=Harness(),(session,owner)=try fixture(h);defer {owner.close()}
        try await launched(session,h)
        h.line?("{\"event\":\"dataflowCompleted\"}")
        XCTAssertEqual(h.cancellations,1);XCTAssertTrue(h.outcomes.isEmpty);XCTAssertEqual(h.releases,0)
        h.line?(h.ready());h.exit?(0,"")
        XCTAssertEqual(h.outcomes.first?.status,.failed);XCTAssertEqual(h.releases,1)
        XCTAssertTrue(h.outcomes.first?.error?.contains("ready") == true)
    }
    func testWrongDuplicateAndMissingReadyCannotSucceed()async throws {
        for scenario in ["wrong","wrongPhase","extraKeys","duplicate","missing"] {
            let h=Harness(),(session,owner)=try fixture(h);defer {owner.close()}
            try await launched(session,h)
            if scenario == "wrong" {h.line?(h.ready(nonce:UUID().uuidString))}
            if scenario == "wrongPhase" {h.line?(h.ready(phase:"training"))}
            if scenario == "extraKeys" {h.line?(h.ready().dropLast()+",\"extra\":true}")}
            if scenario == "duplicate" {h.line?(h.ready());h.line?(h.ready())}
            XCTAssertTrue(h.outcomes.isEmpty)
            h.exit?(0,"")
            XCTAssertEqual(h.outcomes.first?.status,.failed,scenario)
            XCTAssertTrue(h.outcomes.first?.error?.lowercased().contains("ready") == true,scenario)
        }
    }
    func testSynchronousProtocolFailureCancelsAssignedHandleAndKeepsFirstError()async throws {
        let h=Harness();h.synchronous="{\"event\":\"epoch\",\"epoch\":1}"
        let(session,owner)=try fixture(h);defer {owner.close()}
        try await launched(session,h)
        XCTAssertEqual(h.cancellations,1)
        h.line?("{\"event\":\"error\",\"message\":\"later failure\"}")
        h.exit?(1,"secondary stderr")
        XCTAssertTrue(h.outcomes.first?.error?.contains("ready") == true)
        XCTAssertFalse(h.outcomes.first?.error?.contains("later failure") == true)
    }
    func testStartupErrorAllowedBeforeReadyAndStaleEventsIgnoredAfterExit()async throws {
        let h=Harness(),(session,owner)=try fixture(h);defer {owner.close()}
        try await launched(session,h)
        h.line?("{\"event\":\"error\",\"message\":\"bootstrap rejected\"}");h.exit?(1,"")
        XCTAssertEqual(h.outcomes.first?.error,"bootstrap rejected")
        h.line?(h.ready());h.exit?(0,"")
        XCTAssertEqual(h.outcomes.count,1);XCTAssertEqual(h.releases,1)
    }
}
