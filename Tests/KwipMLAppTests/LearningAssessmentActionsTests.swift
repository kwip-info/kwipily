import XCTest
@testable import KwipMLApp
@testable import KwipMLCore
final class LearningAssessmentActionsTests:XCTestCase {
    @MainActor func testAsyncAdmissionCancellationStaysBusyUntilAcknowledgedAndNeverAttachesLateHandle() async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let adapter=LearningAssessmentAdapter()
        let policy=try ResolvedExecutionBudget.resolve(global:.init(maximumRunSeconds:120),projectMaximumRunSeconds:nil)
        try adapter.configureExecution(workspace:directory,projectID:UUID().uuidString,budget:policy,storageLimitBytes:9_000_000)
        var continuation:CheckedContinuation<Void,Never>?
        var exits:((Int32,String)->Void)?
        var cancellations=0,completions=0
        adapter.completion={completions+=1}
        adapter.launchTool={ _,_,_,window,mode,_,maximum,limit,_,onExit in
            XCTAssertEqual(window.policy.maximumRunSeconds,65);XCTAssertEqual(mode,.learningRuntime)
            XCTAssertEqual(maximum,10);XCTAssertEqual(limit,9_000_000)
            exits=onExit
            await withCheckedContinuation {continuation=$0}
            return WorkflowWorkerHandle(cancel:{cancellations+=1})
        }
        adapter.discoverRuntime(runtime)
        for _ in 0..<100 where continuation == nil {try await Task.sleep(nanoseconds:1_000_000)}
        XCTAssertNotNil(continuation);adapter.cancel();adapter.cancel()
        XCTAssertTrue(adapter.isRunning);XCTAssertEqual(completions,0)
        continuation?.resume()
        for _ in 0..<100 where cancellations == 0 {try await Task.sleep(nanoseconds:1_000_000)}
        XCTAssertEqual(cancellations,1);XCTAssertTrue(adapter.isRunning)
        exits?(143,"")
        XCTAssertFalse(adapter.isRunning);XCTAssertEqual(adapter.grade?.status,.cancelled);XCTAssertEqual(completions,1)
        exits?(0,"");XCTAssertEqual(completions,1)
    }

    @MainActor func testStorageRefusalBeforeDiscoveryCreatesNoOperationAndReleasesBusy() async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let directory=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:directory)}
        let adapter=LearningAssessmentAdapter()
        try adapter.configureExecution(workspace:directory,projectID:UUID().uuidString,
            budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil),storageLimitBytes:1)
        adapter.discoverRuntime(runtime);try await finished(adapter)
        XCTAssertEqual(adapter.grade?.status,.executionFailed);XCTAssertTrue(adapter.runtimeVersions.isEmpty)
        XCTAssertFalse(FileManager.default.fileExists(atPath:directory.appendingPathComponent("workflow-operations").path))
    }

    @MainActor func testLearningCapabilityClampPreservesCapturedResources() async throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let runtime = try RuntimeLocation.resolve(resources: nil, developmentRoot: root, allowDevelopment: true)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let project = UUID().uuidString
        let resources = try ExecutionResourcePolicy(cpuThreads: 2, maximumResidentBytes: 1_073_741_824)
        let budget = try ResolvedExecutionBudget.resolve(global: .init(maximumRunSeconds: 120),
            projectMaximumRunSeconds: 90, capabilityMaximumSeconds: 75, resources: resources)
        let adapter = LearningAssessmentAdapter()
        try adapter.configureExecution(workspace: directory, projectID: project, budget: budget)
        adapter.discoverRuntime(runtime)
        try await finished(adapter)
        XCTAssertNil(adapter.error)
        XCTAssertNotNil(adapter.runtimeVersions["pytorch"])
        let operations = directory.appendingPathComponent("workflow-operations/" + project)
        let entries = try FileManager.default.contentsOfDirectory(at: operations, includingPropertiesForKeys: nil)
        XCTAssertEqual(entries.count, 1)
        let request = try WorkflowToolRequest.decode(Data(contentsOf: XCTUnwrap(entries.first).appendingPathComponent("request.json")))
        XCTAssertEqual(request.executionBudget.maximumRunSeconds, 65)
        XCTAssertEqual(request.executionBudget.requestedSeconds, 90)
        XCTAssertEqual(request.executionBudget.source, .project)
        XCTAssertEqual(request.executionBudget.resources, resources)
    }
    @MainActor func testActualWorkerEvidenceGradesAffineMapAndWrongParameters()async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:root.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let adapter=LearningAssessmentAdapter()
        try adapter.configureExecution(workspace:dir,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil))
        adapter.discoverRuntime(runtime)
        try await finished(adapter)
        let version=try XCTUnwrap(adapter.runtimeVersions["pytorch"],adapter.error ?? "Missing runtime")
        let plan=LearningExamples.affineComposition(runtimeVersion:version,seed:734)
        let graph=GraphDocument(nodes:[GraphNode(id:"input",type:"input",config:["features":2]),GraphNode(id:"map",type:"linear",config:["in_features":2,"out_features":2]),GraphNode(id:"output",type:"output")],edges:[GraphEdge(source:"input",target:"map"),GraphEdge(source:"map",target:"output")])
        let graphValue=try JSONSerialization.jsonObject(with:JSONEncoder().encode(graph))
        var submission:[String:Any]=["schemaVersion":2,"kind":"affineGraph","graph":graphValue,"parameters":["map":["weight":["shape":[2,2],"values":[3,5,-3.5,5]],"bias":["shape":[2],"values":[-2.75,-3]]]]]
        adapter.assess(plan:plan,submission:try JSONSerialization.data(withJSONObject:submission),runtime:runtime )
        try await finished(adapter)
        XCTAssertNil(adapter.error);XCTAssertEqual(adapter.grade?.status,.passed)
        let legacyPlan=try LearningExamples.affineCompositionForReplay(runtimeVersion:version,seed:734,assessmentVersion:ContractVersion(),contentVersion:ContractVersion())
        var legacySubmission=submission;legacySubmission["schemaVersion"]=1
        adapter.assess(plan:legacyPlan,submission:try JSONSerialization.data(withJSONObject:legacySubmission),runtime:runtime )
        try await finished(adapter)
        XCTAssertEqual(adapter.grade?.status,.passed,adapter.grade?.summary ?? "No legacy grade")
        submission["parameters"]=["map":["weight":["shape":[2,2],"values":[0,0,0,0]],"bias":["shape":[2],"values":[0,0]]]]
        adapter.assess(plan:plan,submission:try JSONSerialization.data(withJSONObject:submission),runtime:runtime )
        try await finished(adapter)
        XCTAssertNil(adapter.error);XCTAssertEqual(adapter.grade?.status,.failed)
        adapter.assess(plan:plan,submission:try JSONSerialization.data(withJSONObject:submission),runtime:runtime )
        adapter.cancel();try await finished(adapter)
        XCTAssertEqual(adapter.grade?.status,.cancelled)
    }
    @MainActor func testActualRetiredNormalizationReportsUnavailableWithoutMintingFixtureEvidence()async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:root.appendingPathComponent(".venv/bin/python").path) else{throw XCTSkip("Development runtime unavailable")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer{try? FileManager.default.removeItem(at:dir)}
        let adapter=LearningAssessmentAdapter();try adapter.configureExecution(workspace:dir,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil));adapter.discoverRuntime(runtime);try await finished(adapter)
        let plan=LearningExamples.trainingOnlyStandardization(runtimeVersion:try XCTUnwrap(adapter.runtimeVersions["numpy"]),seed:918)
        let specs:[(String,String,[String:Any])]=[("s","sourceCSV",[:]),("c","cast",["columns":["x":"float64","y":"float64"]]),("split","split",["seed":42,"validationPercent":25]),("fit","standardScalerFit",["columns":["x","y"]]),("val","transform",[:])]
        let wires=[("s","table","c","table"),("c","table","split","table"),("split","training","fit","training"),("split","validation","val","table"),("fit","state","val","state")]
        let graph:[String:Any]=["schemaVersion":1,"nodes":specs.map{["id":$0.0,"operator":"kwip.data."+$0.1,"config":$0.2]},"edges":wires.map{["from":["node":$0.0,"port":$0.1],"to":["node":$0.2,"port":$0.3]]}]
        let submission:[String:Any]=["schemaVersion":1,"kind":"normalizationGraph","graph":graph,"outputs":["state":["node":"fit","port":"state"],"validation":["node":"val","port":"table"],"training":["node":"split","port":"training"]]]
        adapter.assess(plan:plan,submission:try JSONSerialization.data(withJSONObject:submission),runtime:runtime )
        try await finished(adapter)
        XCTAssertEqual(plan.fixtures.count,3,"Historical Core fixtures stay readable without authorizing new execution")
        XCTAssertNil(adapter.error);XCTAssertEqual(adapter.grade?.status,.unavailable,adapter.grade?.summary ?? "No grade")
        XCTAssertEqual(adapter.grade?.outcomes.count,0)
        let evidence=try XCTUnwrap(adapter.evidence)
        XCTAssertEqual(evidence.runtime.backend,"numpy")
        XCTAssertEqual(evidence.runtime.version,adapter.runtimeVersions["numpy"])
        XCTAssertTrue(evidence.runtime.capabilities.isEmpty,"The actual worker must not advertise removed fit/transform operators")
        XCTAssertTrue(evidence.fixtures.isEmpty);XCTAssertEqual(evidence.maximumTensorElements,0)
        XCTAssertEqual(evidence.assessmentID,plan.id);XCTAssertEqual(evidence.fixtureDigest,plan.fixtureDigest)
    }
    @MainActor func testTransformPlansGradeActualWorkerValuesAndGradients()async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:root.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:dir)}
        let adapter=LearningAssessmentAdapter();try adapter.configureExecution(workspace:dir,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil));adapter.discoverRuntime(runtime);try await finished(adapter)
        let version=try XCTUnwrap(adapter.runtimeVersions["pytorch"])
        for (operation,configuration) in [("reshape",["shape":JSONValue.array([3,2])]),("transpose",["dim0":1,"dim1":2]),("flatten",["start_dim":1,"end_dim":-1])] {
            let plan=try TensorTransformAssessment.plan(operation:operation,runtimeVersion:version,seed:734)
            var graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"t",type:operation,config:configuration),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"t"),.init(source:"t",target:"o")])
            let submission=try TensorLearningSubmission(graph:graph).encoded()
            adapter.assess(plan:plan,submission:submission,runtime:runtime );try await finished(adapter)
            XCTAssertNil(adapter.error)
            XCTAssertEqual(adapter.grade?.status,.passed,adapter.grade?.summary ?? "No grade")
            XCTAssertEqual(adapter.grade?.outcomes.count,2)
            XCTAssertEqual(adapter.evidence?.submissionDigest,learningHash(submission))
            let projectID=UUID().uuidString
            let receipt=try TensorLearningReceipt(projectID:projectID,operation:operation,mode:.assessment,
                submissionData:submission,plan:plan,evidence:try XCTUnwrap(adapter.evidence))
            let reopened=try TensorLearningReceipt.decode(receipt.encoded())
            XCTAssertEqual(try reopened.verifiedReplay(expectedProjectID:projectID),plan)
            XCTAssertEqual(reopened.attempt.grade,adapter.grade)
            if operation == "transpose" {
                graph.nodes[1].type="reshape";graph.nodes[1].config=["shape":.array([3,2])]
                adapter.assess(plan:plan,submission:try TensorLearningSubmission(graph:graph).encoded(),runtime:runtime );try await finished(adapter)
                XCTAssertNil(adapter.error)
                XCTAssertEqual(adapter.grade?.status,.failed)
                XCTAssertEqual(adapter.grade?.outcomes.filter {!$0.passed}.count,2)
            }
        }
    }
    @MainActor func testSoftmaxPlansGradeMeasuredWorkerOutputsAndGradients()async throws {
        let root=URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        guard FileManager.default.fileExists(atPath:root.appendingPathComponent(".venv/bin/python").path) else {throw XCTSkip("Development runtime unavailable")}
        let runtime=try RuntimeLocation.resolve(resources:nil,developmentRoot:root,allowDevelopment:true)
        let dir=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {try? FileManager.default.removeItem(at:dir)}
        let adapter=LearningAssessmentAdapter();try adapter.configureExecution(workspace:dir,projectID:UUID().uuidString,budget:ResolvedExecutionBudget.resolve(global:.factory,projectMaximumRunSeconds:nil));adapter.discoverRuntime(runtime);try await finished(adapter)
        for axis in [1,-1] {
            let plan=try SoftmaxAssessment.plan(axis:axis,runtimeVersion:XCTUnwrap(adapter.runtimeVersions["pytorch"]),seed:734)
            for correct in [true,false] {
                let selected=correct ? axis:(axis == 1 ? -1:1)
                let graph=GraphDocument(nodes:[.init(id:"i",type:"input"),.init(id:"s",type:"softmax",config:["dim":.integer(Int64(selected))]),.init(id:"o",type:"output")],edges:[.init(source:"i",target:"s"),.init(source:"s",target:"o")])
                let bytes=try TensorLearningSubmission(graph:graph,allowSoftmax:true).encoded()
                adapter.assess(plan:plan,submission:bytes,runtime:runtime );try await finished(adapter)
                XCTAssertNil(adapter.error)
                XCTAssertEqual(adapter.grade?.status,correct ? .passed:.failed,adapter.grade?.summary ?? "No grade")
                XCTAssertEqual(adapter.grade?.outcomes.count,2)
                XCTAssertEqual(adapter.evidence?.submissionDigest,learningHash(bytes))
                let project=UUID().uuidString
                let operation=axis == 1 ? "softmax-first-sample-axis":"softmax-last-axis"
                let receipt=try TensorLearningReceipt(projectID:project,operation:operation,mode:.assessment,submissionData:bytes,plan:plan,evidence:XCTUnwrap(adapter.evidence))
                let restored=try TensorLearningReceipt.decode(receipt.encoded())
                XCTAssertEqual(try restored.verifiedReplay(expectedProjectID:project),plan)
                XCTAssertEqual(restored.attempt.grade,adapter.grade)
            }
        }
    }
    @MainActor private func finished(_ adapter:LearningAssessmentAdapter)async throws {
        let deadline=Date().addingTimeInterval(20)
        while adapter.isRunning && Date()<deadline {try await Task.sleep(nanoseconds:25_000_000)}
        XCTAssertFalse(adapter.isRunning)
    }
}
