import XCTest
import KwipMLCore
@testable import KwipMLApp

final class RecurrentLearningIntegrationTests:XCTestCase {
    private var repository:URL {URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()}
    private func graph(_ task:RecurrentLearningTask)->GraphDocument {
        var nodes=task.sourceRoles.map {GraphNode(id:$0,type:"input",name:$0,config:["fixture":.string($0)])};nodes.append(.init(id:"op",type:task.componentID,name:task.componentTitle,config:task.configuration));nodes+=task.outputRoles.map {GraphNode(id:"output-"+$0,type:"output",name:$0,config:["role":.string($0)])}
        let edges=task.sourceRoles.map {GraphEdge(id:"in-"+$0,source:$0,target:"op",targetPort:$0=="input" ? "in":$0)}+task.outputRoles.map {GraphEdge(id:"out-"+$0,source:"op",sourcePort:$0=="result" ? "out":$0,target:"output-"+$0)}
        return .init(schemaVersion:2,name:"Remember patterns",nodes:nodes,edges:edges)
    }
    @MainActor private func wait(_ w:WorkspaceStore)async throws {let until=Date().addingTimeInterval(30);while w.isAssessingLearning,Date()<until {try await Task.sleep(for:.milliseconds(20))};XCTAssertFalse(w.isAssessingLearning)}
    @MainActor func testEveryTaskNamedStateInfluenceColdReceiptAndRepresentativeModes()async throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),l=TensorLearningStore(workspace:w)
        XCTAssertTrue(l.start(operation:RecurrentLearningTask.rnn.rawValue));XCTAssertNotNil(l.runtimeReadinessError);await verifyRuntimeForIntegration(w)
        for task in RecurrentLearningTask.allCases {
            XCTAssertTrue(l.start(operation:task.rawValue));XCTAssertNil(l.runtimeReadinessError);XCTAssertTrue(l.drafts.update {$0.graph=self.graph(task)})
            // All tasks retain real numerical workers; one task covers all mode lifecycles.
            let workerModes:[LearningMode] = task == .rnn ? [.guided,.practice,.assessment]:[.guided]
            for mode in workerModes {
                XCTAssertTrue(l.setMode(mode));XCTAssertTrue(l.canSubmit,l.statusMessage);l.submit();try await wait(w)
                XCTAssertEqual(l.grade?.status,.passed,"\(task.rawValue) \(mode):\(l.statusMessage)");XCTAssertTrue(l.resultSaved)
                XCTAssertNil(l.formativeProjection);XCTAssertNil(l.optimizerProjection);XCTAssertNil(l.schedulerProjection)
                XCTAssertEqual(l.recurrentProjection==nil,mode == .assessment)
                if let p=l.recurrentProjection {XCTAssertEqual(p.fixtures.count,4);for f in p.fixtures {XCTAssertTrue(Set(task.outputRoles).isSubset(of:f.measured.keys));for key in task.sourceRoles+task.parameterShapes.keys.map({"parameter."+$0}) {XCTAssertEqual(f.measured[key+"Gradient"]?.shape,f.inputs[key]?.shape)}}}
            }
            let id=try XCTUnwrap(w.activeProjectID),doc=w.graph,cold=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),cl=TensorLearningStore(workspace:cold)
            await verifyRuntimeForIntegration(cold);XCTAssertTrue(cold.beginProject(doc,id:id));XCTAssertEqual(cl.grade?.status,.passed);XCTAssertTrue(cl.resultSaved);XCTAssertEqual(cl.recurrentProjection == nil,task == .rnn)
            let savedDrafts=TensorLearningDraftRepository(directory:storage.appendingPathComponent("learning/projects"))
            for mode in [LearningMode.practice,.assessment,.guided] {
                XCTAssertTrue(l.setMode(mode));XCTAssertTrue(l.save())
                XCTAssertNil(l.grade);XCTAssertNil(l.recurrentProjection);XCTAssertFalse(l.resultSaved)
                XCTAssertEqual(try savedDrafts.load(projectID:id).draft.mode,mode)
                if task == .rnn {
                    let reopened=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false)
                    let reopenedLesson=TensorLearningStore(workspace:reopened)
                    XCTAssertTrue(reopened.beginProject(doc,id:id));XCTAssertEqual(reopenedLesson.draft?.mode,mode)
                    if mode == .assessment {XCTAssertNil(reopenedLesson.recurrentProjection)}
                }
            }
        }
    }
    @MainActor func testWrongBiasMeasuresRealFailureThenRepairAndNamedRoleRefusals()async throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),l=TensorLearningStore(workspace:w);await verifyRuntimeForIntegration(w)
        for task in RecurrentLearningTask.allCases {
            XCTAssertTrue(l.start(operation:task.rawValue));var wrong=graph(task);wrong.nodes[task.sourceRoles.count].config["bias"]=false
            XCTAssertTrue(l.drafts.update {$0.graph=wrong});XCTAssertTrue(l.canSubmit);l.submit();try await wait(w);XCTAssertEqual(l.grade?.status,.failed);XCTAssertTrue(l.resultSaved)
            XCTAssertTrue(l.drafts.update {$0.graph=self.graph(task)});XCTAssertFalse(l.resultIsCurrent);XCTAssertNil(l.recurrentProjection);l.submit();try await wait(w);XCTAssertEqual(l.grade?.status,.passed)
            var wrongRole=graph(task);wrongRole.nodes[wrongRole.nodes.count-1].config["role"] = .string("result");XCTAssertFalse(l.drafts.update {$0.graph=wrongRole})
        }
    }
    @MainActor func testUniqueFixtureAndOutputAutoAssignmentNamedPortsAndPendingUndo()throws {
        let storage=FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString);defer {try? FileManager.default.removeItem(at:storage)}
        let w=WorkspaceStore(rootOverride:repository,storageOverride:storage,detectCapabilities:false),l=TensorLearningStore(workspace:w)
        for task in RecurrentLearningTask.allCases {
            XCTAssertTrue(l.start(operation:task.rawValue));let c=try XCTUnwrap(l.canvas)
            XCTAssertEqual(Set(c.registry.components.map(\.id)),["input",task.componentID,"output"])
            for _ in task.sourceRoles {c.addNode(type:"input",x:0,y:0)};c.addNode(type:task.componentID,x:100,y:0);for _ in task.outputRoles {c.addNode(type:"output",x:200,y:0)}
            XCTAssertEqual(Set(c.graph.nodes.filter {$0.type=="input"}.compactMap {$0.config["fixture"]?.stringValue}),Set(task.sourceRoles));XCTAssertEqual(Set(c.graph.nodes.filter {$0.type=="output"}.compactMap {$0.config["role"]?.stringValue}),Set(task.outputRoles))
            let op=try XCTUnwrap(c.graph.nodes.first {$0.type==task.componentID})
            XCTAssertEqual(op.config,try RecurrentConfiguration(componentID:task.componentID,values:task.configuration).values)
            for node in c.graph.nodes where node.type=="input" {let role=try XCTUnwrap(node.config["fixture"]?.stringValue);c.connect(source:node.id,sourcePort:"out",target:op.id,targetPort:role=="input" ? "in":role)}
            for node in c.graph.nodes where node.type=="output" {let role=try XCTUnwrap(node.config["role"]?.stringValue);c.connect(source:op.id,sourcePort:role=="result" ? "out":role,target:node.id,targetPort:"in")}
            XCTAssertTrue(c.validation.isValid,c.statusMessage);XCTAssertEqual(c.validation.outputTensorContracts[.init(op.id,"hidden")]?.layout.sample.dimensions,task.outputSampleShapes["hidden"])
            XCTAssertTrue(c.canvasPortDescription(nodeID:op.id,port:"hidden").contains("float64"))
            XCTAssertFalse(c.canAddType(task.componentID));XCTAssertFalse(c.canAddType("input"));XCTAssertFalse(c.canAddType("output"))
            c.undo();XCTAssertEqual(c.graph.edges.count,task.sourceRoles.count+task.outputRoles.count-1);c.redo();XCTAssertTrue(c.validation.isValid)
            XCTAssertTrue(c.setPending(nodeID:op.id,field:"hidden_size",text:"3"));XCTAssertThrowsError(try l.draft?.submissionData());c.discardPending(nodeID:op.id,field:"hidden_size")
        }
    }
    @MainActor func testSixSearchablePreparedExamplesAndNoCodeAudienceCriterionGuidance()throws {
        let entries=LearnCatalogContent.displayedComponentExampleEntries.filter {RecurrentConfiguration.componentIDs.contains($0.componentID)}
        XCTAssertEqual(entries.count,6);XCTAssertEqual(Set(entries.map(\.componentID)),RecurrentConfiguration.componentIDs)
        for task in RecurrentLearningTask.allCases {
            let entry=try XCTUnwrap(entries.first {$0.componentID==task.componentID});XCTAssertEqual(entry.item.kind,"Example");XCTAssertTrue(entry.item.matches(section:"Examples",search:task.componentTitle))
            let guide=LessonAudienceGuide.make(operation:task.rawValue);XCTAssertEqual(guide.goal,task.title);XCTAssertEqual(guide.steps.count,3);XCTAssertTrue(guide.steps[0].contains("hidden"));XCTAssertTrue(guide.steps[1].contains("Starting state"))
            let requirements=LessonAudienceGuide.requirements(operation:task.rawValue,fallback:"");for h in ["**Build**","**Configure**","**Check**"] {XCTAssertTrue(requirements.contains(h))}
            let plan=try RecurrentAssessment.plan(task:task,runtimeVersion:"2.14.1",seed:41);for criterion in plan.criteria {XCTAssertNotNil(LessonAudienceGuide.criterionFeedback(operation:task.rawValue,criterionID:criterion.id))}
            XCTAssertTrue(guide.terms.contains {$0.name=="State"});XCTAssertTrue(guide.terms.contains {$0.name=="Sequence"})
        }
    }

}
